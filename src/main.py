import os
import torch
import argparse
import logging
from pathlib import Path

import torchvision.transforms as transforms
from transformers import AutoTokenizer

from config import Config, get_default_config, save_config, load_config
from models.captioning_model import ImageCaptioningModel
from data.dataset import build_coco_dataloaders
from train.trainer import CaptioningTrainer
from evaluate.metrics import evaluate_model_on_coco


def main():
    """Main entry point for training or inference."""
    parser = argparse.ArgumentParser(
        description="Image Captioning with Transformers")

    # Mode
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "demo"],
                        help="Mode: train, eval, or demo")

    # Config
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (default: use default config)")
    parser.add_argument("--save_config", type=str, default=None,
                        help="Path to save config file")

    # Training options
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for logs and checkpoints")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate for training")

    # Model options
    parser.add_argument("--encoder_type", type=str, default=None,
                        choices=["resnet", "vit", "swin", "clip"],
                        help="Type of visual encoder to use")
    parser.add_argument("--decoder_type", type=str, default=None,
                        choices=["lstm", "transformer", "gpt2"],
                        help="Type of caption decoder to use")
    parser.add_argument("--attention_type", type=str, default=None,
                        choices=["soft", "multi_head", "adaptive", "aoa"],
                        help="Type of attention mechanism to use")
    parser.add_argument("--use_rl", action="store_true",
                        help="Use reinforcement learning")

    # Data options
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root directory for data")

    # Demo options
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to image for demo")

    # Parse arguments
    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()

    # Override config with command line arguments
    _update_config_from_args(config, args)

    # Save config if requested
    if args.save_config:
        save_config(config, args.save_config)

    # Set up logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    # Set device
    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Choose mode
    if args.mode == "train":
        train(config, args.checkpoint, device)
    elif args.mode == "eval":
        evaluate(config, args.checkpoint, device)
    elif args.mode == "demo":
        if not args.image_path:
            parser.error("--image_path is required for demo mode")
        demo(config, args.checkpoint, args.image_path, device)


def _update_config_from_args(config, args):
    """Update config with command line arguments."""
    # Training options
    if args.output_dir:
        config.output_dir = args.output_dir
        config.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate

    # Model options
    if args.encoder_type:
        config.model.encoder.encoder_type = args.encoder_type
    if args.decoder_type:
        config.model.decoder.decoder_type = args.decoder_type
    if args.attention_type:
        config.model.attention.attention_type = args.attention_type
    if args.use_rl:
        config.training.use_rl = True

    # Data options
    if args.data_root:
        config.data_root = args.data_root


def train(config, checkpoint_path=None, device="cuda"):
    """Train the model."""
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

    # Set up image transforms
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.decoder.pretrained_model_name)

    # Special tokens handling
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Update config with tokenizer info
    config.model.vocab_size = len(tokenizer)
    config.model.pad_token_id = tokenizer.pad_token_id
    config.model.bos_token_id = tokenizer.bos_token_id if hasattr(
        tokenizer, 'bos_token_id') else tokenizer.cls_token_id
    config.model.eos_token_id = tokenizer.eos_token_id

    # Load data
    logger.info("Loading datasets...")
    train_loader, val_loader, curriculum_sampler = build_coco_dataloaders(
        config,
        tokenizer,
        transform_train=transform_train,
        transform_val=transform_val
    )

    # Create model
    logger.info("Creating model...")
    model = ImageCaptioningModel(config, tokenizer)

    # Create trainer
    trainer = CaptioningTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        curriculum_sampler=curriculum_sampler
    )

    # Load checkpoint if provided
    if checkpoint_path:
        trainer.load_checkpoint(checkpoint_path)

    # Train
    trainer.train()


def evaluate(config, checkpoint_path=None, device="cuda"):
    """Evaluate the model."""
    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation...")

    if not checkpoint_path:
        logger.error("Checkpoint path is required for evaluation")
        return

    # Set up image transforms
    transform_eval = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.decoder.pretrained_model_name)

    # Special tokens handling
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Update config with tokenizer info
    config.model.vocab_size = len(tokenizer)
    config.model.pad_token_id = tokenizer.pad_token_id
    config.model.bos_token_id = tokenizer.bos_token_id if hasattr(
        tokenizer, 'bos_token_id') else tokenizer.cls_token_id
    config.model.eos_token_id = tokenizer.eos_token_id

    # Load data
    logger.info("Loading datasets...")
    _, val_loader, _ = build_coco_dataloaders(
        config,
        tokenizer,
        transform_val=transform_eval
    )

    # Create model
    logger.info("Creating model...")
    model = ImageCaptioningModel(config, tokenizer)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Evaluate
    logger.info("Evaluating...")
    metrics = evaluate_model_on_coco(
        model=model,
        coco_dataloader=val_loader,
        tokenizer=tokenizer,
        device=device,
        annotation_file=os.path.join(config.data_root, config.val_json)
    )

    # Log results
    logger.info("Evaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")


def demo(config, checkpoint_path=None, image_path=None, device="cuda"):
    """Run a demo with the model on a single image."""
    import matplotlib.pyplot as plt
    from PIL import Image

    logger = logging.getLogger(__name__)
    logger.info("Starting demo...")

    if not checkpoint_path:
        logger.error("Checkpoint path is required for demo")
        return

    if not image_path:
        logger.error("Image path is required for demo")
        return

    # Set up image transforms
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.decoder.pretrained_model_name)

    # Special tokens handling
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Update config with tokenizer info
    config.model.vocab_size = len(tokenizer)
    config.model.pad_token_id = tokenizer.pad_token_id
    config.model.bos_token_id = tokenizer.bos_token_id if hasattr(
        tokenizer, 'bos_token_id') else tokenizer.cls_token_id
    config.model.eos_token_id = tokenizer.eos_token_id

    # Create model
    logger.info("Creating model...")
    model = ImageCaptioningModel(config, tokenizer)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load and process image
    logger.info(f"Loading image from {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Generate caption
    logger.info("Generating caption...")
    with torch.no_grad():
        captions, _ = model.generate(
            images=image_tensor,
            max_length=config.inference.max_length,
        )

    # Decode caption
    caption = tokenizer.decode(captions[0], skip_special_tokens=True)

    # Display results
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title(caption)
    plt.axis('off')
    plt.show()

    logger.info(f"Generated caption: {caption}")


if __name__ == "__main__":
    main()
