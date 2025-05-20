import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from ..config import Config
from ..models.captioning_model import ImageCaptioningModel
from ..evaluate.metrics import calculate_metrics


class CaptioningTrainer:
    """Trainer class for image captioning models."""

    def __init__(
        self,
        config: Config,
        model: ImageCaptioningModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer,
        device: str = None
    ):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
            model: The image captioning model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            tokenizer: Tokenizer for converting between IDs and text
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer

        # Set device
        self.device = device or config.device
        if not torch.cuda.is_available() and self.device == 'cuda':
            self.device = 'cpu'
            print("CUDA not available, using CPU instead.")

        # Move model to device
        self.model = self.model.to(self.device)

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Create learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Automatic mixed precision
        self.use_amp = config.training.use_amp
        self.scaler = GradScaler() if self.use_amp else None

        # Output directories
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = Path(config.checkpoint_dir)

        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self._setup_logging()

        # Best validation score for model selection
        self.best_val_score = 0.0

    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _create_optimizer(self):
        """Create optimizer for the model."""
        # Get different parameter groups with different learning rates
        no_decay = ['bias', 'LayerNorm.weight']

        # Parameters with weight decay
        params_with_wd = [
            p for n, p in self.model.named_parameters()
            if not any(nd in n for nd in no_decay) and p.requires_grad
        ]

        # Parameters without weight decay
        params_without_wd = [
            p for n, p in self.model.named_parameters()
            if any(nd in n for nd in no_decay) and p.requires_grad
        ]

        # Create optimizer with parameter groups
        optimizer = optim.AdamW([
            {'params': params_with_wd, 'weight_decay': self.config.training.weight_decay},
            {'params': params_without_wd, 'weight_decay': 0.0}
        ], lr=self.config.training.learning_rate)

        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        # Calculate total steps
        total_steps = len(self.train_loader) * self.config.training.num_epochs

        # Create appropriate scheduler based on config
        if self.config.training.lr_scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.training.warmup_steps,
                num_training_steps=total_steps
            )
        elif self.config.training.lr_scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.training.warmup_steps,
                num_training_steps=total_steps
            )
        else:
            # Default to StepLR
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=total_steps // 3,  # Decay 3 times throughout training
                gamma=0.1
            )

        return scheduler

    def train(self):
        """Train the model."""
        self.logger.info("Starting training...")

        for epoch in range(self.config.training.num_epochs):
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.training.num_epochs}")

            # Train for one epoch
            train_loss = self._train_epoch(epoch)

            # Validate
            val_loss, val_metrics = self._validate_epoch(epoch)

            # Log results
            self.logger.info(f"Epoch {epoch + 1}: "
                             f"Train Loss: {train_loss:.4f}, "
                             f"Val Loss: {val_loss:.4f}, "
                             f"Val CIDEr: {val_metrics['CIDEr']:.4f}")

            # Save checkpoint based on best validation score
            if val_metrics['CIDEr'] > self.best_val_score:
                self.best_val_score = val_metrics['CIDEr']
                self._save_checkpoint(epoch, is_best=True)
                self.logger.info(
                    f"New best model saved with CIDEr: {self.best_val_score:.4f}")

            # Save regular checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)

        # Use tqdm for progress bar
        progress_bar = tqdm(self.train_loader,
                            desc=f"Epoch {epoch + 1} [Train]")

        for i, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with optional mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        images=batch['image'],
                        captions=batch['caption_tokens'],
                        caption_lengths=None
                    )

                    # Calculate loss
                    logits = outputs['logits']
                    targets = batch['caption_tokens']

                    # Shift logits and targets for language modeling task
                    # We predict each next token based on previous tokens
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_targets = targets[..., 1:].contiguous()

                    # Calculate loss
                    loss_fct = nn.CrossEntropyLoss(
                        ignore_index=self.model.decoder.pad_token_id)
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_targets.view(-1)
                    )

                # Backward and optimize with scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training without mixed precision
                outputs = self.model(
                    images=batch['image'],
                    captions=batch['caption_tokens'],
                    caption_lengths=None
                )

                # Calculate loss
                logits = outputs['logits']
                targets = batch['caption_tokens']

                # Shift logits and targets for language modeling task
                shift_logits = logits[..., :-1, :].contiguous()
                shift_targets = targets[..., 1:].contiguous()

                # Calculate loss
                loss_fct = nn.CrossEntropyLoss(
                    ignore_index=self.model.decoder.pad_token_id)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1)
                )

                # Backward and optimize
                loss.backward()
                self.optimizer.step()

            # Update learning rate
            self.scheduler.step()

            # Update progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': epoch_loss / (i + 1)})

            # Log every config.log_every batches
            if (i + 1) % self.config.log_every == 0:
                self.logger.info(f"Epoch {epoch + 1}, Batch {i + 1}/{num_batches}, "
                                 f"Loss: {loss.item():.4f}, "
                                 f"LR: {self.scheduler.get_last_lr()[0]:.6f}")

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / num_batches

        # Implement reinforcement learning if needed
        if self.config.training.use_rl and epoch >= self.config.training.rl_start_epoch:
            self._train_reinforcement_learning(epoch)

        return avg_loss

    def _train_reinforcement_learning(self, epoch: int):
        """
        Train with reinforcement learning (Self-Critical Sequence Training).

        Args:
            epoch: Current epoch number
        """
        # This is a simplified implementation of SCST
        # In practice, you would:
        # 1. Generate captions with the model
        # 2. Calculate rewards (e.g., CIDEr scores) for generated and greedy captions
        # 3. Use REINFORCE algorithm to update model parameters

        self.logger.info(
            f"Running reinforcement learning for epoch {epoch + 1}")

        self.model.train()

        # Use tqdm for progress bar
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [RL]")

        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Zero gradients
            self.optimizer.zero_grad()

            # Sample captions (Monte Carlo)
            sample_captions, sample_logprobs = self._sample_captions(
                batch['image'])

            # Get baseline captions (greedy decoding)
            greedy_captions, _ = self.model.generate(
                images=batch['image'],
                max_length=self.config.inference.max_length
            )

            # Convert to text for reward calculation
            sample_texts = [self.tokenizer.decode(caption, skip_special_tokens=True)
                            for caption in sample_captions]
            greedy_texts = [self.tokenizer.decode(caption, skip_special_tokens=True)
                            for caption in greedy_captions]
            gt_texts = [self.tokenizer.decode(caption, skip_special_tokens=True)
                        for caption in batch['caption_tokens']]

            # Calculate rewards
            sample_rewards = self._calculate_rewards(sample_texts, [gt_texts])
            greedy_rewards = self._calculate_rewards(greedy_texts, [gt_texts])

            # Calculate advantage (A = R - b)
            advantages = sample_rewards - greedy_rewards

            # Calculate reinforce loss
            rl_loss = -torch.mean(sample_logprobs * advantages)

            # Backward and optimize
            rl_loss.backward()
            self.optimizer.step()

            # Update learning rate
            self.scheduler.step()

    def _sample_captions(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample captions from the model for reinforcement learning.

        Args:
            images: Image tensor [batch_size, channels, height, width]

        Returns:
            Tuple of (sampled captions, log probabilities)
        """
        # This is a simplified implementation
        # In practice, we would handle this with a dedicated sampling method in the model
        batch_size = images.size(0)
        max_length = self.config.inference.max_length

        # Encode images
        encoder_features = self.model.encoder(images)

        # Initialize with start token
        input_ids = torch.full(
            (batch_size, 1),
            self.model.decoder.bos_token_id,
            dtype=torch.long,
            device=images.device
        )

        # Storage for log probabilities
        log_probs = []

        # Generate tokens one by one
        for t in range(max_length - 1):
            # Get logits for next token
            outputs = self.model.decoder(
                encoder_features=encoder_features,
                captions=input_ids,
                caption_lengths=None
            )
            logits = outputs['logits'][:, -1, :]  # Last token prediction

            # Sample from logits
            probs = F.softmax(logits, dim=-1)
            next_token_distribution = torch.distributions.Categorical(probs)
            next_token = next_token_distribution.sample()

            # Store log probabilities
            log_prob = next_token_distribution.log_prob(next_token)
            log_probs.append(log_prob)

            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

            # Stop if we've generated EOS tokens for all items in batch
            if (next_token == self.model.decoder.eos_token_id).all():
                break

        return input_ids, torch.stack(log_probs, dim=1)

    def _calculate_rewards(self, generated_captions: List[str], gt_captions: List[List[str]]) -> torch.Tensor:
        """
        Calculate rewards (e.g., CIDEr scores) for generated captions.

        Args:
            generated_captions: List of generated caption strings
            gt_captions: List of lists of ground truth caption strings

        Returns:
            Tensor of rewards for each caption
        """
        # Calculate CIDEr scores using pycocoevalcap
        # This is a simplified implementation
        # In practice, we would use the full evaluation module

        # For now, just return dummy scores
        # Replace this with actual metric calculation
        return torch.randn(len(generated_captions), device=self.device)

    def _validate_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (validation loss, validation metrics)
        """
        self.model.eval()
        epoch_loss = 0.0
        num_batches = len(self.val_loader)

        # Initialize lists for generated captions and references
        generated_captions = []
        reference_captions = []
        image_ids = []

        # Use tqdm for progress bar
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [Val]")

        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # Calculate validation loss
                outputs = self.model(
                    images=batch['image'],
                    # Use first caption for loss
                    captions=batch['caption_tokens'][:, 0, :],
                    caption_lengths=None
                )

                logits = outputs['logits']
                targets = batch['caption_tokens'][:, 0, :]  # First caption

                # Shift logits and targets for language modeling task
                shift_logits = logits[..., :-1, :].contiguous()
                shift_targets = targets[..., 1:].contiguous()

                # Calculate loss
                loss_fct = nn.CrossEntropyLoss(
                    ignore_index=self.model.decoder.pad_token_id)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1)
                )

                epoch_loss += loss.item()

                # Generate captions for evaluation
                gen_captions, _ = self.model.generate(
                    images=batch['image'],
                    max_length=self.config.inference.max_length
                )

                # Convert to text
                gen_texts = [self.tokenizer.decode(caption, skip_special_tokens=True)
                             for caption in gen_captions]

                # Get reference captions
                ref_texts = [[self.tokenizer.decode(caption, skip_special_tokens=True)
                             for caption in batch['caption_tokens'][i]]
                             for i in range(len(batch['image']))]

                # Store for evaluation
                generated_captions.extend(gen_texts)
                reference_captions.extend(ref_texts)
                if 'image_id' in batch:
                    image_ids.extend(batch['image_id'].tolist())

        # Calculate average loss
        avg_loss = epoch_loss / num_batches

        # Calculate metrics
        metrics = calculate_metrics(
            generated_captions, reference_captions, image_ids)

        return avg_loss, metrics

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save a checkpoint of the model.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'best_val_score': self.best_val_score
        }

        # Save regular checkpoint
        torch.save(
            checkpoint,
            self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        )

        # Save best model separately
        if is_best:
            torch.save(
                checkpoint,
                self.checkpoint_dir / "best_model.pth"
            )

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_val_score = checkpoint.get('best_val_score', 0.0)

        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1} "
                         f"with best score {self.best_val_score:.4f}")


def compute_loss(logits, targets, pad_token_id):
    """
    Compute cross-entropy loss for captions.

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        targets: Target token IDs [batch_size, seq_len]
        pad_token_id: ID of padding token to ignore

    Returns:
        Loss value
    """
    # Shift logits and targets for language modeling task
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = targets[..., 1:].contiguous()

    # Calculate loss
    loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1)
    )

    return loss
