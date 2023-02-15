import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any


class EncoderType(Enum):
    RESNET = "resnet"
    VIT = "vit"
    SWIN = "swin"
    CONVNEXT = "convnext"
    EFFICIENTNET = "efficientnet"
    CLIP = "clip"


class DecoderType(Enum):
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    GPT2 = "gpt2"
    T5 = "t5"
    BART = "bart"


class AttentionType(Enum):
    SOFT = "soft"  # Original soft attention
    MULTI_HEAD = "multi_head"
    ADAPTIVE = "adaptive"  # Adaptive attention with visual sentinel
    AOA = "aoa"  # Attention on Attention
    OBJECT = "object"  # Object-level attention


@dataclass
class EncoderConfig:
    encoder_type: EncoderType = EncoderType.VIT
    pretrained_model_name: str = "google/vit-base-patch16-224"
    freeze: bool = False
    feature_dim: int = 768  # Output feature dimension
    use_object_features: bool = False  # Whether to use object region features


@dataclass
class DecoderConfig:
    decoder_type: DecoderType = DecoderType.GPT2
    pretrained_model_name: str = "gpt2"
    hidden_dim: int = 768
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_length: int = 50  # Maximum caption length


@dataclass
class AttentionConfig:
    attention_type: AttentionType = AttentionType.MULTI_HEAD
    num_heads: int = 8
    temperature: float = 1.0
    # Whether to use geometric information for object features
    use_geometric: bool = False


@dataclass
class TrainingConfig:
    # Basic training parameters
    batch_size: int = 64
    num_epochs: int = 15
    learning_rate: float = 5e-5
    weight_decay: float = 0.01

    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # linear, cosine, or step
    warmup_steps: int = 2000

    # Reinforcement learning settings
    use_rl: bool = True
    rl_start_epoch: int = 10
    rl_reward: str = "cider"  # cider, bleu, meteor, etc.
    rl_weight: float = 1.0

    # Mixed precision training
    use_amp: bool = True  # Use automatic mixed precision

    # Curriculum learning
    use_curriculum: bool = False
    # caption_length, num_objects, clip_score
    curriculum_strategy: str = "caption_length"

    # Additional auxiliary losses
    use_contrastive_loss: bool = False  # CLIP-style contrastive loss
    use_itm_loss: bool = False  # Image-Text Matching loss
    use_obj_cls_loss: bool = False  # Object classification auxiliary loss


@dataclass
class InferenceConfig:
    decoding_strategy: str = "beam"  # greedy, beam, nucleus
    beam_size: int = 5
    top_p: float = 0.9  # For nucleus sampling
    temperature: float = 1.0
    min_length: int = 5
    max_length: int = 20
    length_penalty: float = 0.8

    # For diverse beam search
    num_beam_groups: int = 1  # 1 means standard beam search
    diversity_penalty: float = 0.5

    # Reranking
    use_clip_reranking: bool = False
    num_candidates: int = 5  # Number of candidates to generate for reranking


@dataclass
class ModelConfig:
    encoder: EncoderConfig = EncoderConfig()
    decoder: DecoderConfig = DecoderConfig()
    attention: AttentionConfig = AttentionConfig()
    projection_dim: int = 768  # Common projection dimension for encoder-decoder
    use_q_former: bool = False  # Whether to use BLIP-2 style Q-Former
    q_former_num_queries: int = 32  # Number of queries in Q-Former

    vocab_size: int = 50257  # GPT2 vocabulary size, will be overridden based on tokenizer
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    inference: InferenceConfig = InferenceConfig()

    # Data configuration
    data_root: str = "data"
    train_json: str = "annotations/captions_train2014.json"
    val_json: str = "annotations/captions_val2014.json"
    train_image_dir: str = "train2014"
    val_image_dir: str = "val2014"

    # Image preprocessing
    image_size: int = 224

    # Logging and checkpointing
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_every: int = 100
    save_every: int = 1  # Save checkpoint every N epochs

    # Hardware settings
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42


def get_default_config() -> Config:
    """Returns the default configuration."""
    return Config()


def save_config(config: Config, path: str):
    """Save config to a file."""
    import json
    import dataclasses

    def _serialize(obj):
        if dataclasses.is_dataclass(obj):
            return {k: _serialize(v) for k, v in dataclasses.asdict(obj).items()}
        elif isinstance(obj, Enum):
            return obj.value
        return obj

    with open(path, 'w') as f:
        json.dump(_serialize(config), f, indent=2)


def load_config(path: str) -> Config:
    """Load config from a file."""
    import json

    with open(path, 'r') as f:
        config_dict = json.load(f)

    # This is a simple implementation; in practice, you might need
    # more sophisticated parsing to handle enums and nested dataclasses
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    inference_config = InferenceConfig(**config_dict.get('inference', {}))

    config = Config()
    config.model = model_config
    config.training = training_config
    config.inference = inference_config

    # Copy other top-level fields
    for k, v in config_dict.items():
        if k not in ['model', 'training', 'inference']:
            setattr(config, k, v)

    return config
