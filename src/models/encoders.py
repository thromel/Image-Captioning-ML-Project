import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, Union, List

from transformers import (
    ViTModel,
    SwinModel,
    CLIPVisionModel,
    ResNetModel,
    AutoImageProcessor
)

from ..config import EncoderType, EncoderConfig


class ImageEncoder(nn.Module, ABC):
    """Base class for all image encoders."""

    @abstractmethod
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process images and return features.

        Args:
            images: Tensor of images of shape [batch_size, channels, height, width]

        Returns:
            Dict containing:
                - features: Image features [batch_size, feature_length, feature_dim]
                - pooled_features: Global image representation [batch_size, feature_dim]
                - attention_mask: Attention mask for features [batch_size, feature_length]
        """
        pass


class ResNetEncoder(ImageEncoder):
    """ResNet encoder using HuggingFace's implementation."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        model_name = config.pretrained_model_name
        if not model_name:
            model_name = "microsoft/resnet-50"  # Default to ResNet-50

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ResNetModel.from_pretrained(model_name)
        self.feature_dim = config.feature_dim

        if self.model.config.hidden_sizes[-1] != self.feature_dim:
            self.proj = nn.Linear(
                self.model.config.hidden_sizes[-1], self.feature_dim)
        else:
            self.proj = nn.Identity()

        if config.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # The ResNet model from transformers returns pooled output by default
        outputs = self.model(images)

        # Process features to get spatial features
        # This depends on the specific ResNet implementation
        # For the transformers ResNet, we take the last_hidden_state
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        else:
            # Fallback to feature extraction from the final feature map
            # This is implementation specific and may need to be adjusted
            features = outputs.hidden_states[-1]
            features = features.reshape(features.size(
                0), features.size(1), -1).permute(0, 2, 1)

        # Project features to the desired dimension
        features = self.proj(features)

        # Create a pooled feature (for global representation)
        pooled_features = outputs.pooler_output if hasattr(
            outputs, 'pooler_output') else features.mean(dim=1)

        # Create an attention mask (all ones for now as ResNet doesn't pad)
        attention_mask = torch.ones(
            features.shape[0], features.shape[1], device=features.device)

        return {
            "features": features,
            "pooled_features": pooled_features,
            "attention_mask": attention_mask
        }


class ViTEncoder(ImageEncoder):
    """Vision Transformer encoder."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        model_name = config.pretrained_model_name
        if not model_name:
            model_name = "google/vit-base-patch16-224"

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.feature_dim = config.feature_dim

        # Add projection layer if dimensions don't match
        if self.model.config.hidden_size != self.feature_dim:
            self.proj = nn.Linear(
                self.model.config.hidden_size, self.feature_dim)
        else:
            self.proj = nn.Identity()

        if config.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.model(images, output_hidden_states=True)

        # Get patch features (excluding CLS token)
        features = outputs.last_hidden_state[:, 1:, :]
        features = self.proj(features)

        # Get pooled feature (CLS token)
        pooled_features = outputs.pooler_output
        pooled_features = self.proj(pooled_features)

        # Create attention mask (all ones for ViT since it doesn't pad)
        attention_mask = torch.ones(
            features.shape[0], features.shape[1], device=features.device)

        return {
            "features": features,
            "pooled_features": pooled_features,
            "attention_mask": attention_mask
        }


class SwinEncoder(ImageEncoder):
    """Swin Transformer encoder."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        model_name = config.pretrained_model_name
        if not model_name:
            model_name = "microsoft/swin-base-patch4-window7-224"

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SwinModel.from_pretrained(model_name)
        self.feature_dim = config.feature_dim

        # Add projection layer if dimensions don't match
        if self.model.config.hidden_size != self.feature_dim:
            self.proj = nn.Linear(
                self.model.config.hidden_size, self.feature_dim)
        else:
            self.proj = nn.Identity()

        if config.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.model(images, output_hidden_states=True)

        # For Swin, we take the last hidden state
        features = outputs.last_hidden_state
        features = self.proj(features)

        # Get pooled feature (mean of all patch features)
        pooled_features = features.mean(dim=1)

        # Create attention mask (all ones for Swin since it doesn't pad)
        attention_mask = torch.ones(
            features.shape[0], features.shape[1], device=features.device)

        return {
            "features": features,
            "pooled_features": pooled_features,
            "attention_mask": attention_mask
        }


class CLIPEncoder(ImageEncoder):
    """CLIP Vision encoder."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        model_name = config.pretrained_model_name
        if not model_name:
            model_name = "openai/clip-vit-base-patch32"

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.feature_dim = config.feature_dim

        # Add projection layer if dimensions don't match
        if self.model.config.hidden_size != self.feature_dim:
            self.proj = nn.Linear(
                self.model.config.hidden_size, self.feature_dim)
        else:
            self.proj = nn.Identity()

        if config.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.model(images, output_hidden_states=True)

        # Get patch features (excluding CLS token)
        features = outputs.last_hidden_state[:, 1:, :]
        features = self.proj(features)

        # Get pooled feature
        # CLIP doesn't use the CLS token in the same way, so use the mean of the patch features
        pooled_features = outputs.pooler_output if hasattr(
            outputs, 'pooler_output') else features.mean(dim=1)
        pooled_features = self.proj(pooled_features)

        # Create attention mask (all ones for CLIP since it doesn't pad)
        attention_mask = torch.ones(
            features.shape[0], features.shape[1], device=features.device)

        return {
            "features": features,
            "pooled_features": pooled_features,
            "attention_mask": attention_mask
        }


class ObjectRegionEncoder(ImageEncoder):
    """
    Encoder that uses pre-extracted object region features.
    This is for Bottom-Up Top-Down Attention (Anderson et al., 2018).
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.feature_dim = config.feature_dim

        # In practice, this would interface with a detector or load pre-extracted features
        # For simplicity, we'll just define the projection
        self.input_dim = 2048  # Typical dimension from Faster R-CNN with ResNet backbone

        if self.input_dim != self.feature_dim:
            self.proj = nn.Linear(self.input_dim, self.feature_dim)
        else:
            self.proj = nn.Identity()

        # Additional layers for processing geometric features
        self.geo_dim = 4  # x, y, width, height
        self.geo_proj = nn.Sequential(
            nn.Linear(self.geo_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.feature_dim)
        )
        self.combine = nn.Linear(self.feature_dim * 2, self.feature_dim)

    def forward(self, features_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Process pre-extracted object region features.

        Args:
            features_dict: Dict containing:
                - region_features: Object region features [batch_size, num_regions, input_dim]
                - region_boxes: Bounding boxes [batch_size, num_regions, 4] (x, y, width, height)
                - region_mask: Attention mask [batch_size, num_regions]
        """
        region_features = features_dict["region_features"]
        region_boxes = features_dict["region_boxes"]
        attention_mask = features_dict["region_mask"]

        # Project object features
        features = self.proj(region_features)

        # Process geometric features if provided
        if region_boxes is not None:
            geo_features = self.geo_proj(region_boxes)
            features = self.combine(
                torch.cat([features, geo_features], dim=-1))

        # Create pooled features (mean of region features with mask applied)
        # Applying mask to avoid computing mean with padding
        expanded_mask = attention_mask.unsqueeze(-1).expand_as(features)
        masked_features = features * expanded_mask
        sum_features = masked_features.sum(dim=1)
        sum_mask = expanded_mask.sum(dim=1)
        pooled_features = sum_features / (sum_mask + 1e-10)

        return {
            "features": features,
            "pooled_features": pooled_features,
            "attention_mask": attention_mask
        }


def build_encoder(config: EncoderConfig) -> ImageEncoder:
    """Factory function to build the encoder based on configuration."""
    if config.encoder_type == EncoderType.RESNET:
        return ResNetEncoder(config)
    elif config.encoder_type == EncoderType.VIT:
        return ViTEncoder(config)
    elif config.encoder_type == EncoderType.SWIN:
        return SwinEncoder(config)
    elif config.encoder_type == EncoderType.CLIP:
        return CLIPEncoder(config)
    elif config.use_object_features:
        return ObjectRegionEncoder(config)
    else:
        raise ValueError(f"Unsupported encoder type: {config.encoder_type}")
