"""
Advanced loss functions for image captioning training.

This module implements various auxiliary losses including:
- CLIP-style contrastive loss
- Image-Text Matching (ITM) loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ContrastiveLoss(nn.Module):
    """
    CLIP-style contrastive loss for vision-language alignment.

    This loss encourages the model to align image and text representations
    in a shared embedding space by maximizing similarity between matched
    image-text pairs and minimizing similarity between unmatched pairs.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initialize the contrastive loss.

        Args:
            temperature: Temperature parameter for scaling logits
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            image_features: Image embeddings [batch_size, embed_dim]
            text_features: Text embeddings [batch_size, embed_dim]
            return_logits: If True, also return similarity logits

        Returns:
            Contrastive loss value, optionally with logits
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Compute similarity matrix
        # logits[i, j] = similarity between image i and text j
        logits = torch.matmul(image_features, text_features.T) / self.temperature

        # Create labels - diagonal elements are positive pairs
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=image_features.device)

        # Compute loss in both directions
        # Image-to-text loss
        loss_i2t = F.cross_entropy(logits, labels)

        # Text-to-image loss
        loss_t2i = F.cross_entropy(logits.T, labels)

        # Average the two losses
        loss = (loss_i2t + loss_t2i) / 2

        if return_logits:
            return loss, logits
        return loss


class ImageTextMatchingLoss(nn.Module):
    """
    Image-Text Matching (ITM) loss for binary classification.

    This loss trains a binary classifier to distinguish between matched
    and mismatched image-text pairs, helping the model learn fine-grained
    alignment between visual and textual features.
    """

    def __init__(
        self,
        hidden_dim: int,
        negative_ratio: float = 0.5
    ):
        """
        Initialize the ITM loss.

        Args:
            hidden_dim: Dimension of the hidden features
            negative_ratio: Ratio of negative samples to generate
        """
        super().__init__()
        self.negative_ratio = negative_ratio

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # Binary: matched or mismatched
        )

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        negative_sampling: bool = True
    ) -> torch.Tensor:
        """
        Compute ITM loss.

        Args:
            image_features: Image embeddings [batch_size, embed_dim]
            text_features: Text embeddings [batch_size, embed_dim]
            negative_sampling: If True, generate negative samples

        Returns:
            ITM loss value
        """
        batch_size = image_features.size(0)
        device = image_features.device

        # Positive pairs (matched)
        pos_image_feats = image_features
        pos_text_feats = text_features
        pos_labels = torch.ones(batch_size, dtype=torch.long, device=device)

        if negative_sampling:
            # Generate negative samples by shuffling text features
            num_negatives = int(batch_size * self.negative_ratio)

            # Random shuffle indices
            neg_indices = torch.randperm(batch_size, device=device)[:num_negatives]
            neg_text_indices = torch.roll(neg_indices, 1)

            # Create negative pairs
            neg_image_feats = image_features[neg_indices]
            neg_text_feats = text_features[neg_text_indices]
            neg_labels = torch.zeros(num_negatives, dtype=torch.long, device=device)

            # Combine positive and negative pairs
            all_image_feats = torch.cat([pos_image_feats, neg_image_feats], dim=0)
            all_text_feats = torch.cat([pos_text_feats, neg_text_feats], dim=0)
            all_labels = torch.cat([pos_labels, neg_labels], dim=0)
        else:
            all_image_feats = pos_image_feats
            all_text_feats = pos_text_feats
            all_labels = pos_labels

        # Concatenate image and text features
        combined_features = torch.cat([all_image_feats, all_text_feats], dim=-1)

        # Classify
        logits = self.classifier(combined_features)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, all_labels)

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function that includes:
    - Standard cross-entropy loss for caption generation
    - Optional contrastive loss
    - Optional ITM loss
    """

    def __init__(
        self,
        pad_token_id: int,
        use_contrastive: bool = False,
        use_itm: bool = False,
        contrastive_weight: float = 0.1,
        itm_weight: float = 0.1,
        temperature: float = 0.07,
        hidden_dim: int = 768
    ):
        """
        Initialize the combined loss.

        Args:
            pad_token_id: Padding token ID to ignore in cross-entropy
            use_contrastive: Whether to use contrastive loss
            use_itm: Whether to use ITM loss
            contrastive_weight: Weight for contrastive loss
            itm_weight: Weight for ITM loss
            temperature: Temperature for contrastive loss
            hidden_dim: Hidden dimension for ITM loss
        """
        super().__init__()
        self.pad_token_id = pad_token_id
        self.use_contrastive = use_contrastive
        self.use_itm = use_itm
        self.contrastive_weight = contrastive_weight
        self.itm_weight = itm_weight

        # Cross-entropy loss for caption generation
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id)

        # Optional auxiliary losses
        if use_contrastive:
            self.contrastive_loss = ContrastiveLoss(temperature=temperature)

        if use_itm:
            self.itm_loss = ImageTextMatchingLoss(hidden_dim=hidden_dim)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size]
            targets: Target token IDs [batch_size, seq_len]
            image_features: Optional pooled image features for contrastive/ITM loss
            text_features: Optional pooled text features for contrastive/ITM loss

        Returns:
            Dictionary with total loss and individual loss components
        """
        # Shift logits and targets for language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = targets[..., 1:].contiguous()

        # Compute cross-entropy loss
        ce_loss = self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1)
        )

        # Start with cross-entropy loss
        total_loss = ce_loss
        loss_dict = {'ce_loss': ce_loss}

        # Add contrastive loss if enabled
        if self.use_contrastive and image_features is not None and text_features is not None:
            contrastive_loss = self.contrastive_loss(image_features, text_features)
            total_loss = total_loss + self.contrastive_weight * contrastive_loss
            loss_dict['contrastive_loss'] = contrastive_loss

        # Add ITM loss if enabled
        if self.use_itm and image_features is not None and text_features is not None:
            itm_loss = self.itm_loss(image_features, text_features)
            total_loss = total_loss + self.itm_weight * itm_loss
            loss_dict['itm_loss'] = itm_loss

        loss_dict['total_loss'] = total_loss

        return loss_dict
