import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Union, Any

from transformers import PreTrainedTokenizer

from ..config import Config, ModelConfig
from .encoders import build_encoder
from .decoders import build_decoder
from .attention import build_attention


class ImageCaptioningModel(nn.Module):
    """
    Full image captioning model integrating encoder, decoder, and optional components.
    """

    def __init__(self, config: Config, tokenizer: Optional[PreTrainedTokenizer] = None):
        super().__init__()
        self.config = config
        self.model_config = config.model

        # Build encoder
        self.encoder = build_encoder(config.model.encoder)

        # Prepare tokenizer info
        if tokenizer:
            vocab_size = len(tokenizer)
            pad_token_id = tokenizer.pad_token_id
            bos_token_id = tokenizer.bos_token_id
            eos_token_id = tokenizer.eos_token_id
        else:
            vocab_size = config.model.vocab_size
            pad_token_id = config.model.pad_token_id
            bos_token_id = config.model.bos_token_id
            eos_token_id = config.model.eos_token_id

        # Build decoder
        self.decoder = build_decoder(
            config=config.model.decoder,
            attention_config=config.model.attention,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id
        )

        # BLIP-2 style Q-Former for enhanced vision-language connection
        if config.model.use_q_former:
            self.q_former = QFormer(
                query_dim=config.model.projection_dim,
                vision_dim=config.model.encoder.feature_dim,
                num_queries=config.model.q_former_num_queries
            )

    def forward(
        self,
        images: torch.Tensor,
        captions: Optional[torch.Tensor] = None,
        caption_lengths: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the full model.

        Args:
            images: Input images tensor of shape [batch_size, channels, height, width]
            captions: Input caption tokens of shape [batch_size, caption_length]
            caption_lengths: Lengths of captions of shape [batch_size]
            return_dict: Whether to return results as a dictionary

        Returns:
            Dictionary containing model outputs including logits and loss
        """
        # Get visual features from encoder
        encoder_features = self.encoder(images)

        # Process with Q-Former if configured
        if hasattr(self, "q_former"):
            q_former_output = self.q_former(
                encoder_features["features"],
                encoder_features["attention_mask"]
            )
            encoder_features["features"] = q_former_output["queries"]
            encoder_features["attention_mask"] = torch.ones(
                q_former_output["queries"].size(0),
                q_former_output["queries"].size(1),
                device=q_former_output["queries"].device
            )

        # Forward pass through decoder
        decoder_output = self.decoder(
            encoder_features=encoder_features,
            captions=captions,
            caption_lengths=caption_lengths,
            **kwargs
        )

        # Return results
        if return_dict:
            return decoder_output
        else:
            return decoder_output["logits"]

    def generate(
        self,
        images: torch.Tensor,
        max_length: int = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate captions for the given images.

        Args:
            images: Input images tensor of shape [batch_size, channels, height, width]
            max_length: Maximum caption length to generate

        Returns:
            Generated caption tokens of shape [batch_size, max_length]
            Dictionary of additional information
        """
        # Use default max length if not provided
        if max_length is None:
            max_length = self.config.inference.max_length

        # Get visual features from encoder
        encoder_features = self.encoder(images)

        # Process with Q-Former if configured
        if hasattr(self, "q_former"):
            q_former_output = self.q_former(
                encoder_features["features"],
                encoder_features["attention_mask"]
            )
            encoder_features["features"] = q_former_output["queries"]
            encoder_features["attention_mask"] = torch.ones(
                q_former_output["queries"].size(0),
                q_former_output["queries"].size(1),
                device=q_former_output["queries"].device
            )

        # Generate captions using the decoder
        captions, additional_info = self.decoder.generate(
            encoder_features=encoder_features,
            max_length=max_length,
            **kwargs
        )

        return captions, additional_info


class QFormer(nn.Module):
    """
    Query-based Transformer (Q-Former) similar to BLIP-2 for mapping
    image features to a set of learnable query vectors.
    """

    def __init__(
        self,
        query_dim: int = 768,
        vision_dim: int = 768,
        num_queries: int = 32,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        # Learnable query vectors (similar to BLIP-2/CoCa)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_queries, query_dim))
        nn.init.normal_(self.query_tokens, std=0.02)

        # Vision projection
        self.vision_proj = nn.Linear(
            vision_dim, query_dim) if vision_dim != query_dim else nn.Identity()

        # Cross-attention layers to attend queries to image features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=query_dim,
            nhead=num_heads,
            dim_feedforward=query_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        # Cross-attention to vision features
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=query_dim,
            nhead=num_heads,
            dim_feedforward=query_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers)

    def forward(
        self,
        vision_features: torch.Tensor,
        vision_attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process visual features with Q-Former.

        Args:
            vision_features: Visual features from encoder [batch_size, seq_len, vision_dim]
            vision_attention_mask: Attention mask for vision features [batch_size, seq_len]

        Returns:
            Dictionary containing:
                - queries: Processed query features [batch_size, num_queries, query_dim]
        """
        batch_size = vision_features.size(0)

        # Project vision features
        vision_features = self.vision_proj(vision_features)

        # Expand query tokens to batch size
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        # Convert attention mask for transformer if provided
        if vision_attention_mask is not None:
            vision_attention_mask = (1.0 - vision_attention_mask) * -10000.0
            vision_attention_mask = vision_attention_mask.unsqueeze(1)

        # Self-attention on query tokens
        query_features = self.encoder(query_tokens)

        # Cross-attention to visual features
        output = self.decoder(
            tgt=query_features,
            memory=vision_features,
            memory_key_padding_mask=None if vision_attention_mask is None else vision_attention_mask.squeeze(
                1)
        )

        return {"queries": output}
