import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Union, Any
from abc import ABC, abstractmethod

from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    PreTrainedModel,
    PreTrainedTokenizer
)

from ..config import DecoderType, DecoderConfig, AttentionConfig
from .attention import build_attention, AttentionMechanism


class CaptionDecoder(nn.Module, ABC):
    """Base class for all caption decoders."""

    @abstractmethod
    def forward(
        self,
        encoder_features: Dict[str, torch.Tensor],
        captions: Optional[torch.Tensor] = None,
        caption_lengths: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the decoder.

        Args:
            encoder_features: Dictionary of encoder features including:
                - features: Image features [batch_size, feature_length, feature_dim]
                - pooled_features: Global image representation [batch_size, feature_dim]
                - attention_mask: Attention mask for features [batch_size, feature_length]
            captions: Tensor of caption tokens of shape [batch_size, max_length]
                     (for teacher forcing during training)
            caption_lengths: Tensor of caption lengths of shape [batch_size]

        Returns:
            Dictionary containing:
                - logits: Output logits of shape [batch_size, max_length, vocab_size]
                - attention_weights: Attention weights (optional)
                - hidden_states: Hidden states (optional)
        """
        pass

    @abstractmethod
    def generate(
        self,
        encoder_features: Dict[str, torch.Tensor],
        max_length: int,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate captions for the given image features.

        Args:
            encoder_features: Dictionary of encoder features
            max_length: Maximum caption length to generate

        Returns:
            Generated caption tokens of shape [batch_size, max_length]
            Dictionary of additional information like attention weights
        """
        pass


class LSTMDecoder(CaptionDecoder):
    """
    LSTM-based decoder with attention, similar to Show, Attend and Tell.
    """

    def __init__(
        self,
        config: DecoderConfig,
        attention_config: AttentionConfig,
        vocab_size: int,
        pad_token_id: int,
        embedding_dim: int = None
    ):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.embedding_dim = embedding_dim or config.hidden_dim
        self.num_layers = config.num_layers
        self.vocab_size = vocab_size
        self.dropout_p = config.dropout
        self.pad_token_id = pad_token_id

        # Word embedding layer
        self.embedding = nn.Embedding(
            vocab_size, self.embedding_dim, padding_idx=pad_token_id
        )

        # Initialize LSTM decoder
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim + self.hidden_dim,  # Input + context vector
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_p if self.num_layers > 1 else 0
        )

        # Initialize attention mechanism
        self.attention = build_attention(attention_config)

        # Output projection layer
        self.output_layer = nn.Linear(self.hidden_dim, vocab_size)

        # Initialization
        self.init_h = nn.Linear(
            self.hidden_dim, self.hidden_dim * self.num_layers)
        self.init_c = nn.Linear(
            self.hidden_dim, self.hidden_dim * self.num_layers)

        # Dropout
        self.dropout = nn.Dropout(self.dropout_p)

    def _init_hidden_states(self, encoder_features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden states using encoder features."""
        pooled_features = encoder_features["pooled_features"]
        batch_size = pooled_features.size(0)

        h0 = self.init_h(pooled_features)
        h0 = h0.view(batch_size, self.num_layers,
                     self.hidden_dim).transpose(0, 1).contiguous()

        c0 = self.init_c(pooled_features)
        c0 = c0.view(batch_size, self.num_layers,
                     self.hidden_dim).transpose(0, 1).contiguous()

        return h0, c0

    def forward(
        self,
        encoder_features: Dict[str, torch.Tensor],
        captions: Optional[torch.Tensor] = None,
        caption_lengths: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with teacher forcing."""
        if captions is None:
            # If no captions provided, we're likely in inference mode
            batch_size = encoder_features["pooled_features"].size(0)
            return self.generate(encoder_features, config.max_length)

        # Prepare encoder features
        image_features = encoder_features["features"]
        batch_size = image_features.size(0)
        feature_length = image_features.size(1)
        attention_mask = encoder_features.get("attention_mask", None)

        # Sort captions by length for packed sequence
        if caption_lengths is not None:
            caption_lengths, sort_indices = caption_lengths.sort(
                descending=True)
            captions = captions[sort_indices]
            image_features = image_features[sort_indices]
            if attention_mask is not None:
                attention_mask = attention_mask[sort_indices]

            # Create original order indices for later unsorting
            _, unsort_indices = sort_indices.sort()

        # Initialize LSTM hidden states
        h, c = self._init_hidden_states(encoder_features)

        # Encode captions
        # [batch_size, max_len, embed_dim]
        embeddings = self.embedding(captions)
        embeddings = self.dropout(embeddings)

        # Prepare output containers
        max_length = captions.size(1)
        outputs = torch.zeros(batch_size, max_length,
                              self.vocab_size, device=captions.device)
        attention_weights_list = []

        # Initialize with zeros
        prev_context = torch.zeros(
            batch_size, self.hidden_dim, device=captions.device)

        # Generate captions word by word
        for t in range(max_length):
            # Create input for current time step
            # Combine embedding of previous token with previous context vector
            lstm_input = torch.cat(
                [embeddings[:, t, :], prev_context], dim=1).unsqueeze(1)

            # Forward pass through LSTM
            # h shape: [num_layers, batch, hidden_dim]
            # output shape: [batch, 1, hidden_dim]
            output, (h, c) = self.lstm(lstm_input, (h, c))

            # Get the top-layer hidden state as query for attention
            query = output.squeeze(1)  # [batch, hidden_dim]

            # Compute attention over image features
            context, attn_weights = self.attention(
                query=query,
                key=image_features,
                value=image_features,
                key_padding_mask=~attention_mask if attention_mask is not None else None,
                memory_state=h[-1],  # Use last layer's hidden state
                cell_state=c[-1]     # Use last layer's cell state
            )

            # Store context for next iteration
            prev_context = context

            # Store attention weights for visualization
            attention_weights_list.append(attn_weights)

            # Compute output logits
            output = self.output_layer(self.dropout(context))
            outputs[:, t] = output

        # Unsort everything if necessary
        if caption_lengths is not None:
            outputs = outputs[unsort_indices]
            attention_weights = torch.stack(
                attention_weights_list, dim=1)  # [batch, time, feature_len]
            attention_weights = attention_weights[unsort_indices]
        else:
            attention_weights = torch.stack(
                attention_weights_list, dim=1)  # [batch, time, feature_len]

        return {
            "logits": outputs,
            "attention_weights": attention_weights
        }

    def generate(
        self,
        encoder_features: Dict[str, torch.Tensor],
        max_length: int,
        start_token_id: int = 1,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate caption using greedy decoding."""
        # Prepare encoder features
        image_features = encoder_features["features"]
        batch_size = image_features.size(0)
        feature_length = image_features.size(1)
        attention_mask = encoder_features.get("attention_mask", None)

        # Initialize LSTM hidden states
        h, c = self._init_hidden_states(encoder_features)

        # Initialize input with start token
        current_input = torch.full(
            (batch_size,), start_token_id, dtype=torch.long, device=image_features.device
        )

        # Prepare output containers
        output_ids = torch.zeros(
            batch_size, max_length, dtype=torch.long, device=image_features.device
        )
        attention_weights_list = []

        # Initialize with zeros
        prev_context = torch.zeros(
            batch_size, self.hidden_dim, device=image_features.device)

        # Generate captions word by word
        for t in range(max_length):
            # Store current token
            output_ids[:, t] = current_input

            # Embed current token
            current_embed = self.embedding(current_input)  # [batch, embed_dim]

            # Create LSTM input
            lstm_input = torch.cat(
                [current_embed, prev_context], dim=1).unsqueeze(1)

            # Forward pass through LSTM
            output, (h, c) = self.lstm(lstm_input, (h, c))

            # Get the top-layer hidden state as query for attention
            query = output.squeeze(1)  # [batch, hidden_dim]

            # Compute attention over image features
            context, attn_weights = self.attention(
                query=query,
                key=image_features,
                value=image_features,
                key_padding_mask=~attention_mask if attention_mask is not None else None,
                memory_state=h[-1],  # Use last layer's hidden state
                cell_state=c[-1]     # Use last layer's cell state
            )

            # Store context for next iteration
            prev_context = context

            # Store attention weights for visualization
            attention_weights_list.append(attn_weights)

            # Compute output logits
            logits = self.output_layer(context)

            # Get next token (greedy decoding)
            current_input = logits.argmax(dim=1)

        # Stack attention weights
        attention_weights = torch.stack(
            attention_weights_list, dim=1)  # [batch, time, feature_len]

        return output_ids, {
            "attention_weights": attention_weights
        }


class TransformerDecoder(CaptionDecoder):
    """
    Transformer decoder that processes visual features using self-attention.
    """

    def __init__(
        self,
        config: DecoderConfig,
        vocab_size: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int
    ):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.vocab_size = vocab_size
        self.dropout_p = config.dropout
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Word embedding and position encoding
        self.embedding = nn.Embedding(
            vocab_size, self.hidden_dim, padding_idx=pad_token_id
        )
        self.position_encoding = nn.Embedding(
            config.max_length, self.hidden_dim
        )

        # Initialize transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout_p,
            activation="gelu",
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.num_layers
        )

        # Output projection layer
        self.output_layer = nn.Linear(self.hidden_dim, vocab_size)

        # Visual projection
        self.visual_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(
        self,
        encoder_features: Dict[str, torch.Tensor],
        captions: Optional[torch.Tensor] = None,
        caption_lengths: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with transformer decoder."""
        if captions is None:
            # If no captions provided, we're in inference mode
            return self.generate(encoder_features, 50)  # Default max length

        # Prepare encoder features
        # [batch, feature_length, hidden_dim]
        image_features = encoder_features["features"]
        batch_size = image_features.size(0)
        image_attention_mask = encoder_features.get("attention_mask", None)

        # Project visual features if necessary
        image_features = self.visual_projection(image_features)

        # Convert attention mask to appropriate format for transformer
        # For transformer, 0 means masked (invalid) and 1 means unmasked (valid)
        if image_attention_mask is not None:
            mask = (~image_attention_mask).float() * -1e9
            image_attention_mask = mask.unsqueeze(
                1).repeat(1, captions.size(1), 1)

        # Prepare target mask (to ensure causal attention)
        tgt_len = captions.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len).to(captions.device)

        # Generate target padding mask
        tgt_padding_mask = (captions == self.pad_token_id)

        # Embed captions
        # [batch, max_len, hidden_dim]
        caption_embeds = self.embedding(captions)

        # Add positional encodings
        pos_indices = torch.arange(tgt_len, device=captions.device).unsqueeze(
            0).expand(batch_size, -1)
        caption_embeds = caption_embeds + self.position_encoding(pos_indices)

        # Apply dropout
        caption_embeds = self.dropout(caption_embeds)

        # Forward pass through transformer decoder
        # outputs: [batch, max_len, hidden_dim]
        outputs = self.transformer_decoder(
            tgt=caption_embeds,
            memory=image_features,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=None if image_attention_mask is None else image_attention_mask.squeeze(
                1)
        )

        # Project to vocabulary
        logits = self.output_layer(outputs)  # [batch, max_len, vocab_size]

        return {
            "logits": logits,
            # Note: transformer attention weights are inside the model and not directly accessible
            "hidden_states": outputs
        }

    def generate(
        self,
        encoder_features: Dict[str, torch.Tensor],
        max_length: int,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate captions using the transformer decoder."""
        # Prepare encoder features
        # [batch, feature_length, hidden_dim]
        image_features = encoder_features["features"]
        batch_size = image_features.size(0)
        device = image_features.device

        # Project visual features
        image_features = self.visual_projection(image_features)

        # Initialize with start token
        input_ids = torch.full(
            (batch_size, 1), self.bos_token_id, dtype=torch.long, device=device
        )

        # Iteratively generate tokens
        for i in range(max_length - 1):
            # Create position indices
            curr_len = input_ids.size(1)
            pos_indices = torch.arange(curr_len, device=device).unsqueeze(
                0).expand(batch_size, -1)

            # Embed inputs
            inputs_embeds = self.embedding(input_ids)
            inputs_embeds = inputs_embeds + self.position_encoding(pos_indices)

            # Create causal mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                curr_len).to(device)

            # Decode next token
            outputs = self.transformer_decoder(
                tgt=inputs_embeds,
                memory=image_features,
                tgt_mask=tgt_mask
            )

            # Get logits and next token
            next_token_logits = self.output_layer(outputs[:, -1, :])
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if we've generated EOS
            if (next_token == self.eos_token_id).all():
                break

        return input_ids, {}


class GPT2Decoder(CaptionDecoder):
    """
    GPT-2 based decoder for image captioning.
    Uses a pretrained GPT-2 model conditioned on image features.
    """

    def __init__(
        self,
        config: DecoderConfig,
        vocab_size: int = None,
        pad_token_id: int = None,
        bos_token_id: int = None,
        eos_token_id: int = None
    ):
        super().__init__()
        # Use pretrained model or create new one
        if config.pretrained_model_name:
            self.model = GPT2LMHeadModel.from_pretrained(
                config.pretrained_model_name)
            if vocab_size and vocab_size != self.model.config.vocab_size:
                # Resize token embeddings if vocab size doesn't match
                self.model.resize_token_embeddings(vocab_size)
        else:
            # Create custom GPT-2 config
            model_config = GPT2Config(
                vocab_size=vocab_size,
                n_positions=config.max_length,
                n_ctx=config.max_length,
                n_embd=config.hidden_dim,
                n_layer=config.num_layers,
                n_head=config.num_heads,
                resid_pdrop=config.dropout,
                embd_pdrop=config.dropout,
                attn_pdrop=config.dropout,
            )
            self.model = GPT2LMHeadModel(model_config)

        # Set special token IDs
        self.pad_token_id = pad_token_id or 0
        self.bos_token_id = bos_token_id or 1
        self.eos_token_id = eos_token_id or 2

        # Visual features projection
        self.visual_projection = nn.Linear(
            config.hidden_dim, self.model.config.n_embd
        )

        # Control prefix parameter for better conditioning
        self.prefix_length = 10  # How many tokens of context to use
        self.image_prefix = nn.Parameter(
            torch.randn(1, self.prefix_length, self.model.config.n_embd)
        )

        # Map image features to prefix tokens
        self.image_to_prefix = nn.Linear(
            config.hidden_dim, self.prefix_length * self.model.config.n_embd
        )

    def forward(
        self,
        encoder_features: Dict[str, torch.Tensor],
        captions: Optional[torch.Tensor] = None,
        caption_lengths: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with GPT-2 decoder."""
        if captions is None:
            # If no captions provided, we're in inference mode
            return self.generate(encoder_features, 50)  # Default max length

        # Get visual features
        # [batch, hidden_dim]
        pooled_features = encoder_features["pooled_features"]

        # Map image features to GPT-2 prefix context
        image_prefix = self.image_to_prefix(pooled_features)
        image_prefix = image_prefix.view(
            pooled_features.size(0), self.prefix_length, -1
        )

        # Prepare inputs for GPT-2
        # Shift the captions for teacher forcing: captions become labels
        labels = captions.clone()

        # Prepare attention mask, ignoring padding tokens
        attention_mask = (captions != self.pad_token_id).float()

        # Run the model
        outputs = self.model(
            input_ids=captions,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=self._create_prefix_past_key_values(image_prefix),
            return_dict=True
        )

        return {
            "logits": outputs.logits,
            "loss": outputs.loss
        }

    def _create_prefix_past_key_values(self, prefix_embeds: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Create past key values for GPT-2 from prefix embeddings."""
        # This is a simplified placeholder implementation
        # In a real implementation, you would project the prefix embeddings into the correct shapes
        # for each attention layer's past key and value states
        batch_size = prefix_embeds.size(0)

        # Initialize past key values list
        past_key_values = []

        # For each layer in the model
        for i in range(self.model.config.n_layer):
            # Project prefix for this layer (this is a simplified approach)
            # In a real implementation, you might use layer-specific projections
            past_key = prefix_embeds.clone()  # [batch, prefix_len, hidden]
            past_value = prefix_embeds.clone()  # [batch, prefix_len, hidden]

            # Add to the list
            past_key_values.append((past_key, past_value))

        return past_key_values

    def generate(
        self,
        encoder_features: Dict[str, torch.Tensor],
        max_length: int,
        num_beams: int = 4,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate captions using the GPT-2 decoder with beam search."""
        # Get visual features
        # [batch, hidden_dim]
        pooled_features = encoder_features["pooled_features"]
        batch_size = pooled_features.size(0)
        device = pooled_features.device

        # Map image features to GPT-2 prefix context
        image_prefix = self.image_to_prefix(pooled_features)
        image_prefix = image_prefix.view(
            batch_size, self.prefix_length, -1
        )

        # Create a dummy input with just the start token
        input_ids = torch.full(
            (batch_size, 1), self.bos_token_id, dtype=torch.long, device=device
        )

        # Use HuggingFace's generate method for beam search
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_beams=num_beams,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            past_key_values=self._create_prefix_past_key_values(image_prefix),
            **kwargs
        )

        return outputs, {}


def build_decoder(
    config: DecoderConfig,
    attention_config: AttentionConfig,
    vocab_size: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int
) -> CaptionDecoder:
    """Factory function to build the decoder based on configuration."""
    if config.decoder_type == DecoderType.LSTM:
        return LSTMDecoder(
            config=config,
            attention_config=attention_config,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id
        )
    elif config.decoder_type == DecoderType.TRANSFORMER:
        return TransformerDecoder(
            config=config,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id
        )
    elif config.decoder_type == DecoderType.GPT2:
        return GPT2Decoder(
            config=config,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id
        )
    else:
        raise ValueError(f"Unsupported decoder type: {config.decoder_type}")
