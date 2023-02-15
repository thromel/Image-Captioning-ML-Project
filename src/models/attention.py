import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union

from ..config import AttentionConfig, AttentionType


class AttentionMechanism(nn.Module):
    """Base class for all attention mechanisms."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention scores and context vector.

        Args:
            query: Query tensor of shape [batch_size, query_dim] or [batch_size, num_queries, query_dim]
            key: Key tensor of shape [batch_size, seq_len, key_dim]
            value: Value tensor of shape [batch_size, seq_len, value_dim]
            key_padding_mask: Boolean mask of shape [batch_size, seq_len] indicating which
                keys are padding (True) and which are valid (False)

        Returns:
            context_vector: Attention-pooled value of shape [batch_size, value_dim] or 
                            [batch_size, num_queries, value_dim]
            attention_weights: Attention weights of shape [batch_size, (num_queries), seq_len]
        """
        raise NotImplementedError


class SoftAttention(AttentionMechanism):
    """
    Classic soft attention (Bahdanau/Luong-style) used in the original Show, Attend and Tell.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.query_dim = config.hidden_dim
        self.key_dim = config.hidden_dim
        self.hidden_dim = config.hidden_dim

        # Additive attention (Bahdanau-style)
        self.query_proj = nn.Linear(self.query_dim, self.hidden_dim)
        self.key_proj = nn.Linear(self.key_dim, self.hidden_dim)
        self.energy = nn.Linear(self.hidden_dim, 1)

        # Optional temperature scaling
        self.temperature = config.temperature

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add a dimension to query if it is 2D [batch, dim]
        if query.dim() == 2:
            query = query.unsqueeze(1)  # [batch, 1, dim]
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, num_queries, _ = query.shape
        seq_len = key.shape[1]

        # Project query and key
        query_proj = self.query_proj(query)  # [batch, num_queries, hidden]
        key_proj = self.key_proj(key)  # [batch, seq_len, hidden]

        # Broadcast to compute attention scores
        # Reshape to [batch, num_queries, 1, hidden] for broadcasting
        query_proj = query_proj.unsqueeze(2)
        # Reshape to [batch, 1, seq_len, hidden] for broadcasting
        key_proj = key_proj.unsqueeze(1)

        # Compute attention scores
        # [batch, num_queries, seq_len, hidden]
        attn_sum = torch.tanh(query_proj + key_proj)
        # [batch, num_queries, seq_len, 1]
        attn_scores = self.energy(attn_sum)
        # [batch, num_queries, seq_len]
        attn_scores = attn_scores.squeeze(-1)

        # Apply temperature scaling
        attn_scores = attn_scores / self.temperature

        # Apply padding mask if provided
        if key_padding_mask is not None:
            # [batch, 1, seq_len] for broadcasting across num_queries
            padding_mask = key_padding_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(padding_mask, -1e9)

        # Compute attention weights with softmax
        # [batch, num_queries, seq_len]
        attention_weights = F.softmax(attn_scores, dim=-1)

        # Compute context vector
        # [batch, num_queries, seq_len, 1] @ [batch, 1, seq_len, value_dim]
        # -> [batch, num_queries, value_dim]
        value = value.unsqueeze(1)  # [batch, 1, seq_len, value_dim]
        context_vector = torch.matmul(
            attention_weights.unsqueeze(-2), value).squeeze(-2)

        # Return the single query result if input was 2D
        if squeeze_output:
            context_vector = context_vector.squeeze(1)
            attention_weights = attention_weights.squeeze(1)

        return context_vector, attention_weights


class MultiHeadAttention(AttentionMechanism):
    """
    Multi-head attention as used in Transformers.
    Uses the scaled dot-product attention formula.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim
        assert self.hidden_dim % self.num_heads == 0, "Hidden dim must be divisible by num heads"

        self.head_dim = self.hidden_dim // self.num_heads
        self.temperature = config.temperature

        # Linear projections
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add a dimension to query if it is 2D [batch, dim]
        if query.dim() == 2:
            query = query.unsqueeze(1)  # [batch, 1, dim]
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, num_queries, _ = query.shape
        seq_len = key.shape[1]

        # Project and reshape for multi-head attention
        def _reshape_for_multihead(x, proj):
            # Project: [batch, seq, dim] -> [batch, seq, hidden]
            x = proj(x)
            # Reshape: [batch, seq, hidden] -> [batch, seq, num_heads, head_dim]
            x = x.view(batch_size, -1, self.num_heads, self.head_dim)
            # Transpose: [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
            return x.transpose(1, 2)

        # [batch, num_heads, num_queries, head_dim]
        q = _reshape_for_multihead(query, self.query_proj)
        # [batch, num_heads, seq_len, head_dim]
        k = _reshape_for_multihead(key, self.key_proj)
        # [batch, num_heads, seq_len, head_dim]
        v = _reshape_for_multihead(value, self.value_proj)

        # Compute scaled dot-product attention
        # [batch, num_heads, num_queries, head_dim] @ [batch, num_heads, head_dim, seq_len]
        # -> [batch, num_heads, num_queries, seq_len]
        attn_scores = torch.matmul(
            q, k.transpose(-1, -2)) / (self.temperature * (self.head_dim ** 0.5))

        # Apply padding mask if provided
        if key_padding_mask is not None:
            # [batch, 1, 1, seq_len] for broadcasting across heads and queries
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            attn_scores = attn_scores.masked_fill(padding_mask, -1e9)

        # Compute attention weights with softmax
        # [batch, num_heads, num_queries, seq_len]
        attention_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention weights to values
        # [batch, num_heads, num_queries, seq_len] @ [batch, num_heads, seq_len, head_dim]
        # -> [batch, num_heads, num_queries, head_dim]
        attended_values = torch.matmul(attention_weights, v)

        # Transpose and reshape
        # [batch, num_heads, num_queries, head_dim] -> [batch, num_queries, num_heads, head_dim]
        attended_values = attended_values.transpose(1, 2).contiguous()
        # [batch, num_queries, num_heads, head_dim] -> [batch, num_queries, hidden]
        attended_values = attended_values.view(
            batch_size, num_queries, self.hidden_dim)

        # Final projection
        context_vector = self.output_proj(
            attended_values)  # [batch, num_queries, hidden]

        # Reshape attention weights for return
        # [batch, num_heads, num_queries, seq_len] -> [batch, num_queries, seq_len]
        # Average over heads for visualization
        avg_attention_weights = attention_weights.mean(dim=1)

        # Return the single query result if input was 2D
        if squeeze_output:
            context_vector = context_vector.squeeze(1)
            avg_attention_weights = avg_attention_weights.squeeze(1)

        return context_vector, avg_attention_weights


class AdaptiveAttention(AttentionMechanism):
    """
    Adaptive Attention with Visual Sentinel (Lu et al., 2017).
    Decides when to attend to the image and when to rely on the language model.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim

        # Base attention module
        self.base_attention = MultiHeadAttention(
            config) if config.num_heads > 1 else SoftAttention(config)

        # Visual sentinel components
        self.sentinel_gate = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.sentinel_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Adaptive gate - decides whether to use visual attention or sentinel
        self.adaptive_weight = nn.Linear(self.hidden_dim * 2, 1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        memory_state: Optional[torch.Tensor] = None,
        cell_state: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # We need LSTM states for the visual sentinel
        assert memory_state is not None and cell_state is not None, \
            "AdaptiveAttention requires memory_state and cell_state"

        # Add a dimension to query if it is 2D [batch, dim]
        if query.dim() == 2:
            query = query.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, num_queries, _ = query.shape

        # Compute the visual sentinel
        sentinel_gate = torch.sigmoid(self.sentinel_gate(
            torch.cat([query, memory_state.unsqueeze(
                1).expand(-1, num_queries, -1)], dim=-1)
        ))
        visual_sentinel = sentinel_gate * \
            torch.tanh(cell_state.unsqueeze(1).expand(-1, num_queries, -1))
        visual_sentinel = self.sentinel_proj(visual_sentinel)

        # Compute base attention using the provided attention mechanism
        context_vector, attention_weights = self.base_attention(
            query, key, value, key_padding_mask, **kwargs
        )

        # Compute adaptive weights - how much to use visual context vs sentinel
        sentinel_context_cat = torch.cat(
            [context_vector, visual_sentinel], dim=-1)
        adaptive_weight = torch.sigmoid(
            self.adaptive_weight(sentinel_context_cat))

        # Combine visual context and sentinel based on adaptive weight
        final_context = adaptive_weight * context_vector + \
            (1 - adaptive_weight) * visual_sentinel

        # Return the single query result if input was 2D
        if squeeze_output:
            final_context = final_context.squeeze(1)
            attention_weights = attention_weights.squeeze(1)

        return final_context, attention_weights


class AttentionOnAttention(AttentionMechanism):
    """
    Attention on Attention (AoA) module (Huang et al., 2019).
    Adds a second attention layer to filter the first attention output.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim

        # Base attention module
        self.base_attention = MultiHeadAttention(
            config) if config.num_heads > 1 else SoftAttention(config)

        # AoA modules
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.info_vector_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh()
        )
        self.info_gate_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid()
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add a dimension to query if it is 2D [batch, dim]
        if query.dim() == 2:
            query = query.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        # Compute base attention
        context_vector, attention_weights = self.base_attention(
            query, key, value, key_padding_mask, **kwargs
        )

        # Transform query for AoA
        query_transformed = self.query_proj(query)

        # Concatenate context and transformed query
        concat = torch.cat([context_vector, query_transformed], dim=-1)

        # Compute information vector and gate
        info_vector = self.info_vector_proj(concat)
        info_gate = self.info_gate_proj(concat)

        # Apply gate to filter information
        filtered_context = info_vector * info_gate

        # Return the single query result if input was 2D
        if squeeze_output:
            filtered_context = filtered_context.squeeze(1)
            attention_weights = attention_weights.squeeze(1)

        return filtered_context, attention_weights


def build_attention(config: AttentionConfig) -> AttentionMechanism:
    """Factory function to build the attention mechanism based on configuration."""
    if config.attention_type == AttentionType.SOFT:
        return SoftAttention(config)
    elif config.attention_type == AttentionType.MULTI_HEAD:
        return MultiHeadAttention(config)
    elif config.attention_type == AttentionType.ADAPTIVE:
        return AdaptiveAttention(config)
    elif config.attention_type == AttentionType.AOA:
        return AttentionOnAttention(config)
    else:
        raise ValueError(
            f"Unsupported attention type: {config.attention_type}")
