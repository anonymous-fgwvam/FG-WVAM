# Copyright 2024 The Genmo team and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import PixArtAlphaTextProjection
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm
import torch.utils.checkpoint

from models.ltx_models.ltx_attention_processor import Attention


from models.action_patches.patches import preprocessing_action_states, add_action_expert
from models.value_patches.value_patches import preprocessing_value_states, add_value_expert
from models.grounding_patches.grounding_patches import preprocessing_grounding_states, add_grounding_expert

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



class LTXVideoAttentionProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the LTX model. It applies a normalization layer and rotary embedding on the query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "LTXVideoAttentionProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        n_view: int = 1,
        cross_view_attn: bool = False,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        

        if cross_view_attn:
            assert n_view is not None
            # hard code here, we also use attention mask in self-attention
            if attention_mask is not None:
                batch_size = batch_size // n_view   # n_view=3 for video self-attn, and attn batch is in-batch // 3
                attention_mask = attention_mask.repeat(batch_size, n_view, n_view)
                sequence_length = attention_mask.shape[-1]

        elif image_rotary_emb is not None and attention_mask is not None:  # for self-attn w/o cross-view-attn
            attention_mask = attention_mask.repeat(batch_size, 1, 1)
            sequence_length = attention_mask.shape[-1]
        

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:   # for self attn, extend the sequence length according to the cross_view_attn param
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
            if cross_view_attn:
                query = rearrange(query, '(b v) l c -> b (v l) c', v=n_view)
                key = rearrange(key, '(b v) l c -> b (v l) c', v=n_view)
                value = rearrange(value, '(b v) l c -> b (v l) c', v=n_view)

        else:   # for cross attn, extend the sequence length
            query = rearrange(query, '(b v) l c -> b (v l) c', v=n_view)
        

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        
        

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if cross_view_attn or image_rotary_emb is None:
            hidden_states = rearrange(hidden_states, 'b (v l) c -> (b v) l c', v=n_view)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class LTXVideoRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        base_num_frames: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        patch_size: int = 1,
        patch_size_t: int = 1,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.base_num_frames = base_num_frames
        self.base_height = base_height
        self.base_width = base_width
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.theta = theta

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_interpolation_scale: Optional[Tuple[torch.Tensor, float, float]] = None,
        num_frames=None, height=None, width=None,
        frame_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = hidden_states.shape[0]

        # Always compute rope in fp32
        grid_h = torch.arange(height, dtype=torch.float32, device=hidden_states.device)
        grid_w = torch.arange(width, dtype=torch.float32, device=hidden_states.device)
        if frame_ids is not None:
            # Use real episode frame IDs instead of sequential token positions,
            # so rope encodes true elapsed time between sparsely/non-uniformly sampled frames.
            # frame_ids can be [num_frames] or [batch_size, num_frames].
            grid_f = frame_ids.to(dtype=torch.float32, device=hidden_states.device)
            if grid_f.dim() == 2:
                # [batch_size, num_frames] -> use first batch's frame_ids for meshgrid (broadcast later)
                grid_f = grid_f[0]
            # grid_f should now be 1D [num_frames]
        else:
            grid_f = torch.arange(num_frames, dtype=torch.float32, device=hidden_states.device)
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

        """
        rope_interpolation_scale = (
            1 / latent_frame_rate,
            self.vae_spatial_compression_ratio,
            self.vae_spatial_compression_ratio,
        )
        """

        if rope_interpolation_scale is not None:
            grid[:, 0:1] = grid[:, 0:1] * rope_interpolation_scale[0] * self.patch_size_t / self.base_num_frames
            grid[:, 1:2] = grid[:, 1:2] * rope_interpolation_scale[1] * self.patch_size / self.base_height
            grid[:, 2:3] = grid[:, 2:3] * rope_interpolation_scale[2] * self.patch_size / self.base_width

        grid = grid.flatten(2, 4).transpose(1, 2)

        start = 1.0
        end = self.theta
        freqs = self.theta ** torch.linspace(
            math.log(start, self.theta),
            math.log(end, self.theta),
            self.dim // 6,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        freqs = freqs * math.pi / 2.0
        freqs = freqs * (grid.unsqueeze(-1) * 2 - 1)
        freqs = freqs.transpose(-1, -2).flatten(2)

        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        if self.dim % 6 != 0:
            cos_padding = torch.ones_like(cos_freqs[:, :, : self.dim % 6])
            sin_padding = torch.zeros_like(cos_freqs[:, :, : self.dim % 6])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        return cos_freqs, sin_freqs

@maybe_allow_in_graph
class LTXVideoTransformerBlock(nn.Module):
    r"""
    Transformer block used in [LTX](https://huggingface.co/Lightricks/LTX-Video).

    Args:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        qk_norm (`str`, defaults to `"rms_norm"`):
            The normalization layer to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            processor=LTXVideoAttentionProcessor2_0(),
        )

        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            processor=LTXVideoAttentionProcessor2_0(),
        )

        self.ff = FeedForward(dim, activation_fn=activation_fn)

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        n_view: int = None,
        cross_view_attn: bool = False,
        self_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        norm_hidden_states = self.norm1(hidden_states)

        num_ada_params = self.scale_shift_table.shape[0]
        ada_values = self.scale_shift_table[None, None] + temb.reshape(batch_size, temb.size(1), num_ada_params, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa


        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            image_rotary_emb=image_rotary_emb,
            n_view=n_view,
            cross_view_attn=cross_view_attn,
            attention_mask=self_attention_mask,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        attn_hidden_states = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=None,
            attention_mask=encoder_attention_mask,
            n_view=n_view,
        )
        hidden_states = hidden_states + attn_hidden_states
        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states

class GoalHeatmapHead(nn.Module):
    """Decode per-frame goal-attention logits from one future DiT latent grid."""

    def __init__(self, in_dim: int, out_hw: int = 32, num_frames: int = 2):
        super().__init__()
        self.out_hw = out_hw
        self.num_frames = num_frames
        self.proj = nn.Linear(in_dim, 128)
        self.conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, num_frames, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch*views, spatial_tokens, hidden_dim]
        batch_views, spatial_tokens, _ = x.shape
        spatial_size = int(round(spatial_tokens**0.5))
        if spatial_size * spatial_size != spatial_tokens:
            raise ValueError(
                f"GoalHeatmapHead requires a square latent grid, got {spatial_tokens} tokens"
            )
        hidden = self.proj(x)
        hidden = hidden.reshape(batch_views, spatial_size, spatial_size, 128)
        hidden = F.gelu(hidden.permute(0, 3, 1, 2))
        hidden = F.gelu(self.conv1(hidden))
        logits = self.conv2(hidden)
        return F.interpolate(
            logits,
            size=(self.out_hw, self.out_hw),
            mode="bilinear",
            align_corners=False,
        )


class GoalQueryAdapter(nn.Module):
    """Read fused multi-view goal tokens and optionally write them back to video tokens."""

    def __init__(
        self,
        video_dim: int,
        goal_dim: int = 512,
        num_heads: int = 8,
        num_queries: int = 2,
        out_hw: int = 32,
        spatial_gate_max: float = 4.0,
    ):
        super().__init__()
        if goal_dim % num_heads != 0:
            raise ValueError(
                f"goal_dim={goal_dim} must be divisible by num_heads={num_heads}"
            )
        self.goal_dim = goal_dim
        self.num_heads = num_heads
        self.head_dim = goal_dim // num_heads
        self.num_queries = num_queries
        self.out_hw = out_hw
        self.spatial_gate_max = float(spatial_gate_max)

        self.goal_queries = nn.Parameter(torch.randn(num_queries, goal_dim) / goal_dim**0.5)
        self.video_read_norm = nn.LayerNorm(video_dim, eps=1e-6)
        self.text_norm = nn.LayerNorm(video_dim, eps=1e-6)
        self.text_to_query = nn.Linear(video_dim, goal_dim)
        self.read_q = nn.Linear(goal_dim, goal_dim)
        self.read_k = nn.Linear(video_dim, goal_dim)
        self.read_v = nn.Linear(video_dim, goal_dim)
        self.goal_norm = nn.LayerNorm(goal_dim, eps=1e-6)
        self.goal_ff = nn.Sequential(
            nn.Linear(goal_dim, 4 * goal_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * goal_dim, goal_dim),
        )
        self.heatmap_q = nn.Linear(goal_dim, goal_dim)

        self.video_write_norm = nn.LayerNorm(video_dim, eps=1e-6)
        self.write_q = nn.Linear(video_dim, goal_dim)
        self.write_k = nn.Linear(goal_dim, goal_dim)
        self.write_v = nn.Linear(goal_dim, goal_dim)
        self.write_out = nn.Linear(goal_dim, video_dim)
        self.write_gate = nn.Parameter(torch.zeros(()))

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _text_pool(
        self,
        text_tokens: torch.Tensor,
        text_attention_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if text_attention_bias is None:
            return text_tokens.mean(dim=1)
        valid = (text_attention_bias.squeeze(1) == 0).to(text_tokens.dtype)
        return (text_tokens * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)

    def read(
        self,
        hidden_states: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_bias: Optional[torch.Tensor],
        n_view: int,
        num_frames: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_views, sequence_length, channels = hidden_states.shape
        if batch_views % n_view != 0:
            raise ValueError(
                f"Goal query batch {batch_views} is not divisible by n_view={n_view}"
            )
        if sequence_length % num_frames != 0:
            raise ValueError(
                f"Goal query cannot split {sequence_length} tokens into {num_frames} frames"
            )
        batch = batch_views // n_view
        spatial_tokens = sequence_length // num_frames
        spatial_size = int(round(spatial_tokens**0.5))
        if spatial_size * spatial_size != spatial_tokens:
            raise ValueError(
                f"Goal query requires a square latent grid, got {spatial_tokens} tokens"
            )

        future_tokens = hidden_states.reshape(
            batch, n_view, num_frames, spatial_tokens, channels
        )[:, :, -1]
        future_tokens = future_tokens.reshape(batch, n_view * spatial_tokens, channels)
        future_tokens = self.video_read_norm(future_tokens)

        text_summary = self._text_pool(text_tokens, text_attention_bias)
        queries = self.goal_queries.unsqueeze(0) + self.text_to_query(
            self.text_norm(text_summary)
        ).unsqueeze(1)
        query = self._split_heads(self.read_q(queries))
        key = self._split_heads(self.read_k(future_tokens))
        value = self._split_heads(self.read_v(future_tokens))
        read_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )
        read_attention = read_scores.softmax(dim=-1)
        goal_tokens = torch.matmul(read_attention, value).transpose(1, 2).reshape(
            batch, self.num_queries, self.goal_dim
        )
        goal_tokens = self.goal_norm(goal_tokens + queries)
        goal_tokens = goal_tokens + self.goal_ff(goal_tokens)

        # Decode localization from the aggregated token, so heatmap supervision
        # trains Q, K, V and the goal-token FFN instead of only the first QK map.
        heatmap_query = self._split_heads(self.heatmap_q(goal_tokens))
        heatmap_scores = torch.matmul(
            heatmap_query, key.transpose(-1, -2)
        ) / math.sqrt(self.head_dim)
        heatmap_attention = heatmap_scores.softmax(dim=-1)
        heatmap_logits = heatmap_scores.mean(dim=1).reshape(
            batch, self.num_queries, n_view, spatial_size, spatial_size
        ).permute(0, 2, 1, 3, 4)
        heatmap_logits = F.interpolate(
            heatmap_logits.reshape(
                batch * n_view * self.num_queries, 1, spatial_size, spatial_size
            ),
            size=(self.out_hw, self.out_hw),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, n_view, self.num_queries, self.out_hw, self.out_hw)
        return goal_tokens, heatmap_logits, heatmap_attention.mean(dim=1)

    def write(
        self,
        hidden_states: torch.Tensor,
        goal_tokens: torch.Tensor,
        read_attention: torch.Tensor,
        n_view: int,
        num_frames: int,
    ) -> torch.Tensor:
        batch_views, sequence_length, channels = hidden_states.shape
        batch = batch_views // n_view
        spatial_tokens = sequence_length // num_frames
        goal_per_view = goal_tokens.repeat_interleave(n_view, dim=0)

        query = self._split_heads(self.write_q(self.video_write_norm(hidden_states)))
        key = self._split_heads(self.write_k(goal_per_view))
        value = self._split_heads(self.write_v(goal_per_view))
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        delta = torch.matmul(scores.softmax(dim=-1), value)
        delta = delta.transpose(1, 2).reshape(batch_views, sequence_length, self.goal_dim)
        delta = self.write_out(delta)

        relevance = read_attention.mean(dim=1) * (n_view * spatial_tokens)
        relevance = relevance.clamp(0.0, self.spatial_gate_max).reshape(
            batch, n_view, spatial_tokens
        )
        token_gate = hidden_states.new_zeros(batch, n_view, num_frames, spatial_tokens)
        token_gate[:, :, -1] = relevance.to(hidden_states.dtype)
        token_gate = token_gate.reshape(batch_views, sequence_length, 1)
        return hidden_states + torch.tanh(self.write_gate) * token_gate * delta


@maybe_allow_in_graph
class LTXVideoTransformer3DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):
    r"""
    A Transformer model for video-like data used in [LTX](https://huggingface.co/Lightricks/LTX-Video).

    Args:
        in_channels (`int`, defaults to `128`):
            The number of channels in the input.
        out_channels (`int`, defaults to `128`):
            The number of channels in the output.
        patch_size (`int`, defaults to `1`):
            The size of the spatial patches to use in the patch embedding layer.
        patch_size_t (`int`, defaults to `1`):
            The size of the tmeporal patches to use in the patch embedding layer.
        num_attention_heads (`int`, defaults to `32`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        cross_attention_dim (`int`, defaults to `2048 `):
            The number of channels for cross attention heads.
        num_layers (`int`, defaults to `28`):
            The number of layers of Transformer blocks to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        qk_norm (`str`, defaults to `"rms_norm_across_heads"`):
            The normalization layer to use.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 2048,
        num_layers: int = 28,
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 4096,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        use_view_embed: bool = True,
        max_view: int = 3,
        action_expert: bool = False,
        value_expert: bool = False,
        value_direct_regression: bool = False,
        grounding_expert: bool = False,
        grounding_future_attn_mask: bool = False,
        grounding_future_frames: int = 2,
        use_goal_heatmap_head: bool = False,
        use_goal_latent_heatmap_head: bool = False,
        goal_heatmap_out_hw: int = 32,
        goal_heatmap_num_frames: int = 2,
        use_goal_query_adapter: bool = False,
        goal_query_writeback: bool = False,
        goal_query_dim: int = 512,
        goal_query_num_heads: int = 8,
        goal_query_read_layer: int = 23,
        goal_query_write_start_layer: int = 24,
        goal_query_spatial_gate_max: float = 4.0,
        **kwargs,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim

        # TODO: replace it with our dim if needed
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.time_embed = AdaLayerNormSingle(inner_dim, use_additional_conditions=False)

        self.use_view_embed = use_view_embed
        if self.use_view_embed:
            self.view_embed = nn.Parameter(torch.randn(max_view, inner_dim))
            self.view_ada = nn.Sequential(
                                            nn.SiLU(),
                                            nn.Linear(inner_dim, 6 * inner_dim, bias=True)
                                        )

        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)

        self.rope = LTXVideoRotaryPosEmbed(
            dim=inner_dim,
            base_num_frames=20,
            base_height=2048,
            base_width=2048,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            theta=10000.0,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                LTXVideoTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels)

        self.gradient_checkpointing = False


        self.action_expert = action_expert
        self.value_expert = value_expert
        self.value_direct_regression = bool(value_direct_regression)
        self.grounding_expert = grounding_expert
        self.grounding_future_attn_mask = grounding_future_attn_mask
        self.grounding_future_frames = grounding_future_frames
        self.use_goal_heatmap_head = use_goal_heatmap_head
        if use_goal_heatmap_head:
            self.goal_heatmap_head = GoalHeatmapHead(
                in_dim=inner_dim,
                out_hw=goal_heatmap_out_hw,
                num_frames=goal_heatmap_num_frames,
            )
        self.use_goal_latent_heatmap_head = use_goal_latent_heatmap_head
        if use_goal_latent_heatmap_head:
            self.goal_latent_heatmap_head = GoalHeatmapHead(
                in_dim=in_channels,
                out_hw=goal_heatmap_out_hw,
                num_frames=goal_heatmap_num_frames,
            )
        self.use_goal_query_adapter = use_goal_query_adapter
        self.goal_query_writeback = goal_query_writeback
        self.goal_query_read_layer = int(goal_query_read_layer)
        self.goal_query_write_start_layer = int(goal_query_write_start_layer)
        if use_goal_query_adapter:
            if not 0 <= self.goal_query_read_layer < num_layers:
                raise ValueError(
                    f"goal_query_read_layer={self.goal_query_read_layer} is outside {num_layers} blocks"
                )
            if self.goal_query_write_start_layer <= self.goal_query_read_layer:
                raise ValueError(
                    "goal_query_write_start_layer must be after goal_query_read_layer"
                )
            self.goal_query_adapter = GoalQueryAdapter(
                video_dim=inner_dim,
                goal_dim=goal_query_dim,
                num_heads=goal_query_num_heads,
                num_queries=goal_heatmap_num_frames,
                out_hw=goal_heatmap_out_hw,
                spatial_gate_max=goal_query_spatial_gate_max,
            )

        if self.action_expert:
            has_grounding_cross_attention = bool(
                kwargs.get("action_grounding_cross_attention", False)
            )
            has_value_cross_attention = bool(
                kwargs.get("action_value_cross_attention", self.value_expert)
            ) or has_grounding_cross_attention

            if has_grounding_cross_attention:
                value_inner_dim = kwargs.get('grounding_num_attention_heads', 16) * kwargs.get('grounding_attention_head_dim', 32)
            elif has_value_cross_attention:
                value_inner_dim = kwargs.get('value_num_attention_heads', 16) * kwargs.get('value_attention_head_dim', 32)
            else:
                value_inner_dim = 512  # default for the optional action conditioning attention.

            add_action_expert(
                self,
                num_layers=num_layers,
                inner_dim=inner_dim,
                activation_fn=activation_fn,
                norm_eps=norm_eps,
                attention_bias=attention_bias,
                norm_elementwise_affine=norm_elementwise_affine,
                attention_out_bias=attention_out_bias,
                qk_norm=qk_norm,
                attention_class=Attention,
                attention_processor=LTXVideoAttentionProcessor2_0(),
                has_value_cross_attention=has_value_cross_attention,
                value_inner_dim=value_inner_dim,
                has_grounding_cross_attention=has_grounding_cross_attention,
                **kwargs
            )

        if self.grounding_expert:
            add_grounding_expert(
                self,
                num_layers=num_layers,
                inner_dim=inner_dim,
                activation_fn=activation_fn,
                norm_eps=norm_eps,
                attention_bias=attention_bias,
                norm_elementwise_affine=norm_elementwise_affine,
                attention_out_bias=attention_out_bias,
                qk_norm=qk_norm,
                attention_class=Attention,
                attention_processor=LTXVideoAttentionProcessor2_0(),
                **kwargs
            )

        if self.value_expert:
            add_value_expert(
                self,
                num_layers=num_layers,
                inner_dim=inner_dim,
                activation_fn=activation_fn,
                norm_eps=norm_eps,
                attention_bias=attention_bias,
                norm_elementwise_affine=norm_elementwise_affine,
                attention_out_bias=attention_out_bias,
                qk_norm=qk_norm,
                attention_class=Attention,
                attention_processor=LTXVideoAttentionProcessor2_0(),
                **kwargs
            )


    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def _build_grounding_future_mask(self, final_hidden_states, n_view, num_frames, num_grounding_tokens):
        """Additive attention mask for grounding cross-attention so it reads ONLY the
        predicted (future) latent video frames; memory-frame keys are masked with -inf.
        Assumes video token layout = n_view x num_frames x spat (frame-major inside view).
        Returns [1, 1, num_grounding_tokens, K] (broadcast over batch/heads)."""
        B, K, _ = final_hidden_states.shape
        l = max(K // max(n_view, 1), 1)                 # per-view tokens
        spat = max(l // max(num_frames or 1, 1), 1)     # spatial tokens per frame per view
        mem_frames = max(int(num_frames) - int(getattr(self, "grounding_future_frames", 2)), 0)
        idx = torch.arange(K, device=final_hidden_states.device)
        frame_idx = (idx % l) // spat                   # frame index within each view
        allow = frame_idx >= mem_frames                 # future frames allowed
        mask = torch.where(allow, 0.0, float("-inf")).to(dtype=torch.float32)
        return mask[None, None, None, :].expand(B, 1, num_grounding_tokens, K).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        encoder_attention_mask: torch.Tensor = None,
        n_view: int = 1,
        rope_interpolation_scale: Optional[Tuple[float, float, float]] = None,
        return_dict: bool = True,
        action_states: torch.Tensor = None,
        action_timestep: torch.LongTensor = None,
        return_video: bool = True,
        return_action: bool = False,
        value_states: torch.Tensor = None,
        value_timestep: torch.LongTensor = None,
        return_value: bool = False,
        grounding_states: torch.Tensor = None,
        grounding_timestep: torch.LongTensor = None,
        return_grounding: bool = False,
        store_buffer=False,
        video_states_buffer=None,
        value_store_buffer=False,
        value_states_buffer=None,
        grounding_store_buffer=False,
        grounding_states_buffer=None,
        video_attention_mask: torch.Tensor = None,
        history_action_state: torch.Tensor = None,
        history_value_state: torch.Tensor = None,
        history_grounding_state: torch.Tensor = None,
        value_pose_state: torch.Tensor = None,
        value_pose_positions: torch.Tensor = None,
        grounding_pose_state: torch.Tensor = None,
        grounding_pose_positions: torch.Tensor = None,
        frame_ids: torch.Tensor = None,
        num_frames: int = None,
        height: int = None,
        width: int = None,
        **kwargs,
    ) -> torch.Tensor:

        input_video_latents = hidden_states

        if return_video or store_buffer:

            if store_buffer:
                video_states_buffer = []

            input_video_latents = hidden_states

            image_rotary_emb = self.rope(
                hidden_states, rope_interpolation_scale, num_frames, height, width, frame_ids=frame_ids
            )

            # convert encoder_attention_mask to a bias the same way we do for attention_mask
            if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
                encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)   # shape b, 1, l_k
            if video_attention_mask is not None and video_attention_mask.ndim == 2:
                video_attention_mask = (1 - video_attention_mask.to(hidden_states.dtype)) * -10000.0
                video_attention_mask = video_attention_mask.unsqueeze(0)  # shape 1, l_q, l_k

            batch_size = hidden_states.size(0)
            hidden_states = self.proj_in(hidden_states)

            temb, embedded_timestep = self.time_embed(
                timestep.flatten(),
                batch_size=batch_size,
                hidden_dtype=hidden_states.dtype,
            )

            temb = temb.view(batch_size, -1, temb.size(-1))

            embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

            if self.use_view_embed:
                embedded_view = self.view_embed[:n_view].unsqueeze(0).repeat(batch_size//n_view, 1, 1)
                embedded_view = rearrange(embedded_view, 'b v c -> (b v) c').unsqueeze(1)
                vemb = self.view_ada(embedded_view)
                temb = temb + vemb
                embedded_timestep = embedded_timestep + embedded_view

            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(hidden_states.size(0) // n_view, -1, hidden_states.size(-1))

        if return_action:
            ### when video_states_buffer is not None, action blocks will directly use the input buffers
            ### when video_states_buffer is None, store_buffer should be true to save video buffers
            if video_states_buffer is None:
                assert store_buffer or return_video
            if history_action_state is not None:
                action_states = torch.cat((history_action_state, action_states), dim=1)
                action_timestep = torch.cat((torch.zeros_like(action_timestep[:,0:1]), action_timestep), dim=1)
            action_temb, action_embedded_timestep, action_rotary_emb, action_hidden_states = preprocessing_action_states(self, action_states, action_timestep)

        if return_value:
            ### when video_states_buffer is not None, value blocks will directly use the input buffers
            ### when video_states_buffer is None, store_buffer should be true to save video buffers
            if video_states_buffer is None:
                assert store_buffer or return_video
            if value_store_buffer:
                value_states_buffer = []

            if history_value_state is not None:
                value_states = torch.cat((history_value_state, value_states), dim=1)
                value_timestep = torch.cat((torch.zeros_like(value_timestep[:,0:1]), value_timestep), dim=1)
            value_temb, value_embedded_timestep, value_rotary_emb, value_hidden_states = preprocessing_value_states(
                self, value_states, value_timestep,
                pose_state=value_pose_state, pose_positions=value_pose_positions,
            )

            # ---- v3: goal-latent KV for value tower attn3 (clean video latent = target signal) ----
            value_goal_latent_kv = None
            if getattr(self, "value_goal_latent_cross_attention", False):
                # value query batch = b (no view dim). input_video_latents is [(b v), l, c],
                # so re-batch to (b, v, l, c) and keep the first view (the current view).
                _lf = int(num_frames or 1)
                _l = input_video_latents.shape[1]
                _sp = _l // _lf
                if _sp * _lf != _l:
                    raise ValueError("value_goal_latent_kv: cannot split latents into frames")
                _bv = input_video_latents.shape[0]
                _b = _bv // n_view
                _gl = input_video_latents.reshape(_bv, _lf, _sp, -1)      # [(b v), lf, spat, c]
                _gl = _gl.reshape(_b, n_view, _lf, _sp, -1)[:, 0, -1]     # [b, spat, c] first view, last frame
                value_goal_latent_kv = self.value_goal_latent_proj_in(
                    _gl.to(self.value_goal_latent_proj_in.weight.dtype)
                )  # [b, spat, value_inner_dim]

        if return_grounding:
            if video_states_buffer is None:
                assert store_buffer or return_video
            if grounding_store_buffer:
                grounding_states_buffer = []

            if history_grounding_state is not None:
                grounding_states = torch.cat((history_grounding_state, grounding_states), dim=1)
                grounding_timestep = torch.cat((torch.zeros_like(grounding_timestep[:,0:1]), grounding_timestep), dim=1)
            grounding_temb, grounding_embedded_timestep, grounding_rotary_emb, grounding_hidden_states = preprocessing_grounding_states(
                self, grounding_states, grounding_timestep,
                pose_state=grounding_pose_state, pose_positions=grounding_pose_positions,
            )

        goal_query_tokens = None
        goal_query_heatmap = None
        goal_query_attention = None

        for block_idx, block in enumerate(self.transformer_blocks):
            if (
                self.use_goal_query_adapter
                and self.goal_query_writeback
                and goal_query_tokens is not None
                and block_idx >= self.goal_query_write_start_layer
                and (return_video or store_buffer)
            ):
                hidden_states = self.goal_query_adapter.write(
                    hidden_states,
                    goal_query_tokens,
                    goal_query_attention,
                    n_view=n_view,
                    num_frames=int(num_frames or 1),
                )
            
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                
                if return_video or store_buffer:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        encoder_attention_mask,
                        n_view,
                        # TODO: we always set cross_view_attn=True in this case
                        block_idx%3==0,
                        video_attention_mask,
                        **ckpt_kwargs,
                    )
                    if store_buffer:
                        video_states_buffer.append(hidden_states.clone())
                else:
                    hidden_states = video_states_buffer[block_idx]
                
                if (
                    self.use_goal_query_adapter
                    and block_idx == self.goal_query_read_layer
                    and (return_video or store_buffer)
                ):
                    goal_query_tokens, goal_query_heatmap, goal_query_attention = (
                        self.goal_query_adapter.read(
                            hidden_states,
                            encoder_hidden_states,
                            encoder_attention_mask,
                            n_view=n_view,
                            num_frames=int(num_frames or 1),
                        )
                    )

                if return_value:
                    ### final_hidden_states:  video features, b (v t h w) c
                    ### value_hidden_states: random values, b t c
                    ###
                    if not value_store_buffer and value_states_buffer is not None:
                        value_hidden_states = value_states_buffer[block_idx]
                    else:
                        final_hidden_states = rearrange(hidden_states, '(b v) l c -> b (v l) c', v=n_view)
                        value_hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(self.value_blocks[block_idx]),
                            value_hidden_states,
                            final_hidden_states,
                            value_temb,
                            value_rotary_emb,
                            None,
                            value_goal_latent_kv,
                            **ckpt_kwargs,
                        )
                        if value_store_buffer:
                            value_states_buffer.append(value_hidden_states.clone())

                if return_grounding:
                    if not grounding_store_buffer and grounding_states_buffer is not None:
                        grounding_hidden_states = grounding_states_buffer[block_idx]
                    else:
                        final_hidden_states = rearrange(hidden_states, '(b v) l c -> b (v l) c', v=n_view)
                        self.grounding_blocks[block_idx]._future_mask = (
                            self._build_grounding_future_mask(final_hidden_states, n_view, num_frames, grounding_hidden_states.shape[1])
                            if self.grounding_future_attn_mask else None
                        )
                        grounding_hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(self.grounding_blocks[block_idx]),
                            grounding_hidden_states,
                            final_hidden_states,
                            grounding_temb,
                            grounding_rotary_emb,
                            None,
                            **ckpt_kwargs,
                        )
                        if grounding_store_buffer:
                            grounding_states_buffer.append(grounding_hidden_states.clone())

                if return_action:
                    ### final_hidden_states:  video features, b (v t h w) c
                    ### action_hidden_states: random actions, b v c
                    ###
                    final_hidden_states = rearrange(hidden_states, '(b v) l c -> b (v l) c', v=n_view)

                    if getattr(self, "action_grounding_cross_attention", False) and return_grounding:
                        value_features = grounding_hidden_states
                    elif return_value:
                        value_features = value_hidden_states
                    else:
                        value_features = None

                    action_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.action_blocks[block_idx]),
                        action_hidden_states,
                        final_hidden_states,
                        action_temb,
                        action_rotary_emb,
                        None,
                        value_features,
                        **ckpt_kwargs,
                    )

                
            else:
                if return_video or store_buffer:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        encoder_attention_mask=encoder_attention_mask,
                        n_view=n_view,
                        # TODO: we always set cross_view_attn=True in this case
                        cross_view_attn=block_idx%3==0,
                        self_attention_mask=video_attention_mask,
                    )
                    if store_buffer:
                        video_states_buffer.append(hidden_states.clone())
                else:
                    hidden_states = video_states_buffer[block_idx]
                
                if (
                    self.use_goal_query_adapter
                    and block_idx == self.goal_query_read_layer
                    and (return_video or store_buffer)
                ):
                    goal_query_tokens, goal_query_heatmap, goal_query_attention = (
                        self.goal_query_adapter.read(
                            hidden_states,
                            encoder_hidden_states,
                            encoder_attention_mask,
                            n_view=n_view,
                            num_frames=int(num_frames or 1),
                        )
                    )

                if return_value:
                    ### final_hidden_states:  video features, b (v t h w) c
                    ### value_hidden_states: random values, b v c
                    ###
                    if not value_store_buffer and value_states_buffer is not None:
                        value_hidden_states = value_states_buffer[block_idx]
                    else:
                        final_hidden_states = rearrange(hidden_states, '(b v) l c -> b (v l) c', v=n_view)

                        value_hidden_states = self.value_blocks[block_idx](
                            hidden_states=value_hidden_states,
                            encoder_hidden_states=final_hidden_states,
                            temb=value_temb,
                            rotary_emb=value_rotary_emb,
                            attn3_hidden_states=value_goal_latent_kv,
                        )
                        if value_store_buffer:
                            value_states_buffer.append(value_hidden_states.clone())

                if return_grounding:
                    if not grounding_store_buffer and grounding_states_buffer is not None:
                        grounding_hidden_states = grounding_states_buffer[block_idx]
                    else:
                        final_hidden_states = rearrange(hidden_states, '(b v) l c -> b (v l) c', v=n_view)
                        self.grounding_blocks[block_idx]._future_mask = (
                            self._build_grounding_future_mask(final_hidden_states, n_view, num_frames, grounding_hidden_states.shape[1])
                            if self.grounding_future_attn_mask else None
                        )

                        grounding_hidden_states = self.grounding_blocks[block_idx](
                            hidden_states=grounding_hidden_states,
                            encoder_hidden_states=final_hidden_states,
                            temb=grounding_temb,
                            rotary_emb=grounding_rotary_emb,
                        )
                        if grounding_store_buffer:
                            grounding_states_buffer.append(grounding_hidden_states.clone())
                    
                if return_action:
                    ### final_hidden_states:  video features, b (v t h w) c
                    ### action_hidden_states: random actions, b v c
                    ###
                    final_hidden_states = rearrange(hidden_states, '(b v) l c -> b (v l) c', v=n_view)

                    if getattr(self, "action_grounding_cross_attention", False) and return_grounding:
                        value_features = grounding_hidden_states
                    elif return_value:
                        value_features = value_hidden_states
                    else:
                        value_features = None

                    action_hidden_states = self.action_blocks[block_idx](
                        hidden_states=action_hidden_states,
                        encoder_hidden_states=final_hidden_states,
                        temb=action_temb,
                        rotary_emb=action_rotary_emb,
                        value_hidden_states=value_features,
                    )

                

                    


        final_output = {}

        if store_buffer:
            final_output['video_states_buffer'] = video_states_buffer
        if value_store_buffer:
            final_output['value_states_buffer'] = value_states_buffer
        if grounding_store_buffer:
            final_output['grounding_states_buffer'] = grounding_states_buffer
        if goal_query_tokens is not None:
            final_output['goal_tokens'] = goal_query_tokens
            final_output['goal_query_heatmap'] = goal_query_heatmap
            if not self.use_goal_heatmap_head:
                final_output['goal_heatmap'] = rearrange(
                    goal_query_heatmap, "b v t h w -> (b v) t h w"
                )
            
        if return_video:
            scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
            shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

            hidden_states = self.norm_out(hidden_states)
            hidden_states = hidden_states * (1 + scale) + shift
            output = self.proj_out(hidden_states)

            final_output['video'] = output
            if self.use_goal_heatmap_head:
                latent_frames = int(num_frames or 1)
                sequence_length = hidden_states.shape[1]
                spatial_tokens = sequence_length // latent_frames
                if spatial_tokens * latent_frames != sequence_length:
                    raise ValueError(
                        f"Cannot split {sequence_length} tokens into {latent_frames} latent frames"
                    )
                frame_tokens = hidden_states.reshape(
                    hidden_states.shape[0], latent_frames, spatial_tokens, hidden_states.shape[-1]
                )
                # Two raw future frames are temporally compressed into the final VAE latent.
                final_output['goal_heatmap'] = self.goal_heatmap_head(frame_tokens[:, -1])

            if self.use_goal_latent_heatmap_head:
                latent_frames = int(num_frames or 1)
                sequence_length = output.shape[1]
                spatial_tokens = sequence_length // latent_frames
                if spatial_tokens * latent_frames != sequence_length:
                    raise ValueError(
                        f"Cannot split {sequence_length} tokens into {latent_frames} latent frames"
                    )
                if input_video_latents.shape != output.shape:
                    raise ValueError(
                        "Clean-latent reconstruction requires matching video input/output shapes, "
                        f"got input={input_video_latents.shape}, output={output.shape}"
                    )

                sigma = timestep.to(device=output.device, dtype=output.dtype) / 1000.0
                if sigma.ndim == 1:
                    sigma = sigma[:, None]
                if sigma.shape[1] == 1:
                    sigma = sigma.expand(-1, sequence_length)
                elif sigma.shape[1] == latent_frames:
                    sigma = sigma.repeat_interleave(spatial_tokens, dim=1)
                elif sigma.shape[1] != sequence_length:
                    raise ValueError(
                        "Unsupported video timestep shape for clean-latent reconstruction: "
                        f"timestep={timestep.shape}, video={output.shape}, frames={latent_frames}"
                    )

                # Flow matching uses z_t=(1-sigma)*z_0+sigma*eps and v=eps-z_0.
                predicted_clean_latents = input_video_latents - sigma.unsqueeze(-1) * output
                clean_frame_tokens = predicted_clean_latents.reshape(
                    predicted_clean_latents.shape[0],
                    latent_frames,
                    spatial_tokens,
                    predicted_clean_latents.shape[-1],
                )
                final_output['goal_latent_heatmap'] = self.goal_latent_heatmap_head(
                    clean_frame_tokens[:, -1]
                )

        if return_action:
            if self.action_final_embeddings:
                action_scale_shift_values = self.action_scale_shift_table[None, None] + action_embedded_timestep[:, :, None]
                action_shift, action_scale = action_scale_shift_values[:,:,0], action_scale_shift_values[:,:,1]
                action_hidden_states = self.action_norm_out(action_hidden_states)
                action_hidden_states = action_hidden_states * (1 + action_scale) + action_shift
            else:
                action_hidden_states = self.action_norm_out(action_hidden_states)
                action_hidden_states = self.action_proj_extra(action_hidden_states)
            if history_action_state is not None:
                action_hidden_states = action_hidden_states[:, 1:]

            action_output = self.action_proj_out(action_hidden_states)

            final_output['action'] = action_output

            # If discrete classification head is present, also return class logits
            if getattr(self, 'action_discrete', False):
                class_logits = self.action_class_head(action_hidden_states)
                # Reshape to (b, t, num_active_dims, num_classes_per_dim)
                b, t, _ = class_logits.shape
                class_logits = class_logits.view(b, t, self.action_num_active_dims, self.action_num_classes_per_dim)
                final_output['action_class_logits'] = class_logits

        if return_value:
            if self.value_final_embeddings:
                value_scale_shift_values = self.value_scale_shift_table[None, None] + value_embedded_timestep[:, :, None]
                value_shift, value_scale = value_scale_shift_values[:,:,0], value_scale_shift_values[:,:,1]
                value_hidden_states = self.value_norm_out(value_hidden_states)
                value_hidden_states = value_hidden_states * (1 + value_scale) + value_shift
            else:
                value_hidden_states = self.value_norm_out(value_hidden_states)
                value_hidden_states = self.value_proj_extra(value_hidden_states)
            if history_value_state is not None:
                value_hidden_states = value_hidden_states[:, 1:]
            if value_pose_state is not None:
                value_hidden_states = value_hidden_states[:, value_pose_state.shape[1]:]

            value_output = self.value_proj_out(value_hidden_states)

            final_output['value'] = value_output

        if return_grounding:
            if self.grounding_final_embeddings:
                grounding_scale_shift_values = self.grounding_scale_shift_table[None, None] + grounding_embedded_timestep[:, :, None]
                grounding_shift, grounding_scale = grounding_scale_shift_values[:,:,0], grounding_scale_shift_values[:,:,1]
                grounding_hidden_states = self.grounding_norm_out(grounding_hidden_states)
                grounding_hidden_states = grounding_hidden_states * (1 + grounding_scale) + grounding_shift
            else:
                grounding_hidden_states = self.grounding_norm_out(grounding_hidden_states)
                grounding_hidden_states = self.grounding_proj_extra(grounding_hidden_states)
            if history_grounding_state is not None:
                grounding_hidden_states = grounding_hidden_states[:, 1:]
            if grounding_pose_state is not None:
                grounding_hidden_states = grounding_hidden_states[:, grounding_pose_state.shape[1]:]

            grounding_output = self.grounding_proj_out(grounding_hidden_states)

            final_output['grounding'] = grounding_output

        if not return_dict:
            return (final_output,)

        return Transformer2DModelOutput(sample=final_output)


def apply_rotary_emb(x, freqs):
    cos, sin = freqs
    batch_size = x.shape[0]
    if cos.shape[0] == 1 and batch_size > 1:
        cos = cos.repeat(batch_size, 1, 1)
    if sin.shape[0] == 1 and batch_size > 1:
        sin = sin.repeat(batch_size, 1, 1)
    
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)  # [B, S, H, D // 2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out
