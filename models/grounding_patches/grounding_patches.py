import math
from typing import Any, Dict, Optional, Tuple


import torch
import torch.nn as nn

from diffusers.models.attention import FeedForward
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph


class GroundingRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        base_seq_length: int = 57,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.base_seq_length = base_seq_length
        self.theta = theta

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_length: int,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Always compute rope in fp32
        if positions is not None:
            # Explicit per-token positions (e.g. real frame ids / time deltas)
            # instead of the default sequential arange -- shape [1, seq_length]
            # or [B, seq_length], broadcastable below.
            grid = positions.to(dtype=torch.float32, device=hidden_states.device)
            if grid.dim() == 1:
                grid = grid.unsqueeze(0)
        else:
            grid = torch.arange(seq_length, dtype=torch.float32, device=hidden_states.device).unsqueeze(0)

        grid = grid / self.base_seq_length

        grid = grid.unsqueeze(-1)

        start = 1.0
        end = self.theta
        freqs = self.theta ** torch.linspace(
            math.log(start, self.theta),
            math.log(end, self.theta),
            self.dim // 2,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        freqs = freqs * math.pi / 2.0
        freqs = freqs * (grid * 2 - 1)

        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        if self.dim % 2 != 0:
            cos_padding = torch.ones_like(cos_freqs[:, :, : self.dim % 2])
            sin_padding = torch.zeros_like(sin_freqs[:, :, : self.dim % 2])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        return cos_freqs, sin_freqs




@maybe_allow_in_graph
class GroundingTransformerBlock(nn.Module):
    r"""
    Modified from Transformer block used in [LTX](https://huggingface.co/Lightricks/LTX-Video).

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
            Epsilon grounding for normalization layers.
    """

    def __init__(
        self,
        attention_class,
        attention_args,
        dim: int = 512,
        num_attention_heads: int = 16,
        attention_head_dim: int = 32,
        cross_attention_dim: int = 2048,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        attn3_cross_attention_dim = 2048,
        num_latent_downsample_block = 0,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = attention_class(
            **(attention_args[0]),
        )

        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = attention_class(
            **(attention_args[1]),
        )

        self.ff = FeedForward(dim, activation_fn=activation_fn)

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        self.num_latent_downsample_block = num_latent_downsample_block
        if self.num_latent_downsample_block > 0:
            self.latent_downsample_block = nn.ModuleList()
            for _i in range(self.num_latent_downsample_block):
                self.latent_downsample_block.append(
                    downsampling_block()
                )


    def _future_masked_cross_attn(self, attn, hidden_states, encoder_hidden_states, mask):
        """Manual cross-attention (mirrors LTXVideoAttentionProcessor2_0) that masks
        memory-frame video keys with -inf, so grounding reads ONLY the predicted
        (future) frames. Video-branch attention is untouched."""
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        scores = (query.float() @ key.float().transpose(-1, -2)) * (query.shape[-1] ** -0.5)
        if mask is not None:
            scores = scores + mask.to(device=scores.device, dtype=scores.dtype)
        attn_probs = scores.softmax(dim=-1).to(hidden_states.dtype)
        out = attn_probs @ value
        out = out.transpose(1, 2).flatten(2, 3)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attn3_hidden_states: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        norm_hidden_states = self.norm1(hidden_states)

        num_ada_params = self.scale_shift_table.shape[0]
        ada_groundings = self.scale_shift_table[None, None] + temb.reshape(batch_size, temb.size(1), num_ada_params, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_groundings.unbind(dim=2)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        
        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            image_rotary_emb=rotary_emb,
            n_view=1,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        
        if getattr(self, "_future_mask", None) is not None:
            # Future-frame-only cross-attention (grounding reads predicted video frames,
            # memory frames are masked out with -inf). Set per-forward by the transformer.
            attn_hidden_states = self._future_masked_cross_attn(
                self.attn2, hidden_states, encoder_hidden_states, self._future_mask
            )
        else:
            attn_hidden_states = self.attn2(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=None,
                attention_mask=encoder_attention_mask,
                n_view=1,
            )
        hidden_states = hidden_states + attn_hidden_states

        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp
        

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp
        

        return hidden_states



def add_grounding_expert(
    self,
    num_layers: int = 28,
    inner_dim: int = 2048,
    activation_fn: str = "gelu",
    norm_eps: float = 1e-6,
    grounding_in_channels: int = 14,
    grounding_out_channels: int = None,
    grounding_num_attention_heads: int = 16,
    grounding_attention_head_dim: int = 32,
    grounding_rope_dim: int = None,
    grounding_final_embeddings: bool = True,
    learnable_grounding_state: bool = False,
    norm_elementwise_affine: bool = False,
    attention_bias: bool = True,
    attention_out_bias: bool = True,
    qk_norm: str = "rms_norm_across_heads",
    attention_class = None,
    attention_processor = None,
    **kwargs,
):

    if grounding_out_channels is None:
        grounding_out_channels = grounding_in_channels

    self.grounding_inner_dim = grounding_num_attention_heads * grounding_attention_head_dim

    self.learnable_grounding_state = learnable_grounding_state
    if self.learnable_grounding_state:
        self.grounding_state = nn.Parameter(torch.randn(1, 1, grounding_in_channels))

    self.grounding_proj_in = nn.Linear(grounding_in_channels, self.grounding_inner_dim)
    # Start/history pose conditioning (pose extended to 14-dim + 1 time-delta scalar).
    self.grounding_pose_proj_in = nn.Linear(15, self.grounding_inner_dim)
    self.grounding_scale_shift_table = nn.Parameter(torch.randn(2, self.grounding_inner_dim) / self.grounding_inner_dim**0.5)
    self.grounding_time_embed = AdaLayerNormSingle(self.grounding_inner_dim, use_additional_conditions=False)

    if grounding_rope_dim is None:
        grounding_rope_dim = self.grounding_inner_dim
    # set to a fixed grounding currently, should adjust according to the action length
    self.grounding_rope = GroundingRotaryPosEmbed(
        dim=grounding_rope_dim,
        base_seq_length=57,
        theta=10000.0,
    )

    attention_args = []
    attention_args.append(dict(
        query_dim=self.grounding_inner_dim,
        heads=grounding_num_attention_heads,
        kv_heads=grounding_num_attention_heads,
        dim_head=grounding_attention_head_dim,
        bias=attention_bias,
        cross_attention_dim=None,
        out_bias=attention_out_bias,
        qk_norm=qk_norm,
        processor=attention_processor,
    ))
    attention_args.append(dict(
        query_dim=self.grounding_inner_dim,
        heads=grounding_num_attention_heads,
        kv_heads=grounding_num_attention_heads,
        dim_head=grounding_attention_head_dim,
        bias=attention_bias,
        cross_attention_dim=inner_dim,
        out_bias=attention_out_bias,
        qk_norm=qk_norm,
        processor=attention_processor,
    ))

    self.grounding_blocks = nn.ModuleList(
        [
            GroundingTransformerBlock(
                attention_class = attention_class,
                attention_args = attention_args,
                dim=self.grounding_inner_dim,
                num_attention_heads=grounding_num_attention_heads,
                attention_head_dim=grounding_attention_head_dim,
                cross_attention_dim=inner_dim,
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

    self.grounding_proj_out = nn.Linear(self.grounding_inner_dim, grounding_out_channels) 
    self.grounding_final_embeddings = grounding_final_embeddings
    if not self.grounding_final_embeddings:
        self.grounding_proj_extra = nn.Linear(self.grounding_inner_dim, self.grounding_inner_dim)

    self.grounding_norm_out = nn.LayerNorm(self.grounding_inner_dim, eps=1e-6, elementwise_affine=False)


def preprocessing_grounding_states(
    self,
    grounding_states: torch.Tensor = None,
    grounding_timestep: torch.LongTensor = None, #[B,grounding_seq_length]
    pose_state: torch.Tensor = None,        #[B,pose_seq_length,15] start/history pose + time-delta
    pose_positions: torch.Tensor = None,    #[B,pose_seq_length] or [pose_seq_length] real frame ids for pose tokens' rope
):

    assert self.grounding_expert == True
    assert grounding_states is not None and grounding_timestep is not None

    batch_size = grounding_states.shape[0]

    grounding_seq_length = grounding_states.shape[1]
    if getattr(self, "learnable_grounding_state") and self.learnable_grounding_state:
        grounding_states = self.grounding_state.repeat(batch_size, grounding_seq_length, 1).to(dtype=grounding_states.dtype, device=grounding_states.device)

    grounding_rotary_emb = self.grounding_rope(grounding_states, grounding_seq_length)
    grounding_states = grounding_states.to(self.grounding_proj_in.weight.dtype)
    grounding_hidden_states = self.grounding_proj_in(grounding_states)

    if pose_state is not None:
        pose_seq_length = pose_state.shape[1]
        pose_state = pose_state.to(self.grounding_pose_proj_in.weight.dtype)
        pose_hidden_states = self.grounding_pose_proj_in(pose_state)
        pose_rotary_emb = self.grounding_rope(pose_state, pose_seq_length, positions=pose_positions)

        grounding_hidden_states = torch.cat([pose_hidden_states, grounding_hidden_states], dim=1)
        rope_batch_size = max(pose_rotary_emb[0].shape[0], grounding_rotary_emb[0].shape[0])

        def _broadcast_rope(t):
            if t.shape[0] == 1 and rope_batch_size > 1:
                return t.repeat(rope_batch_size, 1, 1)
            return t

        grounding_rotary_emb = (
            torch.cat([_broadcast_rope(pose_rotary_emb[0]), _broadcast_rope(grounding_rotary_emb[0])], dim=1),
            torch.cat([_broadcast_rope(pose_rotary_emb[1]), _broadcast_rope(grounding_rotary_emb[1])], dim=1),
        )
        grounding_timestep = torch.cat(
            [torch.zeros_like(grounding_timestep[:, :1]).expand(-1, pose_seq_length), grounding_timestep],
            dim=1,
        )

    grounding_temb, grounding_embedded_timestep = self.grounding_time_embed(
        grounding_timestep.flatten(),
        batch_size=batch_size,
        hidden_dtype=grounding_hidden_states.dtype,
    )

    grounding_temb = grounding_temb.view(batch_size, -1, grounding_temb.size(-1))
    grounding_embedded_timestep = grounding_embedded_timestep.view(batch_size, -1, grounding_embedded_timestep.size(-1))
    
    return grounding_temb, grounding_embedded_timestep, grounding_rotary_emb, grounding_hidden_states