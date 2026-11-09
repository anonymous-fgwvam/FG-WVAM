import os, random, math
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from typing import Any, Dict, List

from datetime import datetime, timedelta
import argparse
import json
import importlib
# ----------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib

from yaml import load, dump, Loader, Dumper
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributed as dist
from einops import rearrange
from copy import deepcopy
import transformers
import logging

# ----------------------------------------------------
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

# ----------------------------------------------------
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
# ----------------------------------------------------
from utils.model_utils import load_condition_models, load_latent_models, load_vae_models, load_diffusion_model, count_model_parameters, unwrap_model
from utils.model_utils import forward_pass
from utils.optimizer_utils import get_optimizer
from utils.memory_utils import get_memory_statistics, free_memory
from utils.dreamer_style_loss import (
    symlog,
    build_twohot_bins,
    twohot_encode
)
# ----------------------------------------------------
from torch.utils.tensorboard import SummaryWriter
from utils import init_logging, import_custom_class, save_video, zero_rank_print

# ----------------------------------------------------
from utils.data_utils import get_latents, get_text_conditions, gen_noise_from_condition_frame_latent, randn_tensor, apply_color_jitter_to_video

# ----------------------------------------------------
from utils.extra_utils import act_metric, value_metric, feature_value_metric

LOG_LEVEL = "INFO"
# LOG_LEVEL = "DEBUG"
logger = get_logger("wm_runner")
logger.setLevel(LOG_LEVEL)


def paired_value_ranking_loss(scores, pair_ids, pair_roles):
    """Logistic ranking loss for successful/failed futures of one observation."""
    scores = scores.float().reshape(-1)
    pair_ids = pair_ids.reshape(-1)
    pair_roles = pair_roles.float().reshape(-1)
    losses = []
    for pair_id in torch.unique(pair_ids):
        mask = pair_ids == pair_id
        positive = scores[mask & (pair_roles > 0.5)]
        negative = scores[mask & (pair_roles <= 0.5)]
        if positive.numel() and negative.numel():
            losses.append(F.softplus(-(positive.mean() - negative.mean())))
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def goal_heatmap_distribution_loss(logits, targets, visible, confidence):
    """KL alignment over one fused view-by-space distribution per future frame."""
    if logits.ndim != 5 or targets.ndim != 5:
        raise ValueError(
            f"Expected [B,V,T,H,W] heatmaps, got logits={logits.shape}, targets={targets.shape}"
        )
    if logits.shape[:3] != targets.shape[:3]:
        raise ValueError(
            f"Goal heatmap time/view mismatch: logits={logits.shape[:3]}, targets={targets.shape[:3]}"
        )
    if visible.shape != logits.shape[:3] or confidence.shape != logits.shape[:3]:
        raise ValueError(
            f"Goal heatmap mask mismatch: logits={logits.shape[:3]}, "
            f"visible={visible.shape}, confidence={confidence.shape}"
        )

    if targets.shape[-2:] != logits.shape[-2:]:
        batch, views, frames, height, width = targets.shape
        targets = F.interpolate(
            targets.reshape(batch * views * frames, 1, height, width),
            size=logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, views, frames, *logits.shape[-2:])

    confidence = torch.nan_to_num(confidence.float(), nan=0.0, posinf=1.0, neginf=0.0)
    confidence = confidence.clamp(0.0, 1.0)
    visible = visible.float().clamp(0.0, 1.0)
    target_mass = targets.float().clamp_min(0.0) * (confidence * visible)[..., None, None]

    logits_flat = rearrange(logits.float(), "b v t h w -> b t (v h w)")
    target_flat = rearrange(target_mass, "b v t h w -> b t (v h w)")
    mass = target_flat.sum(dim=-1, keepdim=True)
    target_prob = target_flat / mass.clamp_min(1e-8)
    log_prob = F.log_softmax(logits_flat, dim=-1)
    target_log_prob = torch.log(target_prob.clamp_min(1e-8))
    kl = torch.where(
        target_prob > 0,
        target_prob * (target_log_prob - log_prob),
        torch.zeros_like(target_prob),
    ).sum(dim=-1)

    valid = (mass.squeeze(-1) > 1e-8).float()
    frame_weight = (confidence * visible).amax(dim=1) * valid
    spatial_normalizer = max(math.log(logits_flat.shape[-1]), 1.0)
    return ((kl / spatial_normalizer) * frame_weight).sum() / frame_weight.sum().clamp_min(1e-8)


def goal_query_distribution_loss(logits, targets, visible, confidence):
    """Supervise goal QKV attention without applying confidence more than once."""
    if logits.ndim != 5 or targets.ndim != 5:
        raise ValueError(
            f"Expected [B,V,T,H,W] heatmaps, got logits={logits.shape}, targets={targets.shape}"
        )
    if logits.shape[:3] != targets.shape[:3]:
        raise ValueError(
            f"Goal-query time/view mismatch: logits={logits.shape[:3]}, targets={targets.shape[:3]}"
        )
    if visible.shape != logits.shape[:3] or confidence.shape != logits.shape[:3]:
        raise ValueError(
            f"Goal-query mask mismatch: logits={logits.shape[:3]}, "
            f"visible={visible.shape}, confidence={confidence.shape}"
        )

    if targets.shape[-2:] != logits.shape[-2:]:
        batch, views, frames, height, width = targets.shape
        targets = F.interpolate(
            targets.reshape(batch * views * frames, 1, height, width),
            size=logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, views, frames, *logits.shape[-2:])

    targets = targets.float().clamp_min(0.0)
    visible = visible.float().clamp(0.0, 1.0)
    confidence = torch.nan_to_num(
        confidence.float(), nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)

    # Each visible view contributes one normalized spatial distribution. The
    # confidence is deliberately excluded here and is used once below.
    view_mass = targets.sum(dim=(-2, -1), keepdim=True)
    valid_view = (view_mass.squeeze(-1).squeeze(-1) > 1e-8).float() * visible
    view_prob = targets / view_mass.clamp_min(1e-8)
    view_weight = valid_view / valid_view.sum(dim=1, keepdim=True).clamp_min(1.0)
    target_prob = view_prob * view_weight[..., None, None]

    logits_flat = rearrange(logits.float(), "b v t h w -> b t (v h w)")
    target_flat = rearrange(target_prob, "b v t h w -> b t (v h w)")
    log_prob = F.log_softmax(logits_flat, dim=-1)
    target_log_prob = torch.log(target_flat.clamp_min(1e-8))
    kl = torch.where(
        target_flat > 0,
        target_flat * (target_log_prob - log_prob),
        torch.zeros_like(target_flat),
    ).sum(dim=-1)

    valid_frame = (valid_view.sum(dim=1) > 0).float()
    frame_weight = (confidence * valid_view).amax(dim=1) * valid_frame
    spatial_normalizer = max(math.log(logits_flat.shape[-1]), 1.0)
    return ((kl / spatial_normalizer) * frame_weight).sum() / frame_weight.sum().clamp_min(1e-8)


def goal_focused_video_loss(
    elementwise_video_loss,
    targets,
    visible,
    conditioning_mask,
    batch_size,
    n_view,
    latent_frames,
    latent_height,
    latent_width,
    future_latent_frames,
    lambda_h=3.0,
):
    """Target-weighted flow-matching loss.

    L = sum_i m_i w_i L_i / sum_i m_i w_i, with w_i = 1 + lambda_h * H~_i,
    where H~ in [0, 1] is the target heatmap pooled to latent resolution and
    m_i selects the non-conditioning (future) tokens.
    """
    expected_tokens = latent_frames * latent_height * latent_width
    if elementwise_video_loss.ndim != 3 or elementwise_video_loss.shape[:2] != (
        batch_size * n_view,
        expected_tokens,
    ):
        raise ValueError(
            "Goal-focused video loss shape mismatch: "
            f"video_loss={elementwise_video_loss.shape}, expected "
            f"[{batch_size * n_view},{expected_tokens},C]"
        )
    if targets.ndim != 5 or targets.shape[:2] != (batch_size, n_view):
        raise ValueError(
            f"Expected goal heatmaps [B,V,T,H,W], got {targets.shape} for "
            f"B={batch_size}, V={n_view}"
        )
    if visible.shape != targets.shape[:3]:
        raise ValueError(
            f"Goal heatmap mask mismatch: targets={targets.shape[:3]}, "
            f"visible={visible.shape}"
        )
    if not 0 < future_latent_frames <= latent_frames:
        raise ValueError(
            f"Invalid future latent count {future_latent_frames} for {latent_frames} frames"
        )
    if lambda_h < 0:
        raise ValueError(f"lambda_h must be non-negative, got {lambda_h}")

    targets = targets.detach().float().clamp_min(0.0)
    visible = visible.detach().float().clamp(0.0, 1.0)

    # Pool the raw future labels into the temporally compressed future latents.
    heat = targets * visible[..., None, None]
    heat = rearrange(heat, "b v t h w -> (b v) 1 t h w")
    heat = F.adaptive_max_pool3d(
        heat,
        output_size=(future_latent_frames, latent_height, latent_width),
    ).squeeze(1)

    # Normalize each frame to [0, 1] so the weight lies in [1, 1 + lambda_h].
    peak = heat.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    heat = (heat / peak).clamp(0.0, 1.0)

    prefix = heat.new_zeros(
        batch_size * n_view,
        latent_frames - future_latent_frames,
        latent_height,
        latent_width,
    )
    heat = torch.cat([prefix, heat], dim=1)
    heat = heat.flatten(1)  # (b*v, latent_frames * latent_height * latent_width)

    if heat.shape != conditioning_mask.shape:
        raise ValueError(
            f"Goal focus/token mask mismatch: heat={heat.shape}, "
            f"conditioning_mask={conditioning_mask.shape}"
        )

    token_loss = elementwise_video_loss.mean(dim=-1)
    future_mask = 1.0 - conditioning_mask.float()
    weights = 1.0 + float(lambda_h) * heat
    weighted = weights * token_loss * future_mask
    denom = (weights * future_mask).sum().clamp_min(1e-8)
    return weighted.sum() / denom


class State:
    # Training state
    seed: int = None
    model_name: str = None
    accelerator: Accelerator = None
    weight_dtype: torch.dtype = None
    train_epochs: int = None
    train_steps: int = None
    overwrote_max_train_steps: bool = False
    num_trainable_parameters: int = 0
    learning_rate: float = None
    train_batch_size: int = None
    generator: torch.Generator = None

    # Hub state
    repo_id: str = None
    # Artifacts state
    output_dir: str = None



class Trainer:

    def __init__(self, config_file, to_log=True, output_dir=None) -> None:
        
        cd = load(open(config_file, "r"), Loader=Loader)
        args = argparse.Namespace(**cd)
        args.lr = float(args.lr)
        args.epsilon = float(args.epsilon)
        args.weight_decay = float(args.weight_decay)

        self.args = args

        if output_dir is not None:
            self.args.output_dir = output_dir

        if self.args.load_weights == False:
            print('You are not loading the pretrained weights, please check the code.')
        self.state = State()

        self.tokenizer = None
        self.text_encoder = None
        self.diffusion_model = None
        self.unet = None
        self.vae = None
        self.scheduler = None

        self._init_distributed()
        self._init_logging()
        self._init_directories_and_repositories()

        self.state.model_name = self.args.model_name

        current_time = datetime.now()
        start_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        if self.state.accelerator.is_main_process:

            self.save_folder = os.path.join(self.args.output_dir, start_time)
            if getattr(self.args, "sub_folder", False):
                self.save_folder = os.path.join(self.args.output_dir, self.args.sub_folder)
            os.makedirs(self.save_folder, exist_ok=True)

            args_dict = vars(deepcopy(self.args))
            for k, v in args_dict.items():
                args_dict[k] = str(v)
            with open(os.path.join(self.save_folder, 'config.json'), "w") as file:
                json.dump(args_dict, file, indent=4, sort_keys=False)
            
            if to_log:
                self.writer = SummaryWriter(log_dir=self.save_folder)
            else:
                self.writer = None

            save_folder_bytes = self.save_folder.encode()
            folder_len_tensor = torch.tensor([len(save_folder_bytes)], device=self.state.accelerator.device)
            dist.broadcast(folder_len_tensor, src=0)
            folder_tensor = torch.ByteTensor(list(save_folder_bytes)).to(self.state.accelerator.device)
            dist.broadcast(folder_tensor, src=0)
        else:
            folder_len_tensor = torch.tensor([0], device=self.state.accelerator.device)
            dist.broadcast(folder_len_tensor, src=0)
            folder_tensor = torch.empty(folder_len_tensor.item(), dtype=torch.uint8, device=self.state.accelerator.device)
            dist.broadcast(folder_tensor, src=0)
            self.save_folder = bytes(folder_tensor.tolist()).decode()

        init_logging(self.save_folder, rank=self.state.accelerator.process_index)

    def _prepare_value_pose_condition(self, pose_history, device, dtype):
        """Build value-pose inputs without leaking absolute episode progress."""
        pose_state = pose_history.to(device=device, dtype=dtype).contiguous()
        mode = getattr(self.args, "value_pose_time_mode", "dataset")
        if mode == "dataset":
            pose_positions = pose_state[..., -1].to(dtype=torch.float32)
        elif mode == "relative_order":
            n_pose = pose_state.shape[1]
            relative = torch.arange(
                1 - n_pose, 1, device=device, dtype=torch.float32
            ).unsqueeze(0).expand(pose_state.shape[0], -1)
            pose_state = pose_state.clone()
            pose_state[..., -1] = (
                relative / max(n_pose - 1, 1)
            ).to(dtype=pose_state.dtype)
            pose_positions = relative
        else:
            raise ValueError(
                f"Unsupported value_pose_time_mode={mode!r}; "
                "expected 'dataset' or 'relative_order'"
            )
        return pose_state, pose_positions.contiguous()


    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to

        if getattr(self.args, "use_deepspeed", False):
            per_device_bs = self.args.batch_size
            world_size = int(os.environ.get("WORLD_SIZE", 1))  # or self.args.world_size
            grad_accum = self.args.gradient_accumulation_steps

            train_batch_size = per_device_bs * world_size * grad_accum
            self.args.deepspeed["train_batch_size"] = train_batch_size
            ds_plugin = DeepSpeedPlugin(
                hf_ds_config=self.args.deepspeed,
                gradient_accumulation_steps=grad_accum
            )
        else:
            ds_plugin = None

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
            deepspeed_plugin=ds_plugin,
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.state.accelerator = accelerator

        if self.args.seed is not None:
            self.state.seed = self.args.seed
            set_seed(self.args.seed)

        weight_dtype = torch.float32
        if self.state.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.state.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            
        self.state.weight_dtype = weight_dtype


    def _init_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.state.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized Trainer")
        logger.info(self.state.accelerator.state, main_process_only=False)
        

    def _init_directories_and_repositories(self):
        if self.state.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            self.state.output_dir = self.args.output_dir


    def prepare_dataset(self) -> None:

        logger.info(f"Training Dataset: {self.args.train_data_class}")
        local_rank = int(os.environ["LOCAL_RANK"])

        train_dataset_class = import_custom_class(
            self.args.train_data_class, self.args.train_data_class_path
        )
        self.train_dataset = train_dataset_class(**self.args.data['train'])

        if hasattr(self.train_dataset, "make_batch_sampler"):
            batch_sampler = self.train_dataset.make_batch_sampler(
                self.args.batch_size, shuffle=True, drop_last=True
            )
            self.train_dataloader = torch.utils.data.DataLoader(
                dataset=self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.args.dataloader_num_workers,
                multiprocessing_context=None,
            )
        else:
            self.train_dataloader = torch.utils.data.DataLoader(
                dataset=self.train_dataset,
                shuffle=True,
                batch_size=self.args.batch_size,
                num_workers=self.args.dataloader_num_workers,
                multiprocessing_context=None,
            )
        logger.info(f">>>>>>>>>>>>>Total Train Eps: {len(self.train_dataset)}<<<<<<<<<<<<<<<<<<\n")


        if 'val' in self.args.data:
            self.prepare_val_dataset()


    def prepare_val_dataset(self) -> None:
        if not hasattr(self.args, "val_data_class"):
            self.args.val_data_class = self.args.train_data_class
        logger.info(f"Validation Dataset: {self.args.val_data_class}")

        val_dataset_class = import_custom_class(
            self.args.val_data_class, self.args.val_data_class_path
        )
        self.val_dataset = val_dataset_class(**self.args.data['val'])

        self.val_index = []
        for _ in range(self.args.batch_size):
            self.val_index.append(random.randint(0, len(self.val_dataset)-1))
        if self.state.accelerator.is_main_process:
            with open(os.path.join(self.save_folder, 'idx.txt'), "w") as file:
                file.write(", ".join(map(str, self.val_index)))

        subset = torch.utils.data.Subset(self.val_dataset, self.val_index)
        self.val_dataloader = torch.utils.data.DataLoader(
            subset, batch_size=self.args.batch_size, shuffle=getattr(self.args, "val_shuffle", False)
        )
        logger.info(f">>>>>>>>>>>>>Total Validatoin Eps: {len(self.val_dataset)}<<<<<<<<<<<<<<<<<<\n")


    def prepare_models(self):

        logger.info("Initializing models")
        device = self.state.accelerator.device
        dtype = self.state.weight_dtype

        ### Load Tokenizer
        tokenizer_class = import_custom_class(
            self.args.tokenizer_class, getattr(self.args, "tokenizer_class_path", "transformers")
        )
        textenc_class = import_custom_class(
            self.args.textenc_class, getattr(self.args, "textenc_class_path", "transformers")
        )
        cond_models = load_condition_models(
            tokenizer_class, textenc_class,
            self.args.pretrained_model_name_or_path if not hasattr(self.args, "tokenizer_pretrained_model_name_or_path") else self.args.tokenizer_pretrained_model_name_or_path,
            load_weights=True
        )
        self.tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
        self.text_encoder = text_encoder.to(device, dtype=dtype).eval()
        self.text_uncond = get_text_conditions(self.tokenizer, self.text_encoder, prompt="")
        self.uncond_prompt_embeds = self.text_uncond['prompt_embeds']
        self.uncond_prompt_attention_mask = self.text_uncond['prompt_attention_mask']

        ### Load VAE
        vae_class = import_custom_class(
            self.args.vae_class, getattr(self.args, "vae_class_path", "transformers")
        )
        if getattr(self.args, 'vae_path', False):
            self.vae = load_vae_models(vae_class, self.args.vae_path).to(device, dtype=dtype).eval()
        else:
            self.vae = load_latent_models(vae_class, self.args.pretrained_model_name_or_path)["vae"].to(device, dtype=dtype).eval()
        if isinstance(self.vae.latents_mean, List):
            self.vae.latents_mean = torch.FloatTensor(self.vae.latents_mean)
        if isinstance(self.vae.latents_std, List):
            self.vae.latents_std = torch.FloatTensor(self.vae.latents_std)
        if self.vae is not None:
            if self.args.enable_slicing:
                self.vae.enable_slicing()
            if self.args.enable_tiling:
                self.vae.enable_tiling()
        self.SPATIAL_DOWN_RATIO = self.vae.spatial_compression_ratio #32
        self.TEMPORAL_DOWN_RATIO = self.vae.temporal_compression_ratio #8
        logger.info(f'SPATIAL_DOWN_RATIO of VAE :{self.SPATIAL_DOWN_RATIO}')
        logger.info(f'TEMPORAL_DOWN_RATIO of VAE :{self.TEMPORAL_DOWN_RATIO}')


        ### Load Diffusion Model
        diffusion_model_class = import_custom_class(
            self.args.diffusion_model_class, getattr(self.args, "diffusion_model_class_path", "transformers")
        )
        self.diffusion_model = load_diffusion_model(
            model_cls=diffusion_model_class,
            model_dir=self.args.diffusion_model['model_path'],
            load_weights=self.args.load_weights and getattr(self.args, "load_diffusion_model_weights", True),
            **self.args.diffusion_model['config']
        ).to(device, dtype=dtype)
        total_params = count_model_parameters(self.diffusion_model)
        logger.info(f'Total parameters for transformer model:{total_params}')


        ### Load Diffuser Scheduler
        diffusion_scheduler_class = import_custom_class(
            self.args.diffusion_scheduler_class, getattr(self.args, "diffusion_scheduler_class_path", "diffusers")
        )
        if hasattr(self.args, "diffusion_scheduler_args"):
            self.scheduler = diffusion_scheduler_class(**self.args.diffusion_scheduler_args)
        else:
            self.scheduler = diffusion_scheduler_class()

        ### Import Inference Pipeline Class
        self.pipeline_class = import_custom_class(
            self.args.pipeline_class, getattr(self.args, "pipeline_class_path", "diffusers")
        )


    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")
        
        components_to_disable_grads = []
            
        for component in components_to_disable_grads:
            if component is not None:
                component.requires_grad_(False)

        if torch.backends.mps.is_available() and self.state.weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        if self.args.gradient_checkpointing:
            self.diffusion_model.enable_gradient_checkpointing()

        # Enable TF32 for faster training on Ampere GPUs: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True


    def prepare_optimizer(self):
        logger.info("Initializing optimizer and lr scheduler")

        train_mode = self.args.train_mode

        self.state.train_epochs = self.args.train_epochs
        self.state.train_steps = self.args.train_steps

        # Make sure the trainable params are in float32
        if self.args.mixed_precision == "fp16":
            cast_training_params([self.diffusion_model], dtype=torch.float32)

        self.state.learning_rate = self.args.lr
        if self.args.scale_lr:
            self.state.learning_rate = (
                self.state.learning_rate
                * self.args.gradient_accumulation_steps
                * self.args.batch_size
                * self.state.accelerator.num_processes
            )

        diffusion_model_trainable_params = []
        if train_mode == 'action_only':
            for name, param in self.diffusion_model.named_parameters():
                if 'action_' in name:
                    param.requires_grad = True
                    diffusion_model_trainable_params.append(param)
                else:
                    param.requires_grad = False
        elif train_mode == "video_only":
            for name, param in self.diffusion_model.named_parameters():
                if 'action_' not in name and 'value_' not in name and 'grounding_' not in name:
                    param.requires_grad = True
                    diffusion_model_trainable_params.append(param)
                else:
                    param.requires_grad = False
        elif train_mode == "video_grounding":
            for name, param in self.diffusion_model.named_parameters():
                if 'action_' not in name and 'value_' not in name:
                    param.requires_grad = True
                    diffusion_model_trainable_params.append(param)
                else:
                    param.requires_grad = False
        elif train_mode == "value_only":
            for name, param in self.diffusion_model.named_parameters():
                if 'value_' in name:
                    param.requires_grad = True
                    diffusion_model_trainable_params.append(param)
                else:
                    param.requires_grad = False
        elif train_mode == "value_action":
            for name, param in self.diffusion_model.named_parameters():
                if 'action_' in name or 'value_' in name:
                    param.requires_grad = True
                    diffusion_model_trainable_params.append(param)
                else:
                    param.requires_grad = False
        elif train_mode == "grounding_only":
            for name, param in self.diffusion_model.named_parameters():
                if 'grounding_' in name:
                    param.requires_grad = True
                    diffusion_model_trainable_params.append(param)
                else:
                    param.requires_grad = False
        elif train_mode == "grounding_action":
            for name, param in self.diffusion_model.named_parameters():
                if 'action_' in name or 'grounding_' in name:
                    param.requires_grad = True
                    diffusion_model_trainable_params.append(param)
                else:
                    param.requires_grad = False
        elif train_mode == "all" or train_mode == 'action_full':
            for name, param in self.diffusion_model.named_parameters():
                param.requires_grad = True
                diffusion_model_trainable_params.append(param)
        else:
            raise NotImplementedError

        # Keep small head updates representable when the frozen backbone runs in
        # BF16. Casting only selected parameters avoids doubling backbone memory.
        if getattr(self.args, "trainable_params_fp32", False):
            for param in diffusion_model_trainable_params:
                param.data = param.data.to(dtype=torch.float32)

        num_trainable_params = sum(p.numel() for p in diffusion_model_trainable_params)
        logger.info(f'Total trainable parameters: {num_trainable_params}')

        goal_module_names = (
            "goal_query_adapter.",
            "goal_heatmap_head.",
            "goal_latent_heatmap_head.",
        )
        goal_param_ids = {
            id(param)
            for name, param in self.diffusion_model.named_parameters()
            if param.requires_grad and any(module_name in name for module_name in goal_module_names)
        }
        goal_params = [
            param for param in diffusion_model_trainable_params if id(param) in goal_param_ids
        ]
        backbone_params = [
            param for param in diffusion_model_trainable_params if id(param) not in goal_param_ids
        ]
        backbone_lr = float(
            getattr(self.args, "video_backbone_lr", self.state.learning_rate)
        )
        goal_module_lr = float(
            getattr(self.args, "goal_module_lr", self.state.learning_rate)
        )
        params_to_optimize = []
        if backbone_params:
            params_to_optimize.append(
                {"params": backbone_params, "lr": backbone_lr, "name": "backbone"}
            )
        if goal_params:
            params_to_optimize.append(
                {"params": goal_params, "lr": goal_module_lr, "name": "goal"}
            )
        logger.info(
            "Optimizer groups: backbone=%d params lr=%g; goal=%d params lr=%g",
            sum(param.numel() for param in backbone_params),
            backbone_lr,
            sum(param.numel() for param in goal_params),
            goal_module_lr,
        )
        self.state.num_trainable_parameters = sum(p.numel() for p in diffusion_model_trainable_params)

        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.lr,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_8bit = self.args.optimizer_8bit,
            use_torchao = self.args.optimizer_torchao,
        )

        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        max_steps_from_epochs = self.state.train_epochs * num_update_steps_per_epoch

        # total training step is decided by the minimum of train_steps and train_epochs
        self.state.train_steps = min(self.args.train_steps, max_steps_from_epochs)
        self.state.overwrote_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.state.accelerator.num_processes,
            num_training_steps=self.state.train_steps * self.state.accelerator.num_processes,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        

    def prepare_for_training(self):
        self.diffusion_model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.state.accelerator.prepare(
            self.diffusion_model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )


    def prepare_trackers(self):
        logger.info("Initializing trackers")
        tracker_name = self.args.tracker_name or "model_train"
        self.state.accelerator.init_trackers(tracker_name, config=self.args.__dict__)


    def train(self):
        logger.info("Starting training")
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.train_batch_size = (
            self.args.batch_size * self.state.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.train_dataset),
            "train epochs": self.state.train_epochs,
            "train steps": self.state.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.train_dataloader),
            "train batch size": self.state.train_batch_size,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")
        
        global_step = 0
        first_epoch = 0
        initial_global_step = 0
        progress_bar = tqdm(
            range(0, self.state.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.state.accelerator.is_local_main_process,
        )

        accelerator = self.state.accelerator
        weight_dtype = self.state.weight_dtype
        scheduler_sigmas = self.scheduler.sigmas.clone().to(device=accelerator.device, dtype=weight_dtype)
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        if getattr(self.args, "eval_only", False):
            n_eval = int(getattr(self.args, "eval_samples", 1))
            eval_repeats = max(1, int(getattr(self.args, "eval_repeats", 1)))
            eval_n_chunks = max(1, int(getattr(self.args, "eval_n_chunks", 1)))
            configured_noise_seeds = getattr(self.args, "eval_noise_seeds", None)
            if configured_noise_seeds:
                noise_seeds = [
                    int(x) for x in str(configured_noise_seeds).split(",") if x.strip()
                ]
                if len(noise_seeds) < eval_repeats:
                    raise ValueError(
                        f"eval_noise_seeds provides {len(noise_seeds)} seeds, "
                        f"but eval_repeats={eval_repeats}"
                    )
            else:
                noise_seeds = [42 + 1009 * repeat for repeat in range(eval_repeats)]
            self.diffusion_model.eval()
            _peek = next(iter(self.val_dataloader))
            n_view = int(_peek["video"].shape[2])
            _cfg_idx = getattr(self.args, "eval_episodes", None)
            if _cfg_idx:
                # treat each entry as a GLOBAL episode number; map to val-subset index
                _want = [int(x) for x in str(_cfg_idx).split(",") if x.strip() != ""]
                _g2i = {}
                for _ii in range(len(self.val_dataset.dataset)):
                    _pp = str(self.val_dataset.dataset[_ii][2])
                    try:
                        _gid = int(os.path.basename(_pp).split("_")[1].split(".")[0])
                    except Exception:
                        continue
                    if _gid not in _g2i:
                        _g2i[_gid] = _ii
                _eval_global_ids = [x for x in _want if x in _g2i]
                _idx = [_g2i[x] for x in _eval_global_ids]
                if n_eval > 0:
                    _idx = _idx[:n_eval]
                    _eval_global_ids = _eval_global_ids[:n_eval]
                logger.info(f"eval_only GLOBAL episodes requested: {_want} -> resolved idx {_idx}")
            else:
                _rng = random.Random(42)
                _all = list(range(len(self.val_dataset)))
                _idx = sorted(_rng.sample(_all, min(n_eval, len(_all))))
                _eval_global_ids = _idx
            logger.info(f"eval_only episodes: {_idx}")
            self.rng_saved = _idx
            for _k in range(len(_idx)):
                # Reset the data RNG before every repeat so all diffusion seeds see
                # exactly the same episode and temporal observation window.
                _sample_seed = int(getattr(self.args, "seed", 42) or 42) + int(_eval_global_ids[_k])
                for _repeat in range(eval_repeats):
                    random.seed(_sample_seed)
                    import numpy as _np
                    _np.random.seed(_sample_seed % (2 ** 32 - 1))
                    torch.manual_seed(_sample_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(_sample_seed)
                    _noise_seed = noise_seeds[_repeat]
                    self.current_eval_episode_id = int(_eval_global_ids[_k])
                    self.current_eval_sample_seed = _sample_seed
                    self.current_eval_noise_seed = _noise_seed
                    self.current_eval_repeat = _repeat
                    self.val_index = [_idx[_k]]
                    self.val_dataloader = torch.utils.data.DataLoader(
                        torch.utils.data.Subset(self.val_dataset, self.val_index),
                        batch_size=1, shuffle=False)
                    model_save_dir = os.path.join(
                        self.save_folder,
                        f"Validation_eval_only_ep{_eval_global_ids[_k]}_idx{_idx[_k]}"
                        f"_pred{_repeat}_seed{_noise_seed}",
                    )
                    logger.info(
                        f"eval_only episode {_k + 1}/{len(_idx)} repeat "
                        f"{_repeat + 1}/{eval_repeats} -> global episode "
                        f"{_eval_global_ids[_k]} val idx {_idx[_k]} "
                        f"sample_seed {_sample_seed} noise_seed {_noise_seed}"
                    )
                    self.validate(
                        accelerator,
                        model_save_dir,
                        _k * eval_repeats + _repeat,
                        n_view=n_view,
                        n_chunk=eval_n_chunks,
                    )
                    torch.cuda.empty_cache()
            accelerator.wait_for_everyone()
            logger.info(
                "eval_only finished ({} episodes x {} predictions x {} chunks)".format(
                    len(_idx), eval_repeats, eval_n_chunks
                )
            )
            return

        # loss spikes
        anomalies = []

        for epoch in range(first_epoch, self.state.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.state.train_epochs})")

            self.diffusion_model.train()

            running_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}
                with accelerator.accumulate([ self.diffusion_model ]):
                    
                    video = batch['video']

                    # shape: {b, c, v, t, h, w}; ranging from -1 to 1
                    video = video.to(accelerator.device, dtype=weight_dtype).contiguous()
                    batch_size, c, n_view, _, h, w = video.shape
                    video = rearrange(video, 'b c v t h w -> (b v) c t h w')

                    # here we use color jitter to the video, with different views or different batches different jitter
                    if self.args.use_color_jitter:
                        video = apply_color_jitter_to_video(video)

                    mem_size = self.args.data['train']['n_previous']

                    # Prepend episode frame-0 image to mem window (visual anchor)
                    if getattr(self.args, "use_start_video", False) and batch.get('start_video') is not None:
                        start_video = batch['start_video'].to(accelerator.device, dtype=weight_dtype).contiguous()
                        start_video = rearrange(start_video, 'b c v t h w -> (b v) c t h w')  # [bv, c, 1, h, w]
                        mem = torch.cat([start_video, video[:,:,:mem_size]], dim=2)  # [bv, c, mem_size+1, h, w]
                        mem_size_total = mem_size + 1
                    else:
                        mem = video[:,:,:mem_size]
                        mem_size_total = mem_size

                    future_video = video[:,:,self.args.data['train']['n_previous']:]

                    if self.args.return_action:
                        future_video = future_video[:,:,:1].repeat(1,1,self.args.data['train']['chunk'],1,1)

                    # get the shape params
                    _, _, raw_frames, raw_height, raw_width = future_video.shape
                    latent_frames = raw_frames // self.TEMPORAL_DOWN_RATIO + 1 + mem_size_total
                    latent_height = raw_height // self.SPATIAL_DOWN_RATIO
                    latent_width = raw_width // self.SPATIAL_DOWN_RATIO

                    dropout_factor = torch.rand(batch_size).to(accelerator.device, dtype=weight_dtype)
                    if getattr(self.args, "paired_shared_noise", False) and "pair_id" in batch:
                        dropout_factor[1::2] = dropout_factor[::2]
                    dropout_mask_prompt = dropout_factor < self.args.caption_dropout_p
                    dropout_mask_prompt = dropout_mask_prompt.unsqueeze(1).unsqueeze(2)

                    mem_latents, future_video_latents = get_latents(
                        self.vae, mem, future_video
                    )

                    mem_latents = rearrange(mem_latents, '(b v m) (h w) c -> (b v) c m h w', b=batch_size, m=mem_size_total, h=latent_height)
                    future_video_latents = rearrange(future_video_latents, '(b v) (f h w) c -> (b v) c f h w',b=batch_size,h=latent_height,w=latent_width)
                    latents = torch.cat((mem_latents, future_video_latents), dim=2)

                    # Construct frame_ids for video-branch rope: real episode frame numbers
                    # for each latent token (mem + future), so rope encodes true elapsed time.
                    # batch['frame_ids']: [b, len(vid_indexes)] where vid_indexes = mem_indexes + video_end[stride-1::stride]
                    # After VAE: mem is 1:1 (m frames -> m latents), future is compressed (raw_frames -> f_latent).
                    # Use the first mem_size entries for mem latents, and subsample/map the rest for future latents.
                    # Gated by use_video_frame_ids_rope (default False) so existing configs that don't
                    # opt in keep the original sequential-position rope behavior unchanged.
                    if getattr(self.args, "use_video_frame_ids_rope", False) and 'frame_ids' in batch:
                        raw_frame_ids = batch['frame_ids'].to(accelerator.device, dtype=torch.float32)  # [b, mem_size + len(video_end_subsampled)]
                        # mem portion: first mem_size entries are 1:1
                        mem_frame_ids = raw_frame_ids[:, :mem_size_total]  # [b, mem_size_total]
                        # future portion: causal VAE groups frames as [frame0_alone, frames[1:9], frames[9:17], ...]
                        # For simplicity, map future latent i to the "representative" raw frame index from raw_frame_ids.
                        # Since raw_frame_ids[mem_size:] already has stride-subsampled indices, reuse them if count matches,
                        # otherwise construct via arithmetic on the first future-chunk frame index.
                        future_raw_ids = raw_frame_ids[:, mem_size_total:]  # [b, len(video_end_subsampled)]
                        f_latent = future_video_latents.shape[2]  # actual latent count after VAE
                        if future_raw_ids.shape[1] == f_latent:
                            # Already matches (stride == temporal_down_ratio, or close enough)
                            future_frame_ids = future_raw_ids
                        else:
                            # Need to construct: latent i -> frame (first_future_frame + i * temporal_down_ratio)
                            # first_future_frame = raw_frame_ids[:, mem_size:mem_size+1] if available, else fallback to mem_size
                            if future_raw_ids.shape[1] > 0:
                                first_future = future_raw_ids[:, :1]  # [b, 1]
                            else:
                                first_future = mem_frame_ids[:, -1:] + 1  # fallback: next frame after last mem
                            offsets = torch.arange(f_latent, device=accelerator.device, dtype=torch.float32) * self.TEMPORAL_DOWN_RATIO
                            future_frame_ids = first_future + offsets.unsqueeze(0)  # [b, f_latent]
                        frame_ids = torch.cat([mem_frame_ids, future_frame_ids], dim=1).to(accelerator.device, dtype=torch.float32)  # [b, latent_frames]
                        # Expand for n_view: (b, latent_frames) -> (b*n_view, latent_frames)
                        frame_ids = frame_ids.repeat_interleave(n_view, dim=0).contiguous()
                    else:
                        frame_ids = None

                    video_attention_mask = None
                    latents = rearrange(latents, 'bv c f h w -> bv (f h w) c')

                    captions = batch['caption']
                    text_conds = get_text_conditions(self.tokenizer,self.text_encoder,captions)
                    prompt_embeds = text_conds['prompt_embeds']
                    prompt_attention_mask = text_conds['prompt_attention_mask']
                    prompt_embeds = self.uncond_prompt_embeds.repeat(batch_size,1,1)*dropout_mask_prompt + \
                                    prompt_embeds*~dropout_mask_prompt

                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    action_weights = compute_density_for_timestep_sampling(
                        weighting_scheme=self.args.flow_weighting_scheme,
                        batch_size=batch_size,
                        logit_mean=self.args.flow_logit_mean,
                        logit_std=self.args.flow_logit_std,
                        mode_scale=self.args.flow_mode_scale,
                    )
                    if getattr(self.args, "paired_shared_noise", False) and "pair_id" in batch:
                        action_weights[1::2] = action_weights[::2]
                    # 0-1, 0 -> most noisy, 1 -> almost clean
                    action_indices = (action_weights * self.scheduler.config.num_train_timesteps).long()
                    action_sigmas = scheduler_sigmas[action_indices]
                    action_timesteps = (action_sigmas * 1000.0).long()

                    if self.args.return_action and self.args.noisy_video:
                        weights = torch.full_like(action_weights, 0.0).unsqueeze(1).repeat(1,n_view)
                    else:
                        weights = action_weights.unsqueeze(1).repeat(1,n_view)

                    weights = rearrange(weights, 'b v -> (b v)')
                    indices = (weights * self.scheduler.config.num_train_timesteps).long()
                    sigmas = scheduler_sigmas[indices]
                    timesteps = (sigmas * 1000.0).long()

                    if self.args.return_action:
                        if getattr(self.args, "add_state", False):
                            # NOTE add states from the batch:
                            act_state = batch['state']
                            if act_state.shape[1] != 1:
                                act_state = act_state[:, mem_size-1:mem_size]
                            act_state = act_state.to(accelerator.device, dtype=weight_dtype).contiguous()
                        else:
                            act_state = None


                        actions = batch['actions'][:, -self.args.data['train']['action_chunk']:].to(accelerator.device, dtype=weight_dtype).contiguous()   # shape b,t,c
                        action_dim = actions.shape[-1]

                        action_discrete = getattr(self.args, "action_discrete", False)
                        noise_actions = randn_tensor(actions.shape, device=accelerator.device, dtype=weight_dtype)

                        # here we get action_timesteps, shape (b,) originally, target shape (b, l)
                        action_timesteps = action_timesteps.unsqueeze(-1).repeat(1, actions.shape[1])
                        action_ss= action_sigmas.reshape(-1, 1, 1).repeat(1, 1, actions.shape[-1])

                        noisy_actions = (1.0 - action_ss) * actions + action_ss * noise_actions

                        action_weights = compute_loss_weighting_for_sd3(
                            weighting_scheme=self.args.flow_weighting_scheme, sigmas=action_sigmas
                        ).reshape(-1, 1, 1).repeat(1, 1, actions.size(-1))

                        # If action_in_channels > actions.shape[-1], pad to match (e.g., state concatenation)
                        target_action_dim = self.args.diffusion_model['config'].get('action_in_channels', actions.shape[-1])
                        if actions.shape[-1] < target_action_dim:
                            pad = torch.zeros(actions.shape[0], actions.shape[1], target_action_dim - actions.shape[-1],
                                              device=actions.device, dtype=actions.dtype)
                            noisy_actions = torch.cat([noisy_actions, pad], dim=-1)
                        action_normed_target = actions  # keep for continuous loss (if needed in future)
                        # For discrete: load ground-truth class labels
                        if action_discrete:
                            action_classes = batch['action_classes'][:, -self.args.data['train']['action_chunk']:].to(accelerator.device, dtype=torch.long).contiguous()
                    else:
                        actions = None
                        action_timesteps = None
                        noisy_actions = None
                        act_state = None
                        action_discrete = False

                    if self.args.return_value:
                        value_chunk = self.args.data['train']['action_chunk'] if self.args.data['train'].get("value_dense", False) else self.args.data['train']['chunk']
                        values = batch['value_targets'][:, -value_chunk:].to(accelerator.device, dtype=weight_dtype).contiguous()  # shape b,t,c
                        value_target_type = getattr(self.args, "value_target_type", "scalar")
                        value_direct_regression = getattr(
                            self.args, "value_direct_regression", False
                        )
                        
                        # state_value = values[..., -1]  # shape b,t

                        if value_target_type == "feature":
                            symlog_values = F.normalize(values.float(), dim=-1).to(dtype=weight_dtype)
                        elif getattr(self.args, "disable_value_symlog", False):
                            symlog_values = values  # train in original space
                        else:
                            symlog_values = symlog(values)  # shape b,t,c
                        # value_bins = build_twohot_bins(num_bins = self.args.diffusion_model['config'].get("value_in_channels",21))

                        # values_twohot = torch.stack([
                        #     twohot_encode(state_value_symlog[i], value_bins)
                        #     for i in range(state_value_symlog.size(0))
                        # ])  # shape b,t,num_bins

                        if value_direct_regression:
                            # A scalar evaluator is a deterministic query, not a
                            # generative variable. A learned value token removes
                            # value-noise seed bias and target leakage at low sigma.
                            noisy_values = torch.zeros_like(symlog_values)
                            value_timesteps = torch.zeros(
                                symlog_values.shape[:2],
                                device=accelerator.device,
                                dtype=torch.long,
                            )
                            noise_values = None
                            value_ss = None
                            value_weights = None
                        else:
                            value_weights = compute_density_for_timestep_sampling(
                                weighting_scheme=self.args.flow_weighting_scheme,
                                batch_size=batch_size,
                                logit_mean=self.args.flow_logit_mean,
                                logit_std=self.args.flow_logit_std,
                                mode_scale=self.args.flow_mode_scale,
                            )
                            value_indices = (value_weights * self.scheduler.config.num_train_timesteps).long()
                            value_sigmas = scheduler_sigmas[value_indices]
                            value_timesteps = (value_sigmas * 1000.0).long()
                            value_timesteps = value_timesteps.unsqueeze(-1).repeat(1, symlog_values.shape[1])

                            noise_values = randn_tensor(symlog_values.shape, device=accelerator.device, dtype=weight_dtype)
                            value_ss = value_sigmas.reshape(-1, 1, 1).repeat(1, 1, symlog_values.shape[-1])
                            
                            # Legacy joint training forced value inputs to pure noise while
                            # retaining a randomly sampled value timestep. Keep that behavior
                            # available for old configs, but allow standard flow-matching pairs.
                            if self.args.return_action and getattr(
                                self.args, "value_force_full_noise_with_action", True
                            ):
                                value_ss = torch.full_like(value_ss, 1.0)

                            noisy_values = (1.0 - value_ss) * symlog_values + value_ss * noise_values

                            value_weights = compute_loss_weighting_for_sd3(
                                weighting_scheme=self.args.flow_weighting_scheme, sigmas=value_sigmas
                            ).reshape(-1, 1, 1).repeat(1, 1, symlog_values.size(-1))

                        if getattr(self.args, "use_value_pose_condition", False):
                            # batch['pose_history']: [b, 1+n_previous, 15] (14-dim pose + time-delta)
                            value_pose_state, value_pose_positions = self._prepare_value_pose_condition(
                                batch['pose_history'], accelerator.device, weight_dtype
                            )
                        else:
                            value_pose_state = None
                            value_pose_positions = None
                    else:
                        noisy_values = None
                        symlog_values = None
                        value_timesteps = None
                        value_pose_state = None
                        value_pose_positions = None

                    if getattr(self.args, "return_grounding", False):
                        grounding_chunk = self.args.data['train']['action_chunk'] if self.args.data['train'].get("grounding_dense", True) else self.args.data['train']['chunk']
                        grounding_targets = batch['grounding_targets'][:, -grounding_chunk:].to(accelerator.device, dtype=weight_dtype).contiguous()
                        grounding_targets = F.normalize(grounding_targets.float(), dim=-1).to(dtype=weight_dtype)
                        if 'grounding_weights' in batch:
                            grounding_confidence = batch['grounding_weights'][:, -grounding_chunk:].to(accelerator.device, dtype=torch.float32).contiguous()
                        else:
                            grounding_confidence = torch.ones(
                                grounding_targets.shape[0],
                                grounding_targets.shape[1],
                                1,
                                device=accelerator.device,
                                dtype=torch.float32,
                            )
                        if grounding_confidence.ndim == 2:
                            grounding_confidence = grounding_confidence.unsqueeze(-1)
                        grounding_confidence = torch.nan_to_num(grounding_confidence.float(), nan=0.0, posinf=1.0, neginf=0.0)
                        grounding_confidence = grounding_confidence.clamp(
                            min=getattr(self.args, "grounding_confidence_floor", 0.05),
                            max=getattr(self.args, "grounding_confidence_ceiling", 1.0),
                        )
                        if getattr(self.args, "grounding_confidence_normalize", True):
                            grounding_confidence = grounding_confidence / grounding_confidence.mean().clamp_min(1e-6)

                        grounding_weights = compute_density_for_timestep_sampling(
                            weighting_scheme=self.args.flow_weighting_scheme,
                            batch_size=batch_size,
                            logit_mean=self.args.flow_logit_mean,
                            logit_std=self.args.flow_logit_std,
                            mode_scale=self.args.flow_mode_scale,
                        )
                        grounding_indices = (grounding_weights * self.scheduler.config.num_train_timesteps).long()
                        grounding_sigmas = scheduler_sigmas[grounding_indices]
                        grounding_timesteps = (grounding_sigmas * 1000.0).long()
                        grounding_timesteps = grounding_timesteps.unsqueeze(-1).repeat(1, grounding_targets.shape[1])

                        noise_grounding = randn_tensor(grounding_targets.shape, device=accelerator.device, dtype=weight_dtype)
                        grounding_ss = grounding_sigmas.reshape(-1, 1, 1).repeat(1, 1, grounding_targets.shape[-1])
                        noisy_grounding = (1.0 - grounding_ss) * grounding_targets + grounding_ss * noise_grounding

                        grounding_loss_weights = compute_loss_weighting_for_sd3(
                            weighting_scheme=self.args.flow_weighting_scheme, sigmas=grounding_sigmas
                        ).reshape(-1, 1, 1).repeat(1, 1, grounding_targets.size(-1))

                        if getattr(self.args, "use_grounding_pose_condition", False):
                            grounding_pose_state = batch['pose_history'].to(accelerator.device, dtype=weight_dtype).contiguous()
                            grounding_pose_positions = batch['pose_history'][..., -1].to(accelerator.device, dtype=torch.float32).contiguous()
                        else:
                            grounding_pose_state = None
                            grounding_pose_positions = None
                    else:
                        noisy_grounding = None
                        grounding_targets = None
                        grounding_confidence = None
                        grounding_timesteps = None
                        grounding_pose_state = None
                        grounding_pose_positions = None

                    # shape:  bv, l, c and bv, l
                    noise, conditioning_mask, cond_indicator = gen_noise_from_condition_frame_latent(
                        mem_latents, latent_frames, latent_height, latent_width, noise_to_condition_frames=self.args.noise_to_first_frame
                    )  # set initial frames noise to 0
                    if getattr(self.args, "paired_shared_noise", False) and "pair_id" in batch:
                        paired_noise = rearrange(noise, "(b v) l c -> b v l c", b=batch_size, v=n_view)
                        paired_noise[1::2] = paired_noise[::2].clone()
                        noise = rearrange(paired_noise, "b v l c -> (b v) l c")
                    if self.args.pixel_wise_timestep:
                        # shape: bv, thw
                        timesteps = timesteps.unsqueeze(-1) * (1 - conditioning_mask)
                    else:
                        # shape: bv, t
                        timesteps = timesteps.unsqueeze(-1) * (1 - cond_indicator)

                    # shape: bv,1,c
                    ss = sigmas.reshape(-1, 1, 1).repeat(1, 1, latents.size(-1))
                    force_noisy_video = (
                        self.args.noisy_video
                        and (self.args.return_action or self.args.return_value or getattr(self.args, "return_grounding", False))
                        and self.args.train_mode != "video_grounding"
                    )
                    if force_noisy_video:
                        ss = torch.full_like(ss, 1.0)

                    noisy_latents = (1.0 - ss) * latents + ss * noise

                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss, shape bv,1,c
                    weights = compute_loss_weighting_for_sd3(
                        weighting_scheme=self.args.flow_weighting_scheme, sigmas=sigmas
                    ).reshape(-1, 1, 1).repeat(1, 1, latents.size(-1))

                    pred_all = forward_pass(
                        model=self.diffusion_model,
                        timesteps=timesteps,
                        noisy_latents=noisy_latents,
                        prompt_embeds=prompt_embeds,
                        prompt_attention_mask=prompt_attention_mask,
                        num_frames=latent_frames,
                        height=latent_height,
                        width=latent_width,
                        n_view=n_view,
                        action_states=noisy_actions,
                        action_timestep=action_timesteps,
                        value_states=noisy_values,
                        value_timestep=value_timesteps,
                        grounding_states=noisy_grounding,
                        grounding_timestep=grounding_timesteps,
                        return_video=self.args.return_video or self.args.return_action or self.args.return_value or getattr(self.args, "return_grounding", False),
                        return_action=self.args.return_action,
                        return_value=self.args.return_value,
                        return_grounding=getattr(self.args, "return_grounding", False),
                        video_attention_mask=video_attention_mask,
                        history_action_state=act_state,
                        history_value_state=None,
                        value_pose_state=value_pose_state,
                        value_pose_positions=value_pose_positions,
                        history_grounding_state=None,
                        grounding_pose_state=grounding_pose_state,
                        grounding_pose_positions=grounding_pose_positions,
                        frame_ids=frame_ids,
                        condition_mask=conditioning_mask,
                    )['latents']

                    loss_goal_video_focus = None
                    if self.args.train_mode == 'all' or self.args.train_mode == 'video_only' or self.args.train_mode == 'video_grounding':
                        pred = pred_all['video']
                        target = noise - latents
                        elementwise_video_loss = weights.float() * (pred.float() - target.float()).pow(2)
                        loss_video = elementwise_video_loss * (
                            1 - conditioning_mask.unsqueeze(-1).repeat(
                                1, 1, elementwise_video_loss.size(-1)
                            )
                        )
                        # Average loss across channel dimension
                        loss_video = loss_video.mean(list(range(1, loss_video.ndim)))
                        # Average loss across batch dimension
                        loss_video = loss_video.mean()

                        if getattr(self.args, "use_goal_focused_video_loss", False):
                            required_goal_keys = ("goal_heatmaps", "goal_vis", "goal_conf")
                            missing_goal_keys = [key for key in required_goal_keys if key not in batch]
                            if missing_goal_keys:
                                raise KeyError(
                                    "Goal-focused video loss is enabled but the batch is missing: "
                                    + ", ".join(missing_goal_keys)
                                )
                            loss_goal_video_focus = goal_focused_video_loss(
                                elementwise_video_loss=elementwise_video_loss,
                                targets=batch["goal_heatmaps"].to(
                                    device=accelerator.device, dtype=torch.float32
                                ),
                                visible=batch["goal_vis"].to(
                                    device=accelerator.device, dtype=torch.float32
                                ),
                                conditioning_mask=conditioning_mask,
                                batch_size=batch_size,
                                n_view=n_view,
                                latent_frames=latent_frames,
                                latent_height=latent_height,
                                latent_width=latent_width,
                                future_latent_frames=future_video_latents.shape[2],
                                lambda_h=float(
                                    getattr(self.args, "goal_video_focus_loss_scale", 3.0)
                                ),
                            )
                            # The target-weighted average replaces the
                            # unweighted video loss.
                            loss_video = loss_goal_video_focus
                    else:
                        loss_video = 0.

                    if self.args.train_mode == 'all' or self.args.train_mode == 'action_only' or self.args.train_mode == 'action_full' or self.args.train_mode == 'value_action' or self.args.train_mode == 'grounding_action':
                        if action_discrete:
                            # Cross-entropy loss for discrete actions
                            # pred_all['action_class_logits']: (b, t, num_active_dims, num_classes)
                            # action_classes:             (b, t, num_active_dims)
                            logits = pred_all['action_class_logits']  # (b, t, n_dims, n_cls)
                            b, t, n_dims, n_cls = logits.shape
                            target = action_classes                   # (b, t, n_dims)
                            # Optional class weighting for imbalanced action classes
                            action_ce_weight = getattr(self.args, 'action_class_weight', None)
                            if action_ce_weight is not None:
                                import numpy as np
                                w = torch.tensor(action_ce_weight, dtype=logits.dtype, device=logits.device)
                                action_ce = F.cross_entropy(
                                    logits.view(-1, n_cls),
                                    target.view(-1),
                                    weight=w,
                                    reduction='none',
                                ).view(b, t, n_dims)
                            else:
                                action_ce = F.cross_entropy(
                                    logits.view(-1, n_cls),
                                    target.view(-1),
                                    reduction='none',
                                ).view(b, t, n_dims)
                            if "action_loss_mask" in batch:
                                action_mask = batch["action_loss_mask"].to(
                                    accelerator.device, dtype=action_ce.dtype
                                ).reshape(b, 1, 1)
                                loss_action = (action_ce * action_mask).sum() / (
                                    action_mask.sum().clamp_min(1.0) * t * n_dims
                                )
                            else:
                                loss_action = action_ce.mean()
                        else:
                            target_action = noise_actions - actions
                            loss_action = action_weights.float() * (pred_all['action'].float() - target_action.float()).pow(2)    # shape b,l,c
                            loss_action = loss_action.mean()
                    else:
                        loss_action = 0.

                    loss_value_regression = None
                    loss_value_ranking = None
                    if self.args.train_mode in ('all', 'value_only', 'value_action'):
                        if 'value' in pred_all:
                            if value_direct_regression:
                                predicted_value = pred_all['value'].float()
                                loss_value_regression = F.smooth_l1_loss(
                                    predicted_value, symlog_values.float()
                                )
                                if "pair_id" in batch and "pair_role" in batch:
                                    # Rank by the final predicted future token.
                                    rank_scores = predicted_value[:, -1].mean(dim=-1)
                                    loss_value_ranking = paired_value_ranking_loss(
                                        rank_scores,
                                        batch["pair_id"].to(accelerator.device),
                                        batch["pair_role"].to(accelerator.device),
                                    )
                                else:
                                    loss_value_ranking = predicted_value.sum() * 0.0
                                loss_value = loss_value_regression + getattr(
                                    self.args, "value_ranking_loss_scale", 1.0
                                ) * loss_value_ranking
                            else:
                                target_value = noise_values - symlog_values
                                loss_value = value_weights.float() * (pred_all['value'].float() - target_value.float()).pow(2)
                                loss_value = loss_value.mean()
                                if getattr(self.args, "value_target_type", "scalar") == "feature":
                                    pred_clean_value = noise_values.float() - pred_all['value'].float()
                                    pred_clean_value = F.normalize(pred_clean_value, dim=-1)
                                    target_clean_value = F.normalize(symlog_values.float(), dim=-1)
                                    loss_value_align = 1.0 - (pred_clean_value * target_clean_value).sum(dim=-1)
                                    loss_value = loss_value + getattr(self.args, "value_feature_alignment_loss_scale", 1.0) * loss_value_align.mean()
                        else:
                            loss_value = 0.
                    else:
                        loss_value = 0.

                    loss_grounding_mse = None
                    loss_grounding_align = None
                    if self.args.train_mode == 'all' or self.args.train_mode == 'grounding_only' or self.args.train_mode == 'grounding_action' or self.args.train_mode == 'video_grounding':
                        if 'grounding' in pred_all:
                            target_grounding = noise_grounding - grounding_targets
                            loss_grounding = grounding_loss_weights.float() * (pred_all['grounding'].float() - target_grounding.float()).pow(2)
                            if grounding_confidence is not None:
                                loss_grounding = loss_grounding * grounding_confidence
                            loss_grounding_mse = loss_grounding.mean()

                            pred_clean_grounding = noise_grounding.float() - pred_all['grounding'].float()
                            pred_clean_grounding = F.normalize(pred_clean_grounding, dim=-1)
                            target_clean_grounding = F.normalize(grounding_targets.float(), dim=-1)
                            loss_grounding_align = 1.0 - (pred_clean_grounding * target_clean_grounding).sum(dim=-1)
                            if grounding_confidence is not None:
                                loss_grounding_align = loss_grounding_align * grounding_confidence.squeeze(-1)
                            loss_grounding_align = loss_grounding_align.mean()
                            loss_grounding = loss_grounding_mse + getattr(self.args, "grounding_feature_alignment_loss_scale", 1.0) * loss_grounding_align
                        else:
                            loss_grounding = 0.
                    else:
                        loss_grounding = 0.


                    action_loss_scale = getattr(self.args, "action_loss_scale", 1.0)
                    value_loss_scale = getattr(self.args, "value_loss_scale", 1.0)
                    grounding_loss_scale = getattr(self.args, "grounding_loss_scale", 1.0)

                    loss = loss_video + action_loss_scale * loss_action + value_loss_scale * loss_value + grounding_loss_scale * loss_grounding
                    loss_goal_hm = None
                    if getattr(self.args, "use_goal_heatmap", False) and "goal_heatmap" in pred_all and "goal_heatmaps" in batch:
                        nv = batch["video"].shape[2]
                        ghm_pred = rearrange(pred_all["goal_heatmap"].float(), "(b v) t hh ww -> b v t hh ww", v=nv)
                        ghm_tar = batch["goal_heatmaps"].to(device=accelerator.device, dtype=torch.float32).contiguous()
                        goal_conf = batch["goal_conf"].to(device=accelerator.device, dtype=torch.float32)
                        goal_vis = batch["goal_vis"].to(device=accelerator.device, dtype=torch.float32)
                        loss_goal_hm = goal_heatmap_distribution_loss(
                            ghm_pred,
                            ghm_tar,
                            goal_vis,
                            goal_conf,
                        )
                        loss = loss + getattr(self.args, "goal_heatmap_loss_scale", 1.0) * loss_goal_hm

                    loss_goal_query = None
                    if getattr(self.args, "use_goal_query_attention", False):
                        if "goal_query_heatmap" not in pred_all:
                            raise KeyError(
                                "Goal-query supervision is enabled, but the model did not return "
                                "'goal_query_heatmap'. Enable diffusion_model.config.use_goal_query_adapter."
                            )
                        required_goal_keys = ("goal_heatmaps", "goal_vis", "goal_conf")
                        missing_goal_keys = [key for key in required_goal_keys if key not in batch]
                        if missing_goal_keys:
                            raise KeyError(
                                "Goal-query supervision is enabled but the batch is missing: "
                                + ", ".join(missing_goal_keys)
                            )
                        loss_goal_query = goal_query_distribution_loss(
                            pred_all["goal_query_heatmap"].float(),
                            batch["goal_heatmaps"].to(
                                device=accelerator.device, dtype=torch.float32
                            ).contiguous(),
                            batch["goal_vis"].to(
                                device=accelerator.device, dtype=torch.float32
                            ),
                            batch["goal_conf"].to(
                                device=accelerator.device, dtype=torch.float32
                            ),
                        )
                        loss = loss + getattr(
                            self.args, "goal_query_loss_scale", 0.05
                        ) * loss_goal_query

                    loss_goal_latent_consistency = None
                    if getattr(self.args, "use_goal_latent_consistency", False):
                        if "goal_latent_heatmap" not in pred_all:
                            raise KeyError(
                                "Goal latent consistency is enabled, but the model did not return "
                                "'goal_latent_heatmap'. Enable diffusion_model.config."
                                "use_goal_latent_heatmap_head."
                            )
                        required_goal_keys = ("goal_heatmaps", "goal_vis", "goal_conf")
                        missing_goal_keys = [key for key in required_goal_keys if key not in batch]
                        if missing_goal_keys:
                            raise KeyError(
                                "Goal latent consistency is enabled but the batch is missing: "
                                + ", ".join(missing_goal_keys)
                            )
                        nv = batch["video"].shape[2]
                        latent_hm_pred = rearrange(
                            pred_all["goal_latent_heatmap"].float(),
                            "(b v) t hh ww -> b v t hh ww",
                            v=nv,
                        )
                        ghm_tar = batch["goal_heatmaps"].to(
                            device=accelerator.device, dtype=torch.float32
                        ).contiguous()
                        goal_conf = batch["goal_conf"].to(
                            device=accelerator.device, dtype=torch.float32
                        )
                        goal_vis = batch["goal_vis"].to(
                            device=accelerator.device, dtype=torch.float32
                        )
                        loss_goal_latent_consistency = goal_heatmap_distribution_loss(
                            latent_hm_pred,
                            ghm_tar,
                            goal_vis,
                            goal_conf,
                        )
                        loss = loss + getattr(
                            self.args, "goal_latent_consistency_loss_scale", 0.1
                        ) * loss_goal_latent_consistency



                    assert torch.isnan(loss) == False, "NaN loss detected"
                    accelerator.backward(loss)
                    if accelerator.sync_gradients and accelerator.distributed_type != DistributedType.DEEPSPEED:
                        grad_norm = accelerator.clip_grad_norm_(self.diffusion_model.parameters(), self.args.max_grad_norm)
                        logs["grad_norm"] = grad_norm
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                

                loss = accelerator.reduce(loss.detach(), reduction='mean')
                if self.args.train_mode == 'all' or self.args.train_mode == 'action_only' or self.args.train_mode == 'action_full' or self.args.train_mode == 'value_action' or self.args.train_mode == 'grounding_action':
                    loss_action = accelerator.reduce(loss_action.detach(), reduction='mean')
                if self.args.train_mode == 'all' or self.args.train_mode == 'video_only' or self.args.train_mode == 'video_grounding':
                    loss_video = accelerator.reduce(loss_video.detach(), reduction='mean')
                if self.args.train_mode in ('all', 'value_only', 'value_action'):
                    loss_value = accelerator.reduce(loss_value.detach(), reduction='mean')
                    if torch.is_tensor(loss_value_regression):
                        loss_value_regression = accelerator.reduce(
                            loss_value_regression.detach(), reduction='mean'
                        )
                    if torch.is_tensor(loss_value_ranking):
                        loss_value_ranking = accelerator.reduce(
                            loss_value_ranking.detach(), reduction='mean'
                        )
                if self.args.train_mode == 'all' or self.args.train_mode == 'grounding_only' or self.args.train_mode == 'grounding_action' or self.args.train_mode == 'video_grounding':
                    if torch.is_tensor(loss_grounding):
                        loss_grounding = accelerator.reduce(loss_grounding.detach(), reduction='mean')
                    if torch.is_tensor(loss_grounding_mse):
                        loss_grounding_mse = accelerator.reduce(loss_grounding_mse.detach(), reduction='mean')
                    if torch.is_tensor(loss_grounding_align):
                        loss_grounding_align = accelerator.reduce(loss_grounding_align.detach(), reduction='mean')
                if torch.is_tensor(loss_goal_hm):
                    loss_goal_hm = accelerator.reduce(loss_goal_hm.detach(), reduction='mean')
                if torch.is_tensor(loss_goal_query):
                    loss_goal_query = accelerator.reduce(
                        loss_goal_query.detach(), reduction='mean'
                    )
                if torch.is_tensor(loss_goal_video_focus):
                    loss_goal_video_focus = accelerator.reduce(
                        loss_goal_video_focus.detach(), reduction='mean'
                    )
                if torch.is_tensor(loss_goal_latent_consistency):
                    loss_goal_latent_consistency = accelerator.reduce(
                        loss_goal_latent_consistency.detach(), reduction='mean'
                    )

                running_loss += loss.item()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                prev_logs = logs
                current_lrs = self.lr_scheduler.get_last_lr()
                logs = {"loss": loss.detach().item(), "lr": current_lrs[0]}
                if len(current_lrs) > 1:
                    logs["goal_lr"] = current_lrs[-1]
                if "grad_norm" in prev_logs:
                    grad_norm = prev_logs["grad_norm"]
                    logs["grad_norm"] = grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm
                if torch.is_tensor(loss_video):
                    logs["video_loss"] = loss_video.detach().item()
                if torch.is_tensor(loss_action):
                    logs["action_loss"] = loss_action.detach().mean().item()
                if torch.is_tensor(loss_value):
                    logs["value_loss"] = loss_value.detach().item()
                if torch.is_tensor(loss_value_regression):
                    logs["value_regression_loss"] = loss_value_regression.detach().item()
                if torch.is_tensor(loss_value_ranking):
                    logs["value_ranking_loss"] = loss_value_ranking.detach().item()
                if torch.is_tensor(loss_grounding):
                    logs["grounding_loss"] = loss_grounding.detach().item()
                if torch.is_tensor(loss_grounding_mse):
                    logs["grounding_mse"] = loss_grounding_mse.detach().item()
                if torch.is_tensor(loss_grounding_align):
                    logs["grounding_align"] = loss_grounding_align.detach().item()
                if torch.is_tensor(loss_goal_hm):
                    logs["goal_heatmap_loss"] = loss_goal_hm.detach().item()
                if torch.is_tensor(loss_goal_query):
                    logs["goal_query_loss"] = loss_goal_query.detach().item()
                if torch.is_tensor(loss_goal_video_focus):
                    logs["goal_video_focus_loss"] = loss_goal_video_focus.detach().item()
                if torch.is_tensor(loss_goal_latent_consistency):
                    logs["goal_latent_consistency_loss"] = (
                        loss_goal_latent_consistency.detach().item()
                    )
                progress_bar.set_postfix(logs)
                accelerator.log(logs, step=global_step)

                if global_step >= self.state.train_steps:
                    logger.info(">>> max train step reached")
                    break

                if global_step % self.args.steps_to_log == 0:
                    if accelerator.is_main_process:
                        if self.writer is not None:
                            self.writer.add_scalar("Training Loss", loss.item(), global_step)
                            if self.args.train_mode in ('all', 'action_only', 'action_full', 'value_action', 'grounding_action'):
                                self.writer.add_scalar("Action loss", loss_action.mean().item(), global_step)
                            if self.args.train_mode in ('all', 'value_only', 'value_action'):
                                self.writer.add_scalar("Value loss", loss_value.mean().item(), global_step)
                            if self.args.train_mode == 'all' or self.args.train_mode == 'video_only' or self.args.train_mode == 'video_grounding':
                                self.writer.add_scalar("Video loss", loss_video.item(), global_step)
                            if self.args.train_mode == 'all' or self.args.train_mode == 'grounding_only' or self.args.train_mode == 'grounding_action' or self.args.train_mode == 'video_grounding':
                                if torch.is_tensor(loss_grounding):
                                    self.writer.add_scalar("Grounding loss", loss_grounding.item(), global_step)
                                if torch.is_tensor(loss_grounding_mse):
                                    self.writer.add_scalar("Grounding denoise loss", loss_grounding_mse.item(), global_step)
                                if torch.is_tensor(loss_grounding_align):
                                    self.writer.add_scalar("Grounding align loss", loss_grounding_align.item(), global_step)
                            if torch.is_tensor(loss_goal_hm):
                                self.writer.add_scalar("Goal heatmap loss", loss_goal_hm.item(), global_step)
                            if torch.is_tensor(loss_goal_query):
                                self.writer.add_scalar(
                                    "Goal query loss", loss_goal_query.item(), global_step
                                )
                            if torch.is_tensor(loss_goal_video_focus):
                                self.writer.add_scalar(
                                    "Goal-focused video loss",
                                    loss_goal_video_focus.item(),
                                    global_step,
                                )
                            if torch.is_tensor(loss_goal_latent_consistency):
                                self.writer.add_scalar(
                                    "Goal latent consistency loss",
                                    loss_goal_latent_consistency.item(),
                                    global_step,
                                )
                        component_msg = (
                            f"step={global_step} loss={logs['loss']:.6f} "
                            f"video={logs.get('video_loss', 0.0):.6f} "
                            f"action={logs.get('action_loss', 0.0):.6f} "
                            f"value={logs.get('value_loss', 0.0):.6f} "
                            f"value_reg={logs.get('value_regression_loss', 0.0):.6f} "
                            f"value_rank={logs.get('value_ranking_loss', 0.0):.6f} "
                            f"goal_video_focus={logs.get('goal_video_focus_loss', 0.0):.6f} "
                            f"goal_heatmap={logs.get('goal_heatmap_loss', 0.0):.6f} "
                            f"goal_query={logs.get('goal_query_loss', 0.0):.6f} "
                            f"goal_latent_consistency={logs.get('goal_latent_consistency_loss', 0.0):.6f} "
                            f"grounding={logs.get('grounding_loss', 0.0):.6f} "
                            f"grounding_mse={logs.get('grounding_mse', 0.0):.6f} "
                            f"grounding_align={logs.get('grounding_align', 0.0):.6f} "
                            f"lr={logs['lr']:.6g} "
                            f"goal_lr={logs.get('goal_lr', logs['lr']):.6g}"
                        )
                        logger.info(component_msg)

                if global_step % self.args.steps_to_val == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        model_save_dir = os.path.join(self.save_folder,f'Validation_step_{global_step}')
                        self.validate(accelerator, model_save_dir, global_step, n_view=n_view, n_chunk=1)

                
                if global_step % self.args.steps_to_save == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        model_to_save = unwrap_model(accelerator, self.diffusion_model)
                        dtype = (
                            torch.float16
                            if self.args.mixed_precision == "fp16"
                            else torch.bfloat16
                            if self.args.mixed_precision == "bf16"
                            else torch.float32
                        )

                        model_save_dir = os.path.join(self.save_folder,f'step_{global_step}')
                        model_to_save.save_pretrained(model_save_dir, safe_serialization=True)
                        del  model_to_save
                        
            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

            if accelerator.is_main_process and self.writer is not None:
                avg_loss = running_loss / len(self.train_dataloader)
                self.writer.add_scalar("Average Training Loss", avg_loss, epoch)

            if global_step >= self.state.train_steps:
                break

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            self.diffusion_model = unwrap_model(accelerator, self.diffusion_model)
            dtype = (
                torch.float16
                if self.args.mixed_precision == "fp16"
                else torch.bfloat16
                if self.args.mixed_precision == "bf16"
                else torch.float32
            )

            model_save_dir = os.path.join(self.save_folder,f'step_{global_step}')
            self.diffusion_model.save_pretrained(model_save_dir, safe_serialization=True)

        del self.diffusion_model, self.scheduler
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()


    def validate(self, accelerator, model_save_dir, global_step, n_view=1, n_chunk=30, image=None, prompt=None, cap=None, path=None, gt_actions=None, to_log=True):

        os.makedirs(model_save_dir,exist_ok=True)

        # Build bins for value decoding
        # value_bins = build_twohot_bins(num_bins=self.args.diffusion_model['config'].get("value_in_channels", 21))

        pipe = self.pipeline_class(
            self.scheduler, self.vae, self.text_encoder, self.tokenizer,
            unwrap_model(accelerator, self.diffusion_model) if accelerator is not None else self.diffusion_model
        )

        batch = next(iter(self.val_dataloader))
        image = batch['video'][:,:,:,:self.args.data['train']['n_previous']].clone()  # shape b,c,v,t,h,w 
        prompt = batch['caption']
        gt_video = batch['video']
        b, c, v, t, h, w = image.shape
        negative_prompt = ''

        batch_size = 1

        image = image[:batch_size]

        # Prepend start frame to mem window for inference
        if getattr(self.args, "use_start_video", False) and 'start_video' in batch:
            start_image = batch['start_video'][:batch_size]  # [b, c, v, 1, h, w]
            start_image = rearrange(start_image, 'b c v t h w -> (b v) c t h w')
            image = rearrange(image, 'b c v t h w -> (b v) c t h w')
            image = torch.cat([start_image, image], dim=2)  # [bv, c, n_prev+1, h, w]
        else:
            image = rearrange(image, 'b c v t h w -> (b v) c t h w')

        num_denois_steps = self.args.num_inference_step

        if self.args.return_action and getattr(self.args, "add_state", False):
            history_action_state = batch['state'][:batch_size]
            if history_action_state.shape[1] > 1:
                history_action_state = history_action_state[:, self.args.data['train']['n_previous']-1:self.args.data['train']['n_previous'], :]
            history_action_state = history_action_state.contiguous()
        else:
            history_action_state = None

        if self.args.return_value and getattr(self.args, "use_value_pose_condition", False):
            value_pose_state, value_pose_positions = self._prepare_value_pose_condition(
                batch['pose_history'][:batch_size],
                accelerator.device,
                self.state.weight_dtype,
            )
        else:
            value_pose_state = None
            value_pose_positions = None

        if getattr(self.args, "return_grounding", False) and getattr(self.args, "use_grounding_pose_condition", False):
            grounding_pose_state = batch['pose_history'][:batch_size].to(accelerator.device).contiguous()
            grounding_pose_positions = batch['pose_history'][:batch_size, :, -1].to(accelerator.device, dtype=torch.float32).contiguous()
        else:
            grounding_pose_state = None
            grounding_pose_positions = None

        if getattr(self.args, "use_video_frame_ids_rope", False) and 'frame_ids' in batch:
            mem_frame_ids = batch['frame_ids'][:batch_size, :self.args.data['train']['n_previous']].to(accelerator.device, dtype=torch.float32).contiguous()
            if getattr(self.args, "use_start_video", False) and 'start_video' in batch:
                mem_frame_ids = torch.cat([torch.zeros(batch_size, 1, device=mem_frame_ids.device, dtype=mem_frame_ids.dtype), mem_frame_ids], dim=1)
        else:
            mem_frame_ids = None

        preds = pipe.infer(
            image=image,
            prompt=prompt[:batch_size],
            negative_prompt=negative_prompt,
            num_inference_steps=num_denois_steps,
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            guidance_scale=1.0,
            height=h,
            width=w,
            n_view=v,
            return_action=self.args.return_action,
            return_value=self.args.return_value,
            return_grounding=getattr(self.args, "return_grounding", False),
            value_chunk=(self.args.data['train']['action_chunk'] if self.args.data['train'].get("value_dense", False) else self.args.data['train']['chunk']),
            grounding_chunk=(self.args.data['train']['action_chunk'] if self.args.data['train'].get("grounding_dense", True) else self.args.data['train']['chunk']),
            n_prev=self.args.data['train']['n_previous'] + (1 if getattr(self.args, "use_start_video", False) and 'start_video' in batch else 0),
            chunk=(self.args.data['train']['chunk']-1)//self.TEMPORAL_DOWN_RATIO+1,
            return_video=True,  # generate video for clean value features (not saved)
            noise_seed=int(getattr(self, "current_eval_noise_seed", 42)),
            action_chunk=self.args.data['train']['action_chunk'],
            history_action_state = history_action_state,
            value_pose_state = value_pose_state,
            value_pose_positions = value_pose_positions,
            grounding_pose_state = grounding_pose_state,
            grounding_pose_positions = grounding_pose_positions,
            mem_frame_ids = mem_frame_ids,
            use_video_frame_ids_rope = getattr(self.args, "use_video_frame_ids_rope", False),
            pixel_wise_timestep = self.args.pixel_wise_timestep,
            n_chunk=n_chunk,
            action_dim=self.args.diffusion_model["config"].get("action_in_channels", 14),
            value_dim=self.args.diffusion_model["config"].get("value_in_channels", 21),
            grounding_dim=self.args.diffusion_model["config"].get("grounding_in_channels", 768),
        )[0]

        cap = 'Validation'
        fps = int(getattr(self.args, "basic_fps", 30) / (self.args.data['train']['action_chunk'] // self.args.data['train']['chunk']))
        save_video(rearrange(gt_video[0].data.cpu(), 'c v t h w -> c t h (v w)', v=n_view), os.path.join(model_save_dir, f'{cap}_gt.mp4'), fps=fps)
        if (
            getattr(self.args, "use_goal_heatmap", False)
            or getattr(self.args, "use_goal_query_attention", False)
        ) and "goal_heatmap" in preds and "video" in preds:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            _hm_logits = rearrange(preds["goal_heatmap"].float().cpu(), "(b v) t hh ww -> b v t hh ww", v=n_view)
            _goal_eval_loss = (
                goal_query_distribution_loss
                if getattr(self.args, "use_goal_query_attention", False)
                else goal_heatmap_distribution_loss
            )
            _b, _nv, _nt, _hh, _ww = _hm_logits.shape
            logger.info("goal_hm_debug: pred_video_t=" + str(preds["video"].shape[2]) + " gt_video_t=" + str(gt_video.shape[3]) + " head_t=" + str(_nt) + " pred_all=" + str(tuple(preds["video"].shape)))
            try:
                _pv = preds["video"].float().cpu()
                _gv = gt_video.float().cpu()
                _ms = []
                for _pi in range(_pv.shape[2]):
                    row = []
                    for _gi in range(_gv.shape[3]):
                        _mv = 0.0
                        for _vi in range(_gv.shape[2]):
                            _mv += float((_pv[_vi, :, _pi] - _gv[0, :, _vi, _gi]).pow(2).mean())
                        row.append(round(_mv / _gv.shape[2], 4))
                    _ms.append(str(row))
                logger.info("goal_hm_debug: pred2gt_mse=" + "; ".join(_ms))
            except Exception as _e:
                logger.info("goal_hm_debug: mse_fail " + str(_e))
            # The last head frame maps to the actually decoded future frame. Earlier
            # decoded video slots are copied conditioning frames.
            _hidx = int(getattr(self.args, "goal_eval_head_frame", _nt - 1))
            _hidx = max(0, min(_hidx, _nt - 1))
            _ntv = int(preds["video"].shape[2]) if preds.get("video") is not None else 1
            _fidx = _ntv - (_nt - _hidx)  # map head index -> predicted-video index
            _fidx = max(0, min(_fidx, _ntv - 1))
            _gtidx = int(gt_video.shape[3]) - (_nt - _hidx)
            _gtidx = max(0, min(_gtidx, int(gt_video.shape[3]) - 1))
            logger.info(
                "goal_hm_debug: hidx=" + str(_hidx)
                + " fidx=" + str(_fidx)
                + " ntv=" + str(_ntv)
                + " gt_sel=" + str(_gtidx)
            )
            _hm = F.softmax(
                rearrange(_hm_logits, "b v t h w -> b t (v h w)"), dim=-1
            )
            _hm = rearrange(_hm, "b t (v h w) -> b v t h w", v=_nv, h=_hh, w=_ww)
            _hm_gt = batch.get("goal_heatmaps")
            if _hm_gt is not None:
                _hm_gt = _hm_gt[:batch_size].float().cpu()

            # ---- goal heatmap localization metrics (pred vs GT) ----
            if _hm_gt is not None and batch.get("goal_vis") is not None and batch.get("goal_conf") is not None:
                _vis = batch["goal_vis"][:batch_size, :, -_nt:].float().cpu()
                _conf = batch["goal_conf"][:batch_size, :, -_nt:].float().cpu()
                _gt_hm = _hm_gt[:batch_size].clone()
                if _gt_hm.shape[-2:] != (_hh, _ww):
                    _gg = rearrange(_gt_hm, "b v t h w -> (b v t) 1 h w")
                    _gt_hm = rearrange(
                        F.interpolate(_gg, size=(_hh, _ww), mode="bilinear", align_corners=False),
                        "(b v t) 1 h w -> b v t h w", b=_b, v=_nv, t=_nt)
                _gt_n = _gt_hm / _gt_hm.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                _hm_n = _hm[:batch_size] / _hm[:batch_size].amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                _d, _iou = [], []
                for _i_b in range(_b):
                    for _i_v in range(_nv):
                        for _i_t in range(_nt):
                            if float(_vis[_i_b, _i_v, _i_t]) <= 0.5 or float(_gt_hm[_i_b, _i_v, _i_t].amax()) <= 1e-4:
                                continue
                            _pa = torch.nonzero(_hm_n[_i_b, _i_v, _i_t] == _hm_n[_i_b, _i_v, _i_t].max()).float().mean(0)
                            _ga = torch.nonzero(_gt_n[_i_b, _i_v, _i_t] == _gt_n[_i_b, _i_v, _i_t].max()).float().mean(0)
                            _d.append((((_pa - _ga) ** 2).sum().sqrt() / _hh).item())
                            _pb = (_hm_n[_i_b, _i_v, _i_t] > 0.5).float().view(-1)
                            _gb = (_gt_n[_i_b, _i_v, _i_t] > 0.5).float().view(-1)
                            _un = ((_pb + _gb) > 0).sum().clamp_min(1)
                            _iou.append(((_pb * _gb).sum() / _un).item())
                _peak_dist = float(np.mean(_d)) if _d else float("nan")
                _iou_v = float(np.mean(_iou)) if _iou else float("nan")
                _kl_v = float(_goal_eval_loss(_hm_logits[:batch_size], _gt_hm, _vis, _conf).cpu().item())
                logger.info(f"goal_hm_eval: visible_frames={len(_d)} peak_dist(rel/32)={_peak_dist:.4f} iou0.5={_iou_v:.4f} val_kl={_kl_v:.4f}")
                if to_log:
                    self.writer.add_scalar("Goal heatmap val KL", _kl_v, global_step)
                    self.writer.add_scalar("Goal heatmap peak dist (rel32)", _peak_dist, global_step)
                    self.writer.add_scalar("Goal heatmap IoU@0.5", _iou_v, global_step)

                    if preds.get("video") is not None:
                        _vp_all = rearrange(preds["video"].float().cpu(), "(b v) c t h w -> b c v t h w", b=_b, v=_nv)
                        _gv_f1 = gt_video.float().cpu()[:, :, :, _gtidx]
                        _pv_f1 = _vp_all[:, :, :, _fidx]
                        _fmse_v = ((_pv_f1 - _gv_f1) ** 2).mean(dim=(0, 1)).mean(dim=(1, 2))  # (v,)
                        _fmse = _fmse_v.mean().item()
                        logger.info("goal_hm_eval: video_generated_mse=" + f"{_fmse:.4f} per_view={_fmse_v.tolist()}")
                        if to_log:
                            self.writer.add_scalar("Video generated-frame MSE", _fmse, global_step)
                    # ---- generated-frame metrics ----
                    _d1, _iou1 = [], []
                    for _i_b in range(_b):
                        for _i_v in range(_nv):
                            if float(_vis[_i_b, _i_v, _hidx]) <= 0.5 or float(_gt_hm[_i_b, _i_v, _hidx].amax()) <= 1e-4:
                                continue
                            _pa1 = torch.nonzero(_hm_n[_i_b, _i_v, _hidx] == _hm_n[_i_b, _i_v, _hidx].max()).float().mean(0)
                            _ga1 = torch.nonzero(_gt_n[_i_b, _i_v, _hidx] == _gt_n[_i_b, _i_v, _hidx].max()).float().mean(0)
                            _d1.append((((_pa1 - _ga1) ** 2).sum().sqrt() / _hh).item())
                            _pb1 = (_hm_n[_i_b, _i_v, _hidx] > 0.5).float().view(-1)
                            _gb1 = (_gt_n[_i_b, _i_v, _hidx] > 0.5).float().view(-1)
                            _iou1.append(((_pb1 * _gb1).sum() / ((_pb1 + _gb1) > 0).sum().clamp_min(1)).item())
                    _pd1 = float(np.mean(_d1)) if _d1 else float("nan")
                    _io1 = float(np.mean(_iou1)) if _iou1 else float("nan")
                    _kl1 = float(_goal_eval_loss(
                        _hm_logits[:batch_size, :, _hidx:_hidx + 1],
                        _gt_hm[:, :, _hidx:_hidx + 1],
                        _vis[:, :, _hidx:_hidx + 1],
                        _conf[:, :, _hidx:_hidx + 1]).cpu().item())
                    _dc1, _cv1 = [], []
                    for _i_b in range(_b):
                        for _i_v in range(_nv):
                            if float(_vis[_i_b, _i_v, _hidx]) <= 0.5 or float(_gt_hm[_i_b, _i_v, _hidx].amax()) <= 1e-4:
                                continue
                            _pm = _hm_n[_i_b, _i_v, _hidx]
                            _gm = _gt_n[_i_b, _i_v, _hidx]
                            _ry = torch.arange(_hh, dtype=_pm.dtype)[:, None]
                            _cx = torch.arange(_ww, dtype=_pm.dtype)[None, :]
                            _pwan = _pm.clamp_min(0); _pwt = _pwan.sum()
                            _gwan = _gm.clamp_min(0); _gwt = _gwan.sum()
                            if _pwt > 1e-8 and _gwt > 1e-8:
                                _pc = torch.stack([(_pwan * _ry).sum() / _pwt, (_pwan * _cx).sum() / _pwt])
                                _gc = torch.stack([(_gwan * _ry).sum() / _gwt, (_gwan * _cx).sum() / _gwt])
                                _dc1.append((((_pc - _gc) ** 2).sum().sqrt() / _hh).item())
                                _cv1.append(float((_gm * (_pm > 0.5)).sum() / _gm.sum().clamp_min(1e-8)))
                    _cd1 = float(np.mean(_dc1)) if _dc1 else float("nan")
                    _cov1 = float(np.mean(_cv1)) if _cv1 else float("nan")
                    logger.info(f"goal_hm_eval_f1: visible_frames={len(_d1)} peak_dist(rel/32)={_pd1:.4f} com_dist(rel/32)={_cd1:.4f} iou0.5={_io1:.4f} coverage={_cov1:.4f} val_kl={_kl1:.4f}")
                    if to_log:
                        self.writer.add_scalar("Goal heatmap F1 peak dist (rel32)", _pd1, global_step)
                        self.writer.add_scalar("Goal heatmap F1 COM dist (rel32)", _cd1, global_step)
                        self.writer.add_scalar("Goal heatmap F1 IoU@0.5", _io1, global_step)
                        self.writer.add_scalar("Goal heatmap F1 coverage", _cov1, global_step)
                        self.writer.add_scalar("Goal heatmap F1 val KL", _kl1, global_step)
            _v = preds["video"].float().cpu()
            try:
                np.savez(
                    os.path.join(model_save_dir, "pred_hm_debug.npz"),
                    pred_video=_v.float().numpy(),  # (bv, c, t, h, w) bv=views at batch 1
                    pred_heat=_hm[:batch_size].float().numpy() if "goal_heatmap" in preds else None,
                    gt_video_all=gt_video.float().cpu()[:batch_size].numpy(),
                    frame_ids=batch["frame_ids"][:batch_size].float().cpu().numpy() if "frame_ids" in batch else None,
                    gt_heat=_hm_gt[:batch_size].float().numpy() if _hm_gt is not None else None,
                    fidx=_fidx, hidx=_hidx, gtidx=_gtidx,
                    episode_id=int(getattr(self, "current_eval_episode_id", -1)),
                    sample_seed=int(getattr(self, "current_eval_sample_seed", -1)),
                    noise_seed=int(getattr(self, "current_eval_noise_seed", 42)),
                    repeat_id=int(getattr(self, "current_eval_repeat", 0)),
                    caption=np.asarray(prompt[:batch_size], dtype=str))
                logger.info("goal_hm_debug: saved pred_hm_debug.npz")
            except Exception as _e:
                logger.info("goal_hm_debug: save_fail " + str(_e))
            _tscale = 1.0
            if "goal_heatmap" in preds and _hm.shape[2] > 0:
                _tscale = float(_v.shape[2]) / float(_hm.shape[2]) * 0.0  # not used; heads are map-resolution
            _s = 256.0 / max(_hh, 1)  # map cell -> display pixel (maps are _hh x _ww in a 256-frame)
            for _vi in range(min(int(n_view), int(_v.shape[0]))):
                _pf = _v[_vi, :, _fidx].clamp(-1, 1)
                _pf = ((_pf + 1) / 2).permute(1, 2, 0).numpy()
                _gf = ((gt_video.data.cpu()[0, :, _vi, _gtidx] + 1) / 2).permute(1, 2, 0).numpy()
                _mp_f1 = _hm[0, _vi, _hidx].float()
                _mp = (_mp_f1 / _mp_f1.max().clamp_min(1e-8)).numpy()
                _ppk = torch.nonzero(_mp_f1 == _mp_f1.max()).float().mean(0)  # argmax centroid (row,col)
                _ry = torch.arange(_hh, dtype=_mp_f1.dtype)[:, None]
                _cx = torch.arange(_ww, dtype=_mp_f1.dtype)[None, :]
                _pwan = _mp_f1.clamp_min(0); _pwt = _pwan.sum()
                _ppc = torch.stack([(_pwan * _ry).sum() / _pwt.clamp_min(1e-8), (_pwan * _cx).sum() / _pwt.clamp_min(1e-8)])
                _has_gt = _hm_gt is not None and float(_hm_gt[0, _vi, _hidx].amax()) > 1e-4
                _dist_px = None; _iou_vv = None; _cov_vv = None; _dc_px = None
                if _has_gt:
                    _gtg = _hm_gt[0, _vi, _hidx].float()
                    _gtg_n = _gtg / _gtg.amax().clamp_min(1e-8)
                    _gwan = _gtg_n.clamp_min(0); _gwt = _gwan.sum()
                    _gpk = torch.nonzero(_gtg_n == _gtg_n.max()).float().mean(0)
                    _gpc = torch.stack([(_gwan * _ry).sum() / _gwt.clamp_min(1e-8), (_gwan * _cx).sum() / _gwt.clamp_min(1e-8)])
                    _dist_px = float((((_ppk - _gpk) ** 2).sum().sqrt()))
                    _dc_px = float((((_ppc - _gpc) ** 2).sum().sqrt()))
                    _pb = (_mp_f1 / _mp_f1.max().clamp_min(1e-8)) > 0.5
                    _gb = _gtg_n > 0.5
                    _iou_vv = float((_pb.float() * _gb.float()).sum() / ((_pb.float() + _gb.float()) > 0).sum().clamp_min(1))
                    _cov_vv = float((_gtg_n * _pb.float()).sum() / _gtg_n.sum().clamp_min(1e-8))
                fig, ax = plt.subplots(1, 2, figsize=(13, 6))
                _extent = (-0.5, _pf.shape[1] - 0.5, _pf.shape[0] - 0.5, -0.5)
                ax[0].imshow(_pf)
                ax[0].imshow(_mp, cmap="jet", alpha=0.5, extent=_extent)
                _mreg = np.ma.masked_where(~(_mp > 0.5), _mp)
                ax[0].imshow(_mreg, cmap="Greens", alpha=0.35, extent=_extent)
                ax[0].scatter([_ppc[1] * _s], [_ppc[0] * _s], c="white", marker="x", s=130, linewidths=2.5)
                ax[0].scatter([_ppk[1] * _s], [_ppk[0] * _s], c="black", marker=".", s=22)
                _tt = f"GENERATED pred v{_vi}: x=COM · . =argmax · green=hot"
                if _dc_px is not None and _iou_vv is not None and _cov_vv is not None:
                    _tt += f" | ΔCOM {_dc_px:.0f}px  IoU {_iou_vv:.2f}  cov {_cov_vv:.2f}"
                ax[0].set_title(_tt, fontsize=10); ax[0].axis("off")
                ax[1].imshow(_gf)
                if _has_gt:
                    ax[1].imshow(_gtg_n.numpy(), cmap="jet", alpha=0.5, extent=_extent)
                    ax[1].scatter([_gpc[1] * _s], [_gpc[0] * _s], c="white", marker="x", s=130, linewidths=2.5)
                    ax[1].set_title(f"GT f{_vi}: teacher heatmap; x = GT COM", fontsize=10)
                else:
                    ax[1].set_title(f"GT f{_vi}: (no visible target this frame)", fontsize=10)
                ax[1].axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(model_save_dir, f"{cap}_goalheat_v{_vi}_f1.png"), dpi=160); plt.close(fig)


        if self.args.return_video:
            video = preds['video'].data.cpu()
            save_video(rearrange(video, '(b v) c t h w -> b c t h (v w)', v=n_view)[0], os.path.join(model_save_dir, f'{cap}.mp4'), fps=fps)

        if to_log:
            self.writer.add_text(f'step_{global_step}/{cap} prompt:', prompt[0], global_step)

        if self.args.return_action:
            # shape t, c
            gt_actions = batch['actions'][:, -self.args.data['train']['action_chunk']:]
            action_dim = gt_actions.shape[-1]

            action_logs = act_metric(
                preds['action'][:,:,:action_dim].detach().cpu().to(torch.float).numpy()[:batch_size],
                gt_actions[:,:,:action_dim].detach().cpu().to(torch.float).numpy()[:batch_size],
                prefix=cap,
                start_stop_interval=[(0,1),(1,9),(9,25),(25,self.args.data['train']['action_chunk'])]
            )

            if to_log:
                for key, value in action_logs.items():
                    self.writer.add_scalar(key, value, global_step)

        if self.args.return_value:
            value_chunk = (self.args.data['train']['action_chunk'] if self.args.data['train'].get("value_dense", False) else self.args.data['train']['chunk'])

            gt_values = batch['value_targets'][:, -value_chunk:][:batch_size]  # shape b, t, c extract only last dimension (actual value)

            # Get predicted value (twohot logits)
            pd_values = preds['value'][:batch_size]  # shape b, t, c

            if getattr(self.args, "value_target_type", "scalar") == "feature":
                value_logs = feature_value_metric(
                    pd_values.detach().cpu().to(torch.float),
                    gt_values.detach().cpu().to(torch.float),
                    prefix=f'{cap}_step_{global_step}',
                    start_stop_interval=[(0,1),(1,value_chunk)],
                )
            else:
                # Compute value metrics
                value_logs = value_metric(
                    pd_values.detach().cpu().to(torch.float),
                    gt_values.detach().cpu().to(torch.float),
                    prefix=f'{cap}_step_{global_step}',
                    start_stop_interval=[(0,1),(1,value_chunk)],
                    save_plot=True,
                    plot_dir=model_save_dir,
                    disable_symlog=getattr(self.args, "disable_value_symlog", False)
                )

            if to_log:
                for key, value in value_logs.items():
                    self.writer.add_scalar(key, value, global_step)

        if getattr(self.args, "return_grounding", False):
            grounding_chunk = (self.args.data['train']['action_chunk'] if self.args.data['train'].get("grounding_dense", True) else self.args.data['train']['chunk'])
            gt_grounding = batch['grounding_targets'][:, -grounding_chunk:][:batch_size]
            pd_grounding = preds['grounding'][:batch_size]
            grounding_logs = feature_value_metric(
                pd_grounding.detach().cpu().to(torch.float),
                gt_grounding.detach().cpu().to(torch.float),
                prefix=f'{cap}_grounding_step_{global_step}',
                start_stop_interval=[(0,1),(1,grounding_chunk)],
            )

            if to_log:
                for key, value in grounding_logs.items():
                    self.writer.add_scalar(key, value, global_step)
