# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


# ============== ALG (Adaptive Low-Pass Guidance) Utilities ==============

def apply_alg_low_pass_filter(
    tensor: torch.Tensor,
    filter_type: str,
    blur_sigma: float = None,
    blur_kernel_size: float = None,
    resize_factor: float = None,
):
    """
    Applies the specified low-pass filtering operation to the input tensor.
    Handles 4D ([B, C, H, W]) and 5D ([B, C, F, H, W]) tensors by temporarily
    reshaping 5D tensors for spatial filtering.
    """
    # Early exits for no-op cases
    if filter_type == "none":
        return tensor
    if filter_type == "down_up" and resize_factor == 1.0:
        return tensor
    if filter_type == "gaussian_blur" and blur_sigma == 0:
        return tensor

    # Reshape 5D tensor for spatial filtering
    is_5d = tensor.ndim == 5
    if is_5d:
        B, C, K, H, W = tensor.shape
        # Flatten frames into batch dimension using view
        tensor = tensor.view(B * K, C, H, W)
    else:
        B, C, H, W = tensor.shape

    # Apply selected filter
    if filter_type == "gaussian_blur":
        if isinstance(blur_kernel_size, float):
            kernel_val = max(int(blur_kernel_size * H), 1)
        else:
            kernel_val = int(blur_kernel_size)
        if kernel_val % 2 == 0:
            kernel_val += 1
        # Use torchvision's gaussian_blur
        import torchvision.transforms.functional as tvF
        tensor = tvF.gaussian_blur(tensor, kernel_size=[kernel_val, kernel_val], sigma=[blur_sigma, blur_sigma])

    elif filter_type == "down_up":
        h0, w0 = tensor.shape[-2:]
        h1 = max(1, int(round(h0 * resize_factor)))
        w1 = max(1, int(round(w0 * resize_factor)))
        tensor = F.interpolate(tensor, size=(h1, w1), mode="bilinear", align_corners=False, antialias=True)
        tensor = F.interpolate(tensor, size=(h0, w0), mode="bilinear", align_corners=False, antialias=True)

    # Restore original 5D shape if necessary
    if is_5d:
        tensor = tensor.view(B, C, K, H, W)

    return tensor


def get_alg_lp_strength(
    step_index: int,
    total_steps: int,
    lp_strength_schedule_type: str,
    # Interval params
    schedule_interval_start_time: float = 0.0,
    schedule_interval_end_time: float = 0.05,
    # Linear params
    schedule_linear_start_weight: float = 1.0,
    schedule_linear_end_weight: float = 0.0,
    schedule_linear_end_time: float = 0.5,
    # Exponential params
    schedule_exp_decay_rate: float = 10.0,
) -> float:
    """
    Calculates the low-pass guidance strength multiplier for the current timestep
    based on the specified schedule.
    """
    step_norm = step_index / max(total_steps - 1, 1)

    if lp_strength_schedule_type == "linear":
        schedule_duration_fraction = schedule_linear_end_time
        if schedule_duration_fraction <= 0:
            return schedule_linear_start_weight
        if step_norm >= schedule_duration_fraction:
            current_strength = schedule_linear_end_weight
        else:
            progress = step_norm / schedule_duration_fraction
            current_strength = schedule_linear_start_weight * (1 - progress) + schedule_linear_end_weight * progress
        return current_strength

    elif lp_strength_schedule_type == "interval":
        if schedule_interval_start_time <= step_norm <= schedule_interval_end_time:
            return 1.0
        else:
            return 0.0

    elif lp_strength_schedule_type == "exponential":
        decay_rate = schedule_exp_decay_rate
        if decay_rate < 0:
            logging.warning(f"Warning: Negative exponential_decay_rate ({decay_rate}) is unusual. Using abs value.")
            decay_rate = abs(decay_rate)
        return math.exp(-decay_rate * step_norm)

    elif lp_strength_schedule_type == "none":
        return 1.0
    else:
        logging.warning(f"Warning: Unknown lp_strength_schedule_type '{lp_strength_schedule_type}'. Using constant strength 1.0.")
        return 1.0


class WanI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def _prepare_alg_lp_condition(self,
                                  orig_image,
                                  y_condition,
                                  msk,
                                  frame_num,
                                  h, w,
                                  lp_filter_type,
                                  lp_filter_in_latent,
                                  lp_blur_sigma,
                                  lp_blur_kernel_size,
                                  lp_resize_factor,
                                  seed_g):
        """
        Prepares low-pass filtered condition for ALG.
        
        Args:
            orig_image: Original input image tensor [C, H, W]
            y_condition: VAE-encoded condition with mask [20, frames//4+1, lat_h, lat_w]
            msk: Mask tensor [4, frames//4+1, lat_h, lat_w]
            frame_num: Number of frames
            h, w: Height and width  
            lp_filter_type: Type of low-pass filter
            lp_filter_in_latent: Whether to filter in latent space
            lp_blur_sigma: Gaussian blur sigma
            lp_blur_kernel_size: Gaussian blur kernel size
            lp_resize_factor: Resize factor for down-up filtering
            seed_g: Random generator
            
        Returns:
            Low-pass filtered condition tensor
        """
        if not lp_filter_in_latent:
            # Filter in image space
            image_lp = apply_alg_low_pass_filter(
                orig_image.unsqueeze(0),  # Add batch dimension
                filter_type=lp_filter_type,
                blur_sigma=lp_blur_sigma,
                blur_kernel_size=lp_blur_kernel_size,
                resize_factor=lp_resize_factor,
            ).squeeze(0)  # Remove batch dimension
            
            # Prepare video tensor with filtered first frame
            video_lp = torch.concat([
                F.interpolate(image_lp[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
                torch.zeros(3, frame_num - 1, h, w)
            ], dim=1).to(self.device)
            
            # Encode through VAE - Note: Wan2.1's VAE doesn't expose latent_dist.sample()
            # so we use the direct output, which is already normalized in Wan's implementation
            y_lp = self.vae.encode([video_lp])[0]
            y_lp_condition = torch.concat([msk, y_lp])
        else:
            # Filter in latent space - filter the ENTIRE condition tensor (mask + latents)
            # This matches the diffusers implementation exactly
            y_lp_condition = apply_alg_low_pass_filter(
                y_condition.unsqueeze(0),  # Add batch dimension
                filter_type=lp_filter_type,
                blur_sigma=lp_blur_sigma,
                blur_kernel_size=lp_blur_kernel_size,
                resize_factor=lp_resize_factor,
            ).squeeze(0)  # Remove batch dimension
            
        return y_lp_condition

    def generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 disable_tqdm=False,
                 use_alg=False,
                 alg_kwargs=None):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            use_alg (`bool`, *optional*, defaults to False):
                If True, enables Adaptive Low-Pass Guidance (ALG) for improved motion dynamics
            alg_kwargs (`dict`, *optional*, defaults to None):
                ALG configuration parameters. If None, uses default values

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # Set default ALG kwargs matching diffusers wan_alg.yaml config
        # These are the ACTUAL ALG defaults from configs/wan_alg.yaml
        if alg_kwargs is None:
            alg_kwargs = {
                'lp_filter_type': 'down_up',
                'lp_filter_in_latent': True,
                'lp_blur_sigma': 15.0,  # Not used with down_up
                'lp_blur_kernel_size': 0.02734375,  # Not used with down_up
                'lp_resize_factor': 0.4,
                'lp_strength_schedule_type': 'interval',
                'schedule_blur_kernel_size': False,
                'schedule_interval_start_time': 0.0,
                'schedule_interval_end_time': 0.20,
                'schedule_linear_start_weight': 1.0,  # Not used with interval
                'schedule_linear_end_weight': 0.0,  # Not used with interval
                'schedule_linear_end_time': 0.5,  # Not used with interval
                'schedule_exp_decay_rate': 10.0,  # Not used with interval
            }
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16, (F - 1) // 4 + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, 81, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F - 1, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            arg_c = {
                'context': [context[0]],
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }

            arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }

            if offload_model:
                torch.cuda.empty_cache()

            self.model.to(self.device)
            for step_idx, t in enumerate(tqdm(timesteps, disable=disable_tqdm)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                # ALG: Compute low-pass strength and prepare conditions
                if guide_scale > 1.0 and use_alg:
                    lp_strength = get_alg_lp_strength(
                        step_index=step_idx,
                        total_steps=sampling_steps,
                        lp_strength_schedule_type=alg_kwargs['lp_strength_schedule_type'],
                        schedule_interval_start_time=alg_kwargs['schedule_interval_start_time'],
                        schedule_interval_end_time=alg_kwargs['schedule_interval_end_time'],
                        schedule_linear_start_weight=alg_kwargs['schedule_linear_start_weight'],
                        schedule_linear_end_weight=alg_kwargs['schedule_linear_end_weight'],
                        schedule_linear_end_time=alg_kwargs['schedule_linear_end_time'],
                        schedule_exp_decay_rate=alg_kwargs['schedule_exp_decay_rate'],
                    )
                    
                    if lp_strength > 0.0:
                        # Modulate filter parameters based on strength
                        modulated_lp_blur_sigma = alg_kwargs['lp_blur_sigma'] * lp_strength
                        modulated_lp_blur_kernel_size = (
                            alg_kwargs['lp_blur_kernel_size'] * lp_strength 
                            if alg_kwargs['schedule_blur_kernel_size'] 
                            else alg_kwargs['lp_blur_kernel_size']
                        )
                        modulated_lp_resize_factor = 1.0 - (1.0 - alg_kwargs['lp_resize_factor']) * lp_strength
                        
                        # Prepare low-pass filtered condition
                        y_lp_condition = self._prepare_alg_lp_condition(
                            orig_image=img,
                            y_condition=y,
                            msk=msk,
                            frame_num=F,
                            h=h, w=w,
                            lp_filter_type=alg_kwargs['lp_filter_type'],
                            lp_filter_in_latent=alg_kwargs['lp_filter_in_latent'],
                            lp_blur_sigma=modulated_lp_blur_sigma,
                            lp_blur_kernel_size=modulated_lp_blur_kernel_size,
                            lp_resize_factor=modulated_lp_resize_factor,
                            seed_g=seed_g
                        )
                        
                        # Three-pass inference for ALG
                        # Pass 1: Unconditional with original condition
                        noise_pred_uncond_orig = self.model(
                            latent_model_input, t=timestep, **arg_null)[0].to(
                                torch.device('cpu') if offload_model else self.device)
                        if offload_model:
                            torch.cuda.empty_cache()
                        
                        # Pass 2: Unconditional with low-pass condition
                        arg_null_lp = {
                            'context': context_null,
                            'clip_fea': clip_context,
                            'seq_len': max_seq_len,
                            'y': [y_lp_condition],
                        }
                        noise_pred_uncond_lp = self.model(
                            latent_model_input, t=timestep, **arg_null_lp)[0].to(
                                torch.device('cpu') if offload_model else self.device)
                        if offload_model:
                            torch.cuda.empty_cache()
                        
                        # Pass 3: Conditional with low-pass condition
                        arg_c_lp = {
                            'context': [context[0]],
                            'clip_fea': clip_context,
                            'seq_len': max_seq_len,
                            'y': [y_lp_condition],
                        }
                        noise_pred_cond_lp = self.model(
                            latent_model_input, t=timestep, **arg_c_lp)[0].to(
                                torch.device('cpu') if offload_model else self.device)
                        if offload_model:
                            torch.cuda.empty_cache()
                        
                        # Combine predictions using ALG formula
                        noise_pred = noise_pred_uncond_orig + guide_scale * (noise_pred_cond_lp - noise_pred_uncond_lp)
                    else:
                        # Standard two-pass CFG when lp_strength = 0
                        noise_pred_cond = self.model(
                            latent_model_input, t=timestep, **arg_c)[0].to(
                                torch.device('cpu') if offload_model else self.device)
                        if offload_model:
                            torch.cuda.empty_cache()
                        noise_pred_uncond = self.model(
                            latent_model_input, t=timestep, **arg_null)[0].to(
                                torch.device('cpu') if offload_model else self.device)
                        if offload_model:
                            torch.cuda.empty_cache()
                        noise_pred = noise_pred_uncond + guide_scale * (
                            noise_pred_cond - noise_pred_uncond)
                else:
                    # Standard two-pass CFG when ALG is disabled
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0].to(
                            torch.device('cpu') if offload_model else self.device)
                    if offload_model:
                        torch.cuda.empty_cache()
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, **arg_null)[0].to(
                            torch.device('cpu') if offload_model else self.device)
                    if offload_model:
                        torch.cuda.empty_cache()
                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)

                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent.to(self.device)]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
