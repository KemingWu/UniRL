"""LTX2 diffusion: per-step kernel + rollout-level stage.

Two classes:
- ``LTX2DiffusionStep`` — stateless per-step kernel.
- ``LTX2DiffusionStage`` — implements ``DiffusionStage[LTX2Conditions]``.

LTX2-specific deviations from other models:
- Unified video+audio latent space: video and audio are concatenated on the
  sequence dimension before the transformer, split after.
- Video uses SDE (stochastic, log_prob for RL gradients).
- Audio uses ODE (deterministic, no gradients) — trained jointly but not
  directly optimized by the RL signal.
- The transformer takes ``hidden_states`` (patchified latents) +
  ``encoder_hidden_states`` (text embeddings) + ``encoder_attention_mask``.
- Timestep is scaled by 1000 (flow matching convention).
- RoPE is computed internally by the transformer from spatial/temporal shapes.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import ClassVar, List, Optional, Set

import torch

from unirl.models.types.diffusion import DiffusionStage, DiffusionStep
from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import StepStrategy
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment, make_video_segment
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import LTX2Bundle
from .conditions import LTX2Conditions
from .config import LTX2_SPATIAL_COMPRESSION, LTX2_TEMPORAL_COMPRESSION

_LTX2_TIMESTEP_SCALE: float = 1000.0

# LTX-2 is a unified audiovisual transformer: its ``forward`` ALWAYS runs both
# the video and audio branches and returns ``(video_out, audio_out)``, even for
# pure T2V. We therefore feed a minimal zeroed audio placeholder (one latent
# frame) and discard the audio output. ``isolate_modalities=True`` disables the
# audio↔video cross-attention so the placeholder cannot perturb the video
# prediction. No audio VAE is needed — every audio dimension comes off the
# transformer config / audio RoPE.
_LTX2_T2V_AUDIO_FRAMES: int = 1
# Playback rate used only to scale RoPE temporal coords; LTX-2 trains at 24fps
# (matches ``LTX2PipelineConfig.default_frame_rate``).
_LTX2_FRAME_RATE: float = 24.0


class LTX2DiffusionStep(DiffusionStep[LTX2Bundle, LTX2Conditions]):
    """Per-step LTX2 denoising kernel — stateless.

    Handles the video-only forward (SDE path for RL). Audio is handled
    separately via ODE in the stage.
    """

    def predict_noise(
        self,
        model: LTX2Bundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: LTX2Conditions,
        *,
        guidance_scale: float,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        """Run the LTX2 audiovisual transformer with optional CFG, returning
        the VIDEO velocity prediction.

        Args:
            model: LTX2Bundle containing the transformer.
            sample: Patchified video latents (B, seq, C).
            sigma: Current noise level (B,).
            conditions: Text embeddings + optional image latent.
            guidance_scale: CFG scale (1.0 = no CFG).
            latent_num_frames / latent_height / latent_width: Video LATENT grid
                dims (post-VAE-compression), needed by the transformer to build
                video RoPE coords.

        Returns:
            Predicted video noise (velocity), same shape as ``sample``. The
            transformer's audio output is discarded (see module docstring).
        """
        transformer = model.transformer
        timestep = (sigma * _LTX2_TIMESTEP_SCALE).to(sample.device)

        # Text conditioning
        text_cond = conditions.text
        encoder_hidden_states = text_cond.embeds
        encoder_attention_mask = text_cond.attn_mask

        # Minimal zeroed audio placeholder. LTX-2's forward always runs the
        # audio branch; isolate_modalities=True keeps it from affecting video.
        audio_in_channels = int(getattr(transformer.config, "audio_in_channels", 128))

        def _run(sample_in, ts_in, enc_hs, enc_mask):
            bsz = sample_in.shape[0]
            audio_hidden_states = torch.zeros(
                (bsz, _LTX2_T2V_AUDIO_FRAMES, audio_in_channels),
                device=sample_in.device,
                dtype=sample_in.dtype,
            )
            out = transformer(
                hidden_states=sample_in,
                audio_hidden_states=audio_hidden_states,
                encoder_hidden_states=enc_hs,
                audio_encoder_hidden_states=enc_hs,
                timestep=ts_in,
                audio_timestep=ts_in,
                encoder_attention_mask=enc_mask,
                audio_encoder_attention_mask=enc_mask,
                num_frames=latent_num_frames,
                height=latent_height,
                width=latent_width,
                fps=_LTX2_FRAME_RATE,
                audio_num_frames=_LTX2_T2V_AUDIO_FRAMES,
                isolate_modalities=True,
                return_dict=False,
            )
            # forward returns (video_out, audio_out); keep video only.
            return out[0]

        if guidance_scale > 1.0 and conditions.negative_text is not None:
            # CFG: batch [uncond, cond]
            neg_cond = conditions.negative_text
            sample_cfg = torch.cat([sample, sample], dim=0)
            timestep_cfg = torch.cat([timestep, timestep], dim=0)
            encoder_hs_cfg = torch.cat([neg_cond.embeds, encoder_hidden_states], dim=0)
            encoder_mask_cfg = torch.cat([neg_cond.attn_mask, encoder_attention_mask], dim=0)

            noise_pred = _run(sample_cfg, timestep_cfg, encoder_hs_cfg, encoder_mask_cfg)
            noise_uncond, noise_cond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = _run(sample, timestep, encoder_hidden_states, encoder_attention_mask)

        return noise_pred


class LTX2DiffusionStage(DiffusionStage[LTX2Conditions]):
    """LTX2 diffusion stage — owns the denoising loop and replay.

    FSDP wrapping hint: the transformer's block class is
    ``LTX2VideoTransformerBlock``.
    """

    _no_split_modules: ClassVar[List[str]] = ["LTX2VideoTransformerBlock"]

    def __init__(
        self,
        bundle: LTX2Bundle,
        *,
        strategy: StepStrategy,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
    ) -> None:
        self.bundle = bundle
        self.step_kernel = LTX2DiffusionStep()
        self.strategy = strategy
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")

    def trainable_module(self) -> torch.nn.Module:
        """The trainable transformer (for FSDP wrapping)."""
        return self.bundle.transformer

    @staticmethod
    def _latent_geometry(params: DiffusionSamplingParams) -> tuple[int, int, int]:
        """Video LATENT grid ``(T_lat, H_lat, W_lat)`` from pixel-space params.

        Mirrors ``LTX2Pipeline.latent_shape``: 32x spatial, 8x temporal (causal,
        so ``T_lat = (num_frames - 1) // 8 + 1``). The transformer needs these
        to build video RoPE coords inside ``predict_noise``.
        """
        latent_t = (int(params.num_frames) - 1) // LTX2_TEMPORAL_COMPRESSION + 1
        latent_h = int(params.height) // LTX2_SPATIAL_COMPRESSION
        latent_w = int(params.width) // LTX2_SPATIAL_COMPRESSION
        return latent_t, latent_h, latent_w

    def generate(
        self,
        conditions: LTX2Conditions,
        *,
        params: DiffusionSamplingParams,
        sigmas: torch.Tensor,
        initial_latents: torch.Tensor,
        sde_indices: Optional[List[int]] = None,
    ) -> LatentSegment:
        """Run the full denoising loop, collecting trajectory for RL.

        Args:
            conditions: Text/image conditioning.
            params: Sampling parameters (guidance_scale, eta, etc.).
            sigmas: Sigma schedule (T+1,) from high → 0.
            initial_latents: Starting noise (B, seq, C) or (B, C, T, H, W).
            sde_indices: Which steps to use SDE (stochastic) for RL.

        Returns:
            LatentSegment with trajectory and log-probs at SDE steps.
        """
        guidance_scale = float(params.guidance_scale)
        eta = float(params.eta)
        latent_t, latent_h, latent_w = self._latent_geometry(params)

        # Determine SDE step set
        num_steps = len(sigmas) - 1
        sde_set: Set[int] = set(sde_indices) if sde_indices else set(range(num_steps))

        # Trajectory storage
        latent_trajectory = []
        log_probs = []

        x = initial_latents
        autocast_ctx = (
            torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype != torch.float32 else nullcontext()
        )

        with autocast_ctx:
            for step_idx in range(num_steps):
                sigma = sigmas[step_idx].expand(x.shape[0])
                sigma_next = sigmas[step_idx + 1].expand(x.shape[0])
                is_sde = step_idx in sde_set

                noise_pred = self.step_kernel.predict_noise(
                    self.bundle,
                    x,
                    sigma,
                    conditions,
                    guidance_scale=guidance_scale,
                    latent_num_frames=latent_t,
                    latent_height=latent_h,
                    latent_width=latent_w,
                )

                if is_sde:
                    # SDE step with log-prob for RL training
                    x_next, lp = self.strategy.denoise_with_logp(x, noise_pred, sigma, sigma_next, eta=eta)
                    latent_trajectory.append(x.to(self.trajectory_dtype))
                    log_probs.append(lp.to(self.logprob_dtype))
                else:
                    # ODE step (deterministic, no log-prob)
                    x_next = self.strategy.denoise(x, noise_pred, sigma, sigma_next, eta=0.0)

                x = x_next

        # Build segment
        segment = make_video_segment(
            final_latents=x,
            latent_trajectory=latent_trajectory if latent_trajectory else None,
            sde_logp=torch.stack(log_probs, dim=1) if log_probs else None,
            sde_indices=sorted(sde_set) if sde_set else None,
            sigmas=sigmas,
        )
        return segment

    def replay(
        self,
        conditions: LTX2Conditions,
        *,
        segment: LatentSegment,
        params: DiffusionSamplingParams,
        step_indices: List[int],
    ) -> ReplayResult:
        """Replay specific SDE steps to recompute log-probs for training.

        Used by DiffusionGRPO to compute new_logp (or replay old_logp).
        """
        guidance_scale = float(params.guidance_scale)
        eta = float(params.eta)
        sigmas = segment.sigmas
        latent_t, latent_h, latent_w = self._latent_geometry(params)

        log_probs = []
        autocast_ctx = (
            torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype != torch.float32 else nullcontext()
        )

        with autocast_ctx:
            for local_idx, step_idx in enumerate(step_indices):
                # Retrieve cached latent at this step
                x = segment.latent_trajectory[:, local_idx].to(device=self.bundle.device, dtype=self.autocast_dtype)
                sigma = sigmas[step_idx].expand(x.shape[0])
                sigma_next = sigmas[step_idx + 1].expand(x.shape[0])

                noise_pred = self.step_kernel.predict_noise(
                    self.bundle,
                    x,
                    sigma,
                    conditions,
                    guidance_scale=guidance_scale,
                    latent_num_frames=latent_t,
                    latent_height=latent_h,
                    latent_width=latent_w,
                )

                _, lp = self.strategy.denoise_with_logp(x, noise_pred, sigma, sigma_next, eta=eta)
                log_probs.append(lp)

        return ReplayResult(log_probs=torch.stack(log_probs, dim=1).to(self.logprob_dtype))


__all__ = ["LTX2DiffusionStep", "LTX2DiffusionStage"]
