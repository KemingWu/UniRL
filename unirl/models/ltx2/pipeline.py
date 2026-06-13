"""LTX2 pipeline — T2V / I2V / T2AV dispatch.

Composes text embedding, diffusion, and VAE decode stages into a complete
rollout pipeline. Mode is determined by the request primitives:
- T2V: text only → video generation
- I2V: text + image → image-conditioned video generation
- T2AV: text → video + audio joint generation (LTX-2.3)
"""

from __future__ import annotations

import logging
from typing import Optional

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import StepStrategy
from unirl.sde.runtime import get_sigma_schedule
from unirl.types.primitives import Images, Texts, Videos
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import get_diffusion_params

from .bundle import LTX2Bundle
from .conditions import LTX2Conditions
from .config import LTX2PipelineConfig
from .diffusion import LTX2DiffusionStage
from .text_embed import LTX2TextEmbedStage
from .vae import LTX2VAEDecodeStage, LTX2VAEEncodeStage

logger = logging.getLogger(__name__)


class LTX2Pipeline(Pipeline):
    """LTX-2/2.3 T2V / I2V / T2AV pipeline."""

    def __init__(
        self,
        *,
        bundle: LTX2Bundle,
        text_embed: LTX2TextEmbedStage,
        diffusion: LTX2DiffusionStage,
        vae_decode: LTX2VAEDecodeStage,
        vae_encode: Optional[LTX2VAEEncodeStage],
        config: LTX2PipelineConfig,
    ) -> None:
        self.bundle = bundle
        self.text_embed = text_embed
        self.diffusion = diffusion
        self.vae_decode = vae_decode
        self.vae_encode = vae_encode
        self.config = config

    @classmethod
    def from_config(
        cls,
        config: LTX2PipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "LTX2Pipeline":
        """Build pipeline from config + bundle."""
        bundle = LTX2Bundle.from_config(config)
        return cls.from_bundle(config=config, bundle=bundle, strategy=strategy)

    @classmethod
    def from_bundle(
        cls,
        *,
        config: LTX2PipelineConfig,
        bundle: LTX2Bundle,
        strategy: Optional[StepStrategy] = None,
    ) -> "LTX2Pipeline":
        """Build pipeline stages from an existing bundle."""
        from unirl.sde.kernels import FlowSDEStrategy

        if strategy is None:
            strategy = FlowSDEStrategy()

        text_embed = LTX2TextEmbedStage(bundle)
        diffusion = LTX2DiffusionStage(
            bundle,
            strategy=strategy,
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
        )
        vae_decode = LTX2VAEDecodeStage(bundle)
        vae_encode = LTX2VAEEncodeStage(bundle)

        return cls(
            bundle=bundle,
            text_embed=text_embed,
            diffusion=diffusion,
            vae_decode=vae_decode,
            vae_encode=vae_encode,
            config=config,
        )

    def generate(self, req: RolloutReq) -> RolloutResp:
        """Run T2V / I2V / T2AV based on request primitives."""
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"LTX2Pipeline.generate: req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )

        images = req.primitives.get("image")
        params = get_diffusion_params(req.sampling_params)
        if params is None:
            raise ValueError("LTX2Pipeline.generate: DiffusionSamplingParams required.")

        # Determine mode
        has_image = isinstance(images, Images)

        # 1. Text embedding
        negative_texts = req.primitives.get("negative_text")
        neg = negative_texts if isinstance(negative_texts, Texts) else None
        embed_result = self.text_embed.encode(texts, negative_texts=neg)

        # 2. Build conditions
        conditions = LTX2Conditions.from_dict(embed_result)

        # I2V: encode condition image
        if has_image and self.vae_encode is not None:
            image_latents = self.vae_encode.encode(images.pixels)
            conditions.image_latent = image_latents

        # 3. Sigma schedule
        num_steps = int(params.num_inference_steps)
        sigmas = get_sigma_schedule(
            num_steps=num_steps,
            shift=self.config.shift,
            device=self.bundle.device,
        )
        if req.sigmas is not None:
            sigmas = req.sigmas.to(self.bundle.device)

        # 4. Initial latents
        # TODO: compute latent shape from config + params (height, width, num_frames)
        # For now, expect initial_latents from the driver
        initial_latents = req.request_conditions.get("initial_latents")
        if initial_latents is None:
            raise ValueError(
                "LTX2Pipeline.generate: initial_latents must be provided in "
                "req.request_conditions['initial_latents'] (driver-authoritative noise)."
            )

        # 5. Diffusion loop
        sde_indices = list(params.sde_indices) if params.sde_indices is not None else None
        segment = self.diffusion.generate(
            conditions,
            params=params,
            sigmas=sigmas,
            initial_latents=initial_latents,
            sde_indices=sde_indices,
        )

        # 6. VAE decode → video frames
        decoded_video = self.vae_decode.decode(segment.final_latents)
        decoded = Videos(frames=decoded_video)

        # 7. Build response
        track = RolloutTrack(
            sample_ids=list(req.sample_ids),
            group_ids=list(req.group_ids),
            conditions=conditions.to_dict(),
            segment=segment,
            decoded={"video": decoded},
        )

        return RolloutResp(tracks={"diffusion": track})


__all__ = ["LTX2Pipeline"]
