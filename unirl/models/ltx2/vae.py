"""LTX2 VAE stages — video encode/decode (and optional audio decode)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .bundle import LTX2Bundle


class LTX2VAEDecodeStage:
    """Decode latents → video frames via the LTX2 3D-VAE.

    The LTX2 VAE uses 32x spatial and 8x temporal compression with 128
    latent channels. Latents are in shape (B, C, T_lat, H_lat, W_lat).
    """

    def __init__(self, bundle: "LTX2Bundle") -> None:
        self.vae = bundle.vae
        self.dtype = bundle.dtype
        self.device = bundle.device

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode video latents → pixel frames.

        Args:
            latents: (B, C, T_lat, H_lat, W_lat) in VAE latent space.

        Returns:
            Video frames (B, C, T, H, W) in [0, 1] float range.
        """
        # LTX2 VAE expects latents scaled by the model's scale factor
        latents = latents.to(dtype=self.vae.dtype)
        frames = self.vae.decode(latents).sample
        # Clamp to [0, 1]
        frames = frames.clamp(0.0, 1.0)
        return frames.to(self.dtype)


class LTX2VAEEncodeStage:
    """Encode video frames → latents for I2V conditioning.

    Used to encode the first frame (source image) into latent space
    for image-to-video conditioning.
    """

    def __init__(self, bundle: "LTX2Bundle") -> None:
        self.vae = bundle.vae
        self.dtype = bundle.dtype
        self.device = bundle.device

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames → latents.

        Args:
            frames: (B, C, T, H, W) or (B, C, H, W) pixel values in [0, 1].

        Returns:
            Latents (B, C_lat, T_lat, H_lat, W_lat).
        """
        if frames.dim() == 4:
            # Single frame → add temporal dim
            frames = frames.unsqueeze(2)
        frames = frames.to(dtype=self.vae.dtype)
        latents = self.vae.encode(frames).latent_dist.sample()
        return latents.to(self.dtype)


class LTX2AudioDecodeStage:
    """Decode audio latents → waveform via audio VAE + vocoder (LTX-2.3)."""

    def __init__(self, bundle: "LTX2Bundle") -> None:
        if bundle.audio_vae is None or bundle.vocoder is None:
            raise RuntimeError("LTX2AudioDecodeStage requires audio_vae and vocoder (LTX-2.3 checkpoint).")
        self.audio_vae = bundle.audio_vae
        self.vocoder = bundle.vocoder
        self.dtype = bundle.dtype

    @torch.no_grad()
    def decode(self, audio_latents: torch.Tensor) -> torch.Tensor:
        """Decode audio latents → waveform.

        Args:
            audio_latents: Audio latent tensor from the diffusion stage.

        Returns:
            Audio waveform tensor.
        """
        # Audio VAE decode → mel spectrogram
        mel = self.audio_vae.decode(audio_latents.to(self.audio_vae.dtype)).sample
        # Vocoder → waveform
        waveform = self.vocoder(mel)
        return waveform


__all__ = ["LTX2VAEDecodeStage", "LTX2VAEEncodeStage", "LTX2AudioDecodeStage"]
