"""LTX2 VAE stages — video encode/decode (and optional audio decode)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from unirl.types.primitives import Video, Videos

if TYPE_CHECKING:
    from .bundle import LTX2Bundle

# LTX-2 audio-VAE geometry fallbacks (used when the audio_vae config does not
# expose these). Match the diffusion stage's constants: 16 kHz vocoder output,
# 64 mel bins, 4x mel compression. Kept local to avoid importing the diffusion
# module's private names.
_AUDIO_SAMPLING_RATE: int = 16000
_AUDIO_MEL_BINS: int = 64
_AUDIO_MEL_COMPRESSION: int = 4


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
    def decode(self, latents: torch.Tensor) -> Videos:
        """Decode (already-denormalized) video latents → packed ``Videos``.

        Args:
            latents: (B, C, T_lat, H_lat, W_lat) in VAE latent space,
                ALREADY denormalized by the pipeline (``_denormalize_latents``).

        Returns:
            ``Videos`` (varlen-packed) with per-frame values in ``[0, 1]``.
        """
        # Decode in fp32: LTX2's VAE decoder (like most) is numerically
        # unstable in bf16. Mirror WAN21VAEDecodeStage.
        vae = self.vae
        latents_f32 = latents.to(torch.float32)

        # The LTX2 VAE is timestep-conditioned: its decoder multiplies a
        # required ``temb`` by a scale factor, so passing ``None`` crashes
        # (None * Parameter). diffusers' pipeline feeds decode_timestep=0.0
        # (and decode_noise_scale defaults to it → the pre-decode noise
        # injection is a no-op), so a zeros timestep reproduces inference.
        timestep = None
        if bool(getattr(vae.config, "timestep_conditioning", False)):
            timestep = torch.zeros(latents_f32.shape[0], device=latents_f32.device, dtype=latents_f32.dtype)

        decoded = vae.to(torch.float32).decode(latents_f32, timestep, return_dict=False)[0]

        # Decoder emits [B, C, T, H, W] in [-1, 1]; normalize to [0, 1].
        decoded = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0).to(self.dtype)

        # Pack into the varlen ``Videos`` primitive: ``Video.frames`` is
        # [T, C, H, W], so permute each sample (C, T, H, W) → (T, C, H, W)
        # and let ``Videos.from_list`` concat along T (computing cu_seqlens).
        videos = [Video(frames=decoded[i].permute(1, 0, 2, 3).contiguous()) for i in range(int(decoded.shape[0]))]
        return Videos.from_list(videos)


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
    """Decode packed audio latents → waveform via audio VAE + vocoder (LTX-2).

    Mirrors the diffusers ``LTX2Pipeline`` audio-decode path (and Flow-Factory's
    reference ``decode``): ``denormalize → unpack → audio_vae.decode → vocoder``.
    Note the order DIFFERS from video (video unpacks first, then denormalizes).
    """

    def __init__(self, bundle: "LTX2Bundle") -> None:
        if bundle.audio_vae is None or bundle.vocoder is None:
            raise RuntimeError("LTX2AudioDecodeStage requires audio_vae and vocoder (LTX-2 audio checkpoint).")
        self.audio_vae = bundle.audio_vae
        self.vocoder = bundle.vocoder
        self.dtype = bundle.dtype
        # Mel-bin geometry: prefer the audio_vae config, fall back to the LTX-2
        # defaults shared with the diffusion stage (64 mel bins, 4x compression).
        self._num_mel_bins = int(getattr(self.audio_vae.config, "mel_bins", _AUDIO_MEL_BINS))
        self._mel_compression = int(getattr(self.audio_vae.config, "mel_compression_ratio", _AUDIO_MEL_COMPRESSION))

    @property
    def sampling_rate(self) -> int:
        """Vocoder output sample rate (Hz). Reward backends need this to resample."""
        for obj in (self.vocoder, self.audio_vae):
            sr = getattr(getattr(obj, "config", None), "sampling_rate", None)
            if sr is not None:
                return int(sr)
        return _AUDIO_SAMPLING_RATE

    @staticmethod
    def _unpack_audio_latents(latents: torch.Tensor, num_frames: int, num_mel_bins: int) -> torch.Tensor:
        """Packed ``(B, seq, C·mel)`` → ``(B, C, T, mel)`` — inverse of the audio
        pack. ``seq == num_frames`` and the feature dim splits into
        ``(channels, mel_bins)``; mirrors diffusers ``_unpack_audio_latents``.
        """
        batch_size, seq_len, feat = latents.shape
        channels = feat // num_mel_bins
        latents = latents.reshape(batch_size, seq_len, channels, num_mel_bins)
        # (B, T, C, mel) → (B, C, T, mel)
        return latents.permute(0, 2, 1, 3).contiguous()

    def _denormalize_audio_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Channel-wise denormalize packed audio latents by the audio VAE's
        ``latents_mean/std`` (diffusers ``_denormalize_audio_latents``). No-op
        if the VAE exposes no normalization stats.
        """
        vae = self.audio_vae
        mean = getattr(vae, "latents_mean", None)
        std = getattr(vae, "latents_std", None)
        if mean is None or std is None:
            return latents
        # Packed latents are (B, seq, C·mel); stats are per audio channel, so
        # broadcast over the feature dim is handled by the unpacked decode below.
        mean = mean.flatten().to(latents.device, latents.dtype)
        std = std.flatten().to(latents.device, latents.dtype)
        scaling = float(getattr(vae.config, "scaling_factor", 1.0))
        # Stats length matches the feature dim when packed; tile if per-channel.
        if mean.numel() == latents.shape[-1]:
            return latents * std / scaling + mean
        return latents

    @torch.no_grad()
    def decode(self, audio_latents: torch.Tensor, num_audio_frames: int) -> torch.Tensor:
        """Decode packed audio latents → waveform.

        Args:
            audio_latents: Packed audio latents ``(B, seq, C·mel)`` from the
                diffusion stage (same packed layout fed to the transformer).
            num_audio_frames: Number of audio LATENT frames (``seq`` length),
                from :func:`unirl.models.ltx2.diffusion._audio_num_frames`.

        Returns:
            Waveform tensor ``(B, L)`` (or ``(B, C, L)``) at :attr:`sampling_rate`.
        """
        latent_mel_bins = max(1, self._num_mel_bins // self._mel_compression)
        # 1. Denormalize FIRST, then unpack (order differs from video!).
        aud = self._denormalize_audio_latents(audio_latents)
        # 2. Unpack: (B, seq, C·mel) → (B, C, T, mel).
        aud = self._unpack_audio_latents(aud, num_audio_frames, latent_mel_bins)
        # 3. Audio VAE decode → mel spectrogram.
        mel = self.audio_vae.decode(aud.to(self.audio_vae.dtype), return_dict=False)[0]
        # 4. Vocoder → waveform.
        waveform = self.vocoder(mel)
        return waveform


__all__ = ["LTX2VAEDecodeStage", "LTX2VAEEncodeStage", "LTX2AudioDecodeStage"]
