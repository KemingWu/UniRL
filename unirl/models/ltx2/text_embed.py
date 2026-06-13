"""LTX2 text embedding stage — Gemma3 encoding + connector projection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts

if TYPE_CHECKING:
    from .bundle import LTX2Bundle


class LTX2TextEmbedStage:
    """Encode text prompts via Gemma3 + optional connector projection.

    For LTX-2.3 with connectors: Gemma3 → connector → (video_embeds, audio_embeds).
    For LTX-2.0 without connectors: Gemma3 → caption_projection on transformer.
    """

    def __init__(self, bundle: "LTX2Bundle") -> None:
        self.text_encoder = bundle.text_encoder
        self.tokenizer = bundle.tokenizer
        self.connectors = bundle.connectors
        self.max_sequence_length = bundle.max_sequence_length
        self.dtype = bundle.dtype
        self.device = bundle.device

    @torch.no_grad()
    def encode(
        self,
        texts: Texts,
        negative_texts: Optional[Texts] = None,
    ) -> dict:
        """Encode prompts → TextEmbedCondition for video (and optionally audio).

        Returns a dict with keys: 'text', optionally 'audio_text',
        'negative_text', 'negative_audio_text'.
        """
        prompt_embeds, attention_mask = self._encode_prompts(texts.texts)

        result = {}

        if self.connectors is not None:
            # LTX-2.3 path: route through connectors
            video_embeds, audio_embeds, connector_mask = self._apply_connectors(prompt_embeds, attention_mask)
            result["text"] = TextEmbedCondition(embeds=video_embeds, attn_mask=connector_mask)
            result["audio_text"] = TextEmbedCondition(embeds=audio_embeds, attn_mask=connector_mask)
        else:
            # LTX-2.0 path: raw embeddings (caption_projection is on transformer)
            result["text"] = TextEmbedCondition(embeds=prompt_embeds, attn_mask=attention_mask)

        # Negative prompts for CFG
        if negative_texts is not None:
            neg_embeds, neg_mask = self._encode_prompts(negative_texts.texts)
            if self.connectors is not None:
                neg_video, neg_audio, neg_conn_mask = self._apply_connectors(neg_embeds, neg_mask)
                result["negative_text"] = TextEmbedCondition(embeds=neg_video, attn_mask=neg_conn_mask)
                result["negative_audio_text"] = TextEmbedCondition(embeds=neg_audio, attn_mask=neg_conn_mask)
            else:
                result["negative_text"] = TextEmbedCondition(embeds=neg_embeds, attn_mask=neg_mask)

        return result

    def _encode_prompts(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and encode through Gemma3."""
        inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.text_encoder(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
        )
        # Use last hidden state as text embeddings
        hidden_states = outputs.hidden_states[-1]  # (B, seq, D)
        return hidden_states.to(self.dtype), inputs.attention_mask

    def _apply_connectors(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route Gemma3 outputs through connectors → video + audio embeddings."""
        # Connector outputs video and audio projections
        connector_out = self.connectors(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        video_embeds = connector_out.video_features  # (B, seq, D_video)
        audio_embeds = connector_out.audio_features  # (B, seq, D_audio)
        out_mask = connector_out.attention_mask  # (B, seq)
        return video_embeds, audio_embeds, out_mask


__all__ = ["LTX2TextEmbedStage"]
