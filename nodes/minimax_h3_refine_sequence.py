"""Sequence-aware MiniMax H3 refinement for H3 Continuum chunk lists.

ComfyUI normally maps a node independently over list-valued inputs.  That is not
semantically sufficient for H3 Continuum refinement: the learned LBH upscaler is
temporal, so independently upscaling the copied low-resolution continuation
prefix does not guarantee that it remains identical to the previous chunk after
the second sampling pass.

This wrapper consumes the complete chunk list once, refines it in order, and
replaces each protected high-resolution continuation prefix with the actual
post-refine tail of the preceding chunk before sampler 2 runs.  The existing
Core-native H3 denoise mask then preserves that carried prefix exactly.
"""
from __future__ import annotations

from typing import Any

import torch

from .minimax_h3_refine import (
    MinimaxH3LatentUpscaler3DRefine as _BaseRefine,
    _model_with_refinement_contract,
    _resolve_refine_contract,
    build_clean_h3_upscale,
    run_h3_refinement,
)
from .minimax_h3_refine_support import _nested_members, _validate_h3_av_samples


def _value_at(value: Any, index: int) -> Any:
    """Mirror ComfyUI's repeat-last list mapping for an INPUT_IS_LIST node."""
    if not isinstance(value, list):
        return value
    if not value:
        return None
    return value[index if index < len(value) else -1]


def _contiguous_zero_prefix_steps(mask: torch.Tensor, temporal_axis: int) -> int:
    """Count leading temporal positions that are fully protected (mask == 0)."""
    if not torch.is_tensor(mask):
        raise TypeError("H3 refinement noise-mask member must be a torch.Tensor")
    axis = int(temporal_axis)
    if axis < 0:
        axis += mask.ndim
    if axis < 0 or axis >= mask.ndim:
        raise ValueError("H3 refinement noise-mask temporal axis is invalid")
    moved = mask.movedim(axis, 0)
    protected = moved.reshape(int(moved.shape[0]), -1).eq(0).all(dim=1)
    count = 0
    for item in protected.detach().to(device="cpu").tolist():
        if not bool(item):
            break
        count += 1
    return count


def _carry_previous_refined_prefix(
    clean: dict,
    previous_refined: dict,
    *,
    carry_audio: bool,
) -> tuple[int, int]:
    """Restore Continuum's exact previous-generated-latent invariant at sampler 2.

    ``clean`` already contains the spatially enlarged denoise mask.  Only a
    *fully* protected prefix is eligible for cross-chunk carry; arbitrary spatial
    masks/keyframes are never interpreted as continuation.
    """
    mask = clean.get("noise_mask")
    if mask is None:
        return 0, 0
    members = _nested_members(mask)
    if len(members) != 2:
        raise ValueError(
            "MiniMax H3 refinement noise_mask must contain exactly [video_mask, audio_mask]"
        )
    video_mask, audio_mask = members
    current_video, current_audio = _validate_h3_av_samples(clean["samples"])
    previous_video, previous_audio = _validate_h3_av_samples(previous_refined["samples"])

    video_steps = _contiguous_zero_prefix_steps(video_mask, 2)
    if video_steps:
        if video_steps > int(previous_video.shape[2]) or video_steps > int(current_video.shape[2]):
            raise ValueError("Protected H3 video continuation prefix exceeds available latent time")
        if (
            int(previous_video.shape[0]) != int(current_video.shape[0])
            or int(previous_video.shape[1]) != int(current_video.shape[1])
            or tuple(previous_video.shape[-2:]) != tuple(current_video.shape[-2:])
        ):
            raise ValueError(
                "Sequential H3 refinement requires identical video geometry across Continuum chunks"
            )
        current_video[:, :, :video_steps].copy_(previous_video[:, :, -video_steps:])

    audio_steps = 0
    if carry_audio:
        audio_steps = _contiguous_zero_prefix_steps(audio_mask, -1)
        if audio_steps:
            if audio_steps > int(previous_audio.shape[-1]) or audio_steps > int(current_audio.shape[-1]):
                raise ValueError("Protected H3 audio continuation prefix exceeds available latent time")
            if tuple(previous_audio.shape[:-1]) != tuple(current_audio.shape[:-1]):
                raise ValueError(
                    "Sequential H3 refinement requires identical audio structure across Continuum chunks"
                )
            current_audio[..., :audio_steps].copy_(previous_audio[..., -audio_steps:])

    return video_steps, audio_steps


class MinimaxH3LatentUpscaler3DRefineSequence(_BaseRefine):
    """Sequence-aware form of the integrated learned-upscale/refinement node."""

    # Continuum emits video/audio/refine_state as list outputs.  Receiving the
    # lists intact is required so sampler 2 can carry the *post-refine* tail of
    # chunk N into the protected prefix of chunk N+1.
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    DESCRIPTION = (
        "Learned LBH 3D upscale followed by low-sigma MiniMax H3 refinement. "
        "H3 Continuum chunk lists are refined sequentially so each Native-Masked "
        "continuation prefix is replaced by the exact post-refine tail of the "
        "preceding chunk before sampler 2. The LATENT output is final/decode-ready."
    )

    def upscale_and_refine(
        self,
        latent,
        noise,
        sampler,
        sigmas,
        model_name,
        mode,
        scale,
        width,
        height,
        megapixels,
        align,
        keep_proportion,
        lock_audio,
        cfg,
        device,
        precision,
        audio_latent=None,
        refine_state=None,
        model=None,
        positive=None,
        negative=None,
    ):
        # Keep direct Python callers/backward-compatible unit integrations working
        # with the historical scalar signature.  ComfyUI itself supplies a list
        # here because INPUT_IS_LIST=True, even for a one-chunk workflow.
        if not isinstance(latent, list):
            return super().upscale_and_refine(
                latent,
                noise,
                sampler,
                sigmas,
                model_name,
                mode,
                scale,
                width,
                height,
                megapixels,
                align,
                keep_proportion,
                lock_audio,
                cfg,
                device,
                precision,
                audio_latent=audio_latent,
                refine_state=refine_state,
                model=model,
                positive=positive,
                negative=negative,
            )
        if not latent:
            raise ValueError("MiniMax H3 refinement received an empty LATENT list")

        outputs: list[dict] = []
        previous_refined: dict | None = None
        for index, current_latent in enumerate(latent):
            current_state = _value_at(refine_state, index)
            current_model = _value_at(model, index)
            current_positive = _value_at(positive, index)
            refine_model, refine_positive = _resolve_refine_contract(
                current_state,
                current_model,
                current_positive,
            )
            current_negative = None if current_state is not None else _value_at(negative, index)
            current_lock_audio = bool(_value_at(lock_audio, index))

            clean, positive_out, negative_out = build_clean_h3_upscale(
                current_latent,
                refine_positive,
                model_name=_value_at(model_name, index),
                mode=_value_at(mode, index),
                scale=float(_value_at(scale, index)),
                width=int(_value_at(width, index)),
                height=int(_value_at(height, index)),
                megapixels=float(_value_at(megapixels, index)),
                align=int(_value_at(align, index)),
                keep_proportion=bool(_value_at(keep_proportion, index)),
                device=_value_at(device, index),
                precision=_value_at(precision, index),
                lock_audio=current_lock_audio,
                audio_latent=_value_at(audio_latent, index),
                negative=current_negative,
            )

            if current_state is not None and previous_refined is not None:
                video_steps, audio_steps = _carry_previous_refined_prefix(
                    clean,
                    previous_refined,
                    carry_audio=not current_lock_audio,
                )
                if video_steps or audio_steps:
                    detail = f"video={video_steps} latent step(s)"
                    if audio_steps:
                        detail += f", audio={audio_steps} step(s)"
                    print(
                        f"[MinimaxH3-3D Refine] Continuum chunk {index + 1}: "
                        f"carried exact post-refine prefix from chunk {index} ({detail})"
                    )

            refine_model = _model_with_refinement_contract(refine_model)
            refined = run_h3_refinement(
                clean,
                model=refine_model,
                positive=positive_out,
                negative=negative_out,
                noise=_value_at(noise, index),
                sampler=_value_at(sampler, index),
                sigmas=_value_at(sigmas, index),
                cfg=float(_value_at(cfg, index)),
                lock_audio=current_lock_audio,
            )
            outputs.append(refined)
            previous_refined = refined

        return (outputs,)


NODE_CLASS_MAPPINGS = {
    "MinimaxH3LatentUpscaler3DRefineHandoff": MinimaxH3LatentUpscaler3DRefineSequence,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MinimaxH3LatentUpscaler3DRefineHandoff": "MiniMax H3 Latent Upscaler + Refine (3D)",
}
