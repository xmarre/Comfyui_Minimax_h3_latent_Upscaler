"""Shared MiniMax H3 helpers for integrated learned-latent refinement.

This module contains only AV validation, conditioning/mask geometry, and delegation
to the existing LBH 3D learned upscaler. Sampling belongs to
``minimax_h3_refine``; there is intentionally no intermediate handoff node.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

H3_VIDEO_CHANNELS = 24
H3_AUDIO_CHANNELS = 32
H3_AUDIO_CHANNELS_PER_SAMPLE = 2
H3_SPATIAL_PATCH = 2


class _FallbackNested:
    """Tiny NestedTensor-like fallback used only when ComfyUI is unavailable in tests."""

    is_nested = True

    def __init__(self, tensors):
        self.tensors = list(tensors)

    def unbind(self):
        return self.tensors

    def to(self, device):
        return _FallbackNested([tensor.to(device) for tensor in self.tensors])


def _lbh_module():
    """Import the existing LBH 3D implementation lazily."""
    from . import minimax_h3_latent_upscaler_3d as lbh

    return lbh


def _scan_models():
    """Return model choices exposed by the existing LBH loader."""
    return _lbh_module().scan_models()


def _upscale_modes() -> tuple[str, ...]:
    """Return the exact sizing modes supported by the LBH 3D node."""
    return tuple(member.value for member in _lbh_module().UpscaleMode)


def _is_nested_tensor(value: Any) -> bool:
    """Return whether ``value`` is a Comfy NestedTensor or compatible test double."""
    try:
        import comfy.nested_tensor
    except ImportError:
        return bool(getattr(value, "is_nested", False)) and hasattr(value, "unbind")
    return isinstance(value, comfy.nested_tensor.NestedTensor) or (
        bool(getattr(value, "is_nested", False)) and hasattr(value, "unbind")
    )


def _nested_members(value: Any) -> list[torch.Tensor]:
    """Extract and type-check members from a NestedTensor-like object."""
    if not _is_nested_tensor(value):
        raise TypeError(f"Expected NestedTensor, got {type(value).__name__}")
    members = list(value.unbind())
    if any(not isinstance(member, torch.Tensor) for member in members):
        raise TypeError("MiniMax H3 NestedTensor members must be torch.Tensor instances")
    return members


def _validate_video(video: Any) -> torch.Tensor:
    """Validate one native MiniMax H3 video latent tensor."""
    if not isinstance(video, torch.Tensor):
        raise TypeError("MiniMax H3 video LATENT samples must be a torch.Tensor")
    if video.ndim != 5 or video.shape[1] != H3_VIDEO_CHANNELS:
        raise ValueError(
            "MiniMax H3 video stream must be Bx24xTxHxW, "
            f"got {tuple(video.shape)}."
        )
    if video.shape[0] != 1:
        raise ValueError(
            f"MiniMax H3 currently supports batch size 1; video batch={video.shape[0]}."
        )
    if min(video.shape[2:]) < 1:
        raise ValueError("MiniMax H3 video stream must have non-empty temporal/spatial axes")
    if not video.is_floating_point():
        raise TypeError("MiniMax H3 video latent must use a floating-point dtype")
    return video


def _validate_audio(audio: Any) -> torch.Tensor:
    """Validate one native MiniMax H3 audio latent tensor."""
    if not isinstance(audio, torch.Tensor):
        raise TypeError("MiniMax H3 audio LATENT samples must be a torch.Tensor")
    if (
        audio.ndim != 4
        or audio.shape[1] != H3_AUDIO_CHANNELS
        or audio.shape[2] != H3_AUDIO_CHANNELS_PER_SAMPLE
    ):
        raise ValueError(
            "MiniMax H3 audio stream must be Bx32x2xT, "
            f"got {tuple(audio.shape)}."
        )
    if audio.shape[0] != 1:
        raise ValueError(
            f"MiniMax H3 currently supports batch size 1; audio batch={audio.shape[0]}."
        )
    if audio.shape[-1] < 1:
        raise ValueError("MiniMax H3 audio stream must have a non-empty temporal axis")
    if not audio.is_floating_point():
        raise TypeError("MiniMax H3 audio latent must use a floating-point dtype")
    return audio


def _validate_h3_av_samples(samples: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate native joint H3 ``NestedTensor([video, audio])`` samples."""
    members = _nested_members(samples)
    if len(members) != 2:
        raise ValueError(
            "MiniMax H3 native AV sampling expects exactly two NestedTensor members "
            f"[video, audio]; got {len(members)}."
        )
    return _validate_video(members[0]), _validate_audio(members[1])


def _plain_latent_samples(latent: Any, *, label: str) -> torch.Tensor:
    """Extract a plain tensor from a LATENT dictionary used by split-stream nodes."""
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError(f"{label} must be a LATENT dictionary containing 'samples'")
    samples = latent["samples"]
    if _is_nested_tensor(samples):
        members = _nested_members(samples)
        if len(members) != 1:
            raise ValueError(
                f"{label} split-stream LATENT must contain one tensor, got {len(members)}"
            )
        samples = members[0]
    if not isinstance(samples, torch.Tensor):
        raise TypeError(f"{label} LATENT samples must be a torch.Tensor")
    return samples


def _resolve_h3_sources(
    latent: dict,
    audio_latent: dict | None,
) -> tuple[torch.Tensor, torch.Tensor, Any, bool]:
    """Resolve native joint AV or H3 Continuum split video/audio LATENT inputs."""
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("latent must be a LATENT dictionary containing 'samples'")
    samples = latent["samples"]
    if _is_nested_tensor(samples):
        if audio_latent is not None:
            raise ValueError(
                "audio_latent must be left disconnected when latent already contains native joint AV samples"
            )
        video, audio = _validate_h3_av_samples(samples)
        return video, audio, samples, False

    video = _validate_video(samples)
    if audio_latent is None:
        raise ValueError(
            "Plain 24-channel video LATENT detected. Connect the matching audio_latent "
            "(for H3 Continuum V3/V3.4, connect its audio_latents output) so the refine "
            "node can rebuild native MiniMax H3 joint AV state."
        )
    audio = _validate_audio(_plain_latent_samples(audio_latent, label="audio_latent"))
    return video, audio, None, True


def _wrap_h3_samples(video: torch.Tensor, audio: torch.Tensor, template: Any = None) -> Any:
    """Build native two-member H3 NestedTensor samples."""
    try:
        import comfy.nested_tensor
    except ImportError:
        if template is not None and _is_nested_tensor(template):
            try:
                return type(template)([video, audio])
            except Exception:
                pass
        return _FallbackNested([video, audio])
    return comfy.nested_tensor.NestedTensor([video, audio])


def _resize_spatial(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Nearest-resize only H/W while preserving time, rank, dtype, and device."""
    if tensor.ndim == 3:
        out = F.interpolate(tensor.unsqueeze(1).float(), size=(height, width), mode="nearest")
        return out.squeeze(1).to(dtype=tensor.dtype)
    if tensor.ndim == 4:
        out = F.interpolate(tensor.float(), size=(height, width), mode="nearest")
        return out.to(dtype=tensor.dtype)
    if tensor.ndim == 5:
        b, c, t, h, w = tensor.shape
        work = tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
        out = F.interpolate(work, size=(height, width), mode="nearest")
        out = out.reshape(b, t, c, height, width).permute(0, 2, 1, 3, 4)
        return out.to(dtype=tensor.dtype)
    raise ValueError(f"Spatial tensor must have rank 3, 4, or 5, got {tensor.ndim}")


def _h3_padded_spatial_size(video: torch.Tensor) -> tuple[int, int]:
    """Return H/W of H3's internal even 2x2 target patch grid."""
    h, w = int(video.shape[-2]), int(video.shape[-1])
    return h + (-h) % H3_SPATIAL_PATCH, w + (-w) % H3_SPATIAL_PATCH


def _resize_target_keyframe(block: dict, target_h: int, target_w: int) -> dict:
    """Resize one target-grid keyframe latent to H3's internal padded target grid."""
    out = dict(block)
    latent = out.get("latent")
    if latent is None:
        return out
    if not isinstance(latent, torch.Tensor):
        raise TypeError(
            f"MiniMax keyframe latent must be torch.Tensor, got {type(latent).__name__}"
        )
    if latent.ndim not in (4, 5) or latent.shape[1] != H3_VIDEO_CHANNELS:
        raise ValueError(
            "MiniMax H3 keyframe visual latent must be Bx24xHxW or Bx24xTxHxW, "
            f"got {tuple(latent.shape)}."
        )
    resized = _resize_spatial(latent, target_h, target_w)
    out["latent"] = resized
    if "latent_h" in out:
        out["latent_h"] = int(target_h)
    if "latent_w" in out:
        out["latent_w"] = int(target_w)
    if "latent_t" in out and resized.ndim == 5:
        out["latent_t"] = int(resized.shape[2])
    return out


def resize_h3_target_conditioning(
    conditioning: list | None,
    target_h: int,
    target_w: int,
) -> list | None:
    """Clone conditioning and resize only target-grid ``minimax_keyframes``."""
    if conditioning is None:
        return None
    result = []
    for entry in conditioning:
        if (
            not isinstance(entry, (list, tuple))
            or len(entry) < 2
            or not isinstance(entry[1], dict)
        ):
            result.append(entry)
            continue
        metadata = entry[1]
        new_meta = metadata.copy()
        keyframes = metadata.get("minimax_keyframes")
        if keyframes is not None:
            new_meta["minimax_keyframes"] = [
                _resize_target_keyframe(block, target_h, target_w) for block in keyframes
            ]
        rebuilt = [entry[0], new_meta, *entry[2:]]
        result.append(tuple(rebuilt) if isinstance(entry, tuple) else rebuilt)
    return result


def _video_mask_for_target(mask: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
    """Nearest-resize one video denoise mask to the exact unpadded target H/W."""
    if not isinstance(mask, torch.Tensor) or mask.ndim not in (3, 4, 5):
        raise ValueError("Video noise_mask must be a rank-3, rank-4, or rank-5 tensor")
    if mask.shape[0] not in (1, video.shape[0]):
        raise ValueError("Video noise_mask batch size is incompatible with the H3 video latent")
    if mask.ndim == 5 and mask.shape[2] not in (1, video.shape[2]):
        raise ValueError("Video noise_mask temporal size must be 1 or match the H3 video latent")
    return _resize_spatial(mask, video.shape[-2], video.shape[-1]).to(device=video.device)


def _audio_mask_for_target(mask: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
    """Validate/broadcast one split or joint audio denoise mask onto the H3 audio stream."""
    if not isinstance(mask, torch.Tensor):
        raise TypeError("Audio noise_mask must be a torch.Tensor")
    work = mask.to(device=audio.device, dtype=torch.float32)
    while work.ndim < audio.ndim:
        work = work.unsqueeze(1)
    try:
        return work.expand(audio.shape[0], 1, audio.shape[2], audio.shape[3]).clone()
    except RuntimeError as exc:
        raise ValueError(
            f"Audio noise_mask shape {tuple(mask.shape)} is incompatible with audio latent {tuple(audio.shape)}"
        ) from exc


def _constant_mask(member: torch.Tensor, value: float) -> torch.Tensor:
    """Create a single-channel mask matching a video or audio latent member."""
    return torch.full(
        (member.shape[0], 1, *member.shape[2:]),
        float(value),
        dtype=torch.float32,
        device=member.device,
    )


def _split_source_masks(
    latent: dict,
    audio_latent: dict | None,
    *,
    split_input: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Extract source video/audio masks from joint or H3 Continuum split LATENTs."""
    existing = latent.get("noise_mask")
    if split_input:
        video_mask = None
        audio_mask = None
        if existing is not None:
            if _is_nested_tensor(existing):
                raise ValueError("Split video LATENT must not carry a nested AV noise_mask")
            video_mask = existing
        if audio_latent is not None and audio_latent.get("noise_mask") is not None:
            candidate = audio_latent["noise_mask"]
            if _is_nested_tensor(candidate):
                members = _nested_members(candidate)
                if len(members) != 1:
                    raise ValueError("Split audio_latent noise_mask must contain one member")
                candidate = members[0]
            audio_mask = candidate
        return video_mask, audio_mask

    if existing is None:
        return None, None
    if _is_nested_tensor(existing):
        members = _nested_members(existing)
        if len(members) != 2:
            raise ValueError(
                "MiniMax H3 nested noise_mask must contain exactly [video_mask, audio_mask]"
            )
        return members[0], members[1]
    return existing, None


def prepare_h3_noise_mask(
    source_latent: dict,
    audio_latent: dict | None,
    video: torch.Tensor,
    audio: torch.Tensor,
    samples_template: Any,
    *,
    split_input: bool,
    lock_audio: bool,
) -> Any | None:
    """Rebuild an explicit AV denoise mask on the high-resolution joint latent."""
    video_mask, audio_mask = _split_source_masks(
        source_latent, audio_latent, split_input=split_input
    )
    if video_mask is None and audio_mask is None and not lock_audio:
        return None

    out_video = (
        _video_mask_for_target(video_mask, video)
        if video_mask is not None
        else _constant_mask(video, 1.0)
    )
    if lock_audio:
        out_audio = _constant_mask(audio, 0.0)
    elif audio_mask is not None:
        out_audio = _audio_mask_for_target(audio_mask, audio)
    else:
        out_audio = _constant_mask(audio, 1.0)
    return _wrap_h3_samples(out_video, out_audio, samples_template)


def _zero_audio_noise(samples: Any, template: Any) -> Any:
    """Zero only the H3 audio noise member."""
    video, audio = _validate_h3_av_samples(samples)
    return _wrap_h3_samples(video, torch.zeros_like(audio), template)


def _node_output_first(value: Any) -> Any:
    """Extract first output from current Comfy NodeOutput or tuple/list fallback."""
    args = getattr(value, "args", None)
    if args is not None:
        if not args:
            raise RuntimeError("LBH 3D upscaler returned an empty NodeOutput")
        return args[0]
    if isinstance(value, (tuple, list)) and value:
        return value[0]
    raise TypeError(f"Unexpected LBH 3D node output type: {type(value).__name__}")


def run_lbh_video_upscale(
    video: torch.Tensor,
    model_name: str,
    mode: str,
    scale: float,
    width: int,
    height: int,
    megapixels: float,
    align: int,
    keep_proportion: bool,
    device: str,
    precision: str,
) -> torch.Tensor:
    """Delegate the video member to the existing LBH 3D node without duplicating its loader."""
    lbh = _lbh_module()
    mode_config = {
        "mode": lbh.UpscaleMode(mode),
        "scale": scale,
        "width": width,
        "height": height,
        "megapixels": megapixels,
    }
    result = lbh.MinimaxH3LatentUpscaler3D.execute(
        {"samples": video},
        model_name,
        mode_config,
        align,
        keep_proportion,
        device,
        precision,
    )
    output_latent = _node_output_first(result)
    if not isinstance(output_latent, dict) or "samples" not in output_latent:
        raise RuntimeError("LBH 3D upscaler did not return a LATENT dictionary")
    upscaled = output_latent["samples"]
    if not isinstance(upscaled, torch.Tensor):
        raise TypeError("LBH 3D upscaler returned non-tensor samples")
    if upscaled.ndim != 5 or upscaled.shape[:3] != video.shape[:3]:
        raise RuntimeError(
            f"LBH 3D upscaler returned incompatible shape {tuple(upscaled.shape)} "
            f"for H3 video input {tuple(video.shape)}"
        )
    return upscaled
