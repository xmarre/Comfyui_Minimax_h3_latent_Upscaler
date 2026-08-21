"""Integrated MiniMax H3 learned latent upscale + low-sigma refinement.

The learned spatial transform remains owned by the existing LBH 3D upscaler.
This node adds the missing sampling stage so its LATENT output is final,
decode-ready H3 AV state rather than an intermediate re-noised handoff.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from .minimax_h3_refine_support import (
    _h3_padded_spatial_size,
    _resolve_h3_sources,
    _scan_models,
    _upscale_modes,
    _validate_h3_av_samples,
    _wrap_h3_samples,
    _zero_audio_noise,
    prepare_h3_noise_mask,
    resize_h3_target_conditioning,
    run_lbh_video_upscale,
)

REFINE_STATE_API = 1
H3_REFINEMENT_REQUEST_KEY = "h3_refinement"
H3_REFINEMENT_API = 1


def _resolve_refine_contract(
    refine_state: Any,
    model: Any,
    positive: list | None,
) -> tuple[Any, list]:
    """Resolve exact Continuum refine state or explicit native H3 fallback inputs.

    A connected Continuum refine_state is authoritative and takes precedence over
    any stale/manual model + positive connections left on an upgraded workflow.
    Invalid refine_state data still fails closed rather than silently falling back.
    """
    if refine_state is not None:
        if not isinstance(refine_state, dict):
            raise TypeError("H3 Continuum refine_state must be a dictionary")
        if int(refine_state.get("api", -1)) != REFINE_STATE_API:
            raise ValueError(
                f"Unsupported H3 Continuum refine_state API {refine_state.get('api')!r}; "
                f"expected {REFINE_STATE_API}"
            )
        state_model = refine_state.get("model")
        state_positive = refine_state.get("positive")
        if state_model is None or state_positive is None:
            raise ValueError("H3 Continuum refine_state is missing model or positive conditioning")
        return state_model, state_positive

    if model is None or positive is None:
        raise ValueError(
            "MiniMax H3 refinement requires either H3 Continuum refine_state or explicit model + positive"
        )
    return model, positive


def _model_with_refinement_contract(model: Any) -> Any:
    """Clone MODEL and mark this invocation as a low-sigma H3 refinement pass.

    The source Continuum MODEL must stay untouched because its h3_continuum
    metadata describes sampler 1.  Sampler 2 needs a distinct contract: no
    inherited generation prefix and a sigma coordinate referenced to the full
    native H3 trajectory rather than to the first sigma of the short sub-run.
    """
    clone_fn = getattr(model, "clone", None)
    get_model_object = getattr(model, "get_model_object", None)
    if not callable(clone_fn) or not callable(get_model_object):
        raise TypeError("MiniMax H3 refinement expects a ComfyUI MODEL/ModelPatcher")

    refined_model = clone_fn()
    if refined_model is model:
        raise RuntimeError("MODEL.clone() returned the original object during H3 refinement")

    try:
        model_sampling = refined_model.get_model_object("model_sampling")
        sigma_max = model_sampling.sigma_max
    except (AttributeError, KeyError, RuntimeError) as exc:
        raise ValueError("MiniMax H3 refinement MODEL does not expose model_sampling.sigma_max") from exc

    if torch.is_tensor(sigma_max):
        if sigma_max.numel() != 1:
            raise ValueError("MiniMax H3 refinement sigma_max must be scalar")
        sigma_reference = float(sigma_max.detach().cpu().reshape(-1)[0].item())
    else:
        try:
            sigma_reference = float(sigma_max)
        except (TypeError, ValueError) as exc:
            raise ValueError("MiniMax H3 refinement sigma_max must be numeric") from exc
    if not math.isfinite(sigma_reference) or sigma_reference <= 0.0:
        raise ValueError(
            f"MiniMax H3 refinement sigma reference must be finite and positive; got {sigma_reference!r}"
        )

    model_options = getattr(refined_model, "model_options", None)
    if not isinstance(model_options, dict):
        raise TypeError("MiniMax H3 refinement MODEL.model_options must be a dictionary")
    copied_options = dict(model_options)
    transformer_options = copied_options.get("transformer_options")
    if transformer_options is None:
        transformer_options = {}
    elif not isinstance(transformer_options, dict):
        raise TypeError("MiniMax H3 refinement transformer_options must be a dictionary")
    else:
        transformer_options = dict(transformer_options)

    transformer_options[H3_REFINEMENT_REQUEST_KEY] = {
        "api": H3_REFINEMENT_API,
        "active": True,
        "min_actual_prefix_steps": 0,
        "sigma_reference": sigma_reference,
    }
    copied_options["transformer_options"] = transformer_options
    refined_model.model_options = copied_options
    return refined_model


def build_clean_h3_upscale(
    latent: dict,
    positive: list,
    *,
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
    lock_audio: bool,
    audio_latent: dict | None = None,
    negative: list | None = None,
) -> tuple[dict, list, list | None]:
    """Run only the learned video upscale and rebuild clean high-res H3 AV state."""
    source_video, source_audio, source_template, split_input = _resolve_h3_sources(
        latent, audio_latent
    )
    source_h, source_w = source_video.shape[-2:]
    upscaled_video = run_lbh_video_upscale(
        source_video,
        model_name,
        mode,
        scale,
        width,
        height,
        megapixels,
        align,
        keep_proportion,
        device,
        precision,
    )
    out_h, out_w = upscaled_video.shape[-2:]
    if out_h < source_h or out_w < source_w:
        raise RuntimeError("LBH 3D refinement path produced a spatial downscale")

    # H3 video/audio members may legally carry different dtypes. Move audio only
    # when needed; never cast it merely because the learned video upscaler uses a
    # different inference precision. lock_audio=True must preserve pass-1 audio.
    audio = source_audio
    if audio.device != upscaled_video.device:
        audio = audio.to(device=upscaled_video.device)
    samples = _wrap_h3_samples(upscaled_video, audio, source_template)
    clean = dict(latent)
    clean["samples"] = samples

    mask = prepare_h3_noise_mask(
        latent,
        audio_latent,
        upscaled_video,
        audio,
        samples,
        split_input=split_input,
        lock_audio=bool(lock_audio),
    )
    if mask is None:
        clean.pop("noise_mask", None)
    else:
        clean["noise_mask"] = mask

    target_h, target_w = _h3_padded_spatial_size(upscaled_video)
    positive_out = resize_h3_target_conditioning(positive, target_h, target_w)
    negative_out = resize_h3_target_conditioning(negative, target_h, target_w)
    return clean, positive_out, negative_out


def _make_guider(model: Any, positive: list, negative: list | None, cfg: float):
    try:
        import comfy.samplers
    except Exception as exc:  # pragma: no cover - real ComfyUI runtime only
        raise RuntimeError(f"ComfyUI sampler API unavailable: {exc}") from exc

    if negative is None:
        class _BasicGuider(comfy.samplers.CFGGuider):
            def set_positive(self, value):
                self.inner_set_conds({"positive": value})

        guider = _BasicGuider(model)
        guider.set_positive(positive)
        return guider

    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(float(cfg))
    return guider


def _without_downscale_metadata(latent: dict) -> dict:
    """Return a LATENT copy without source-grid hints consumed by sampler nodes."""
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    return out


def run_h3_refinement(
    clean: dict,
    *,
    model: Any,
    positive: list,
    negative: list | None,
    noise: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    cfg: float,
    lock_audio: bool,
) -> dict:
    """Run the actual second H3 sampling pass on the learned high-res latent."""
    try:
        import comfy.model_management
        import comfy.sample
        import comfy.utils
        import latent_preview
    except Exception as exc:  # pragma: no cover - real ComfyUI runtime only
        raise RuntimeError(f"ComfyUI sampling runtime unavailable: {exc}") from exc

    if not torch.is_tensor(sigmas) or sigmas.ndim != 1:
        raise TypeError("sigmas must be a one-dimensional torch.Tensor")
    if sigmas.numel() == 0:
        return _without_downscale_metadata(clean)
    if sigmas.numel() < 2:
        raise ValueError("refinement sigmas must contain at least a start and end value")
    if not bool(torch.isfinite(sigmas).all().item()):
        raise ValueError("refinement sigmas must be finite")
    sigma_start = float(sigmas[0].detach().cpu())
    if not math.isfinite(sigma_start) or sigma_start < 0.0 or sigma_start >= 1.0:
        raise ValueError(
            "MiniMax H3 learned-latent refinement requires 0 <= sigmas[0] < 1; "
            f"got {sigma_start!r}. A full-noise sigma discards the learned upscaled latent; "
            "use a partial-denoise second-pass schedule."
        )

    latent = clean.copy()
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent_image,
        latent.get("downscale_ratio_spacial"),
        latent.get("downscale_ratio_temporal"),
    )
    latent["samples"] = latent_image
    clean_video, clean_audio = _validate_h3_av_samples(latent_image)

    # Complete SamplerCustomAdvanced-equivalent path: fresh enlarged-grid noise
    # plus the clean learned latent go directly to CFGGuider.sample. ComfyUI's
    # sampler performs model_sampling.noise_scaling itself; there is no pre-noised
    # handoff and no external DisableNoise stage.
    generated_noise = noise.generate_noise(latent)
    noise_video, noise_audio = _validate_h3_av_samples(generated_noise)
    if noise_video.shape != clean_video.shape or noise_audio.shape != clean_audio.shape:
        raise ValueError(
            "Generated refinement noise must match the enlarged H3 AV latent exactly; "
            f"video {tuple(noise_video.shape)} vs {tuple(clean_video.shape)}, "
            f"audio {tuple(noise_audio.shape)} vs {tuple(clean_audio.shape)}"
        )
    if lock_audio:
        generated_noise = _zero_audio_noise(generated_noise, latent_image)

    guider = _make_guider(model, positive, negative, cfg)
    noise_mask = latent.get("noise_mask")
    x0_output: dict[str, Any] = {}
    callback = latent_preview.prepare_callback(
        guider.model_patcher,
        int(sigmas.shape[-1]) - 1,
        x0_output,
    )
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    seed = int(getattr(noise, "seed", 0))
    samples = guider.sample(
        generated_noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    samples = samples.to(comfy.model_management.intermediate_device())
    sampled_video, sampled_audio = _validate_h3_av_samples(samples)
    if lock_audio:
        sampled_audio = clean_audio.to(device=sampled_audio.device)
        samples = _wrap_h3_samples(sampled_video, sampled_audio, samples)

    out = _without_downscale_metadata(latent)
    out["samples"] = samples
    return out


class MinimaxH3LatentUpscaler3DRefine:
    """LBH learned 3D upscale plus the complete MiniMax H3 refinement pass."""

    @classmethod
    def INPUT_TYPES(cls):
        modes = _upscale_modes()
        return {
            "required": {
                "latent": ("LATENT",),
                "noise": ("NOISE",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "model_name": (_scan_models(),),
                "mode": (list(modes), {"default": modes[0]}),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05}),
                "width": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 704, "min": 64, "max": 4096, "step": 8}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                "align": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1}),
                "keep_proportion": ("BOOLEAN", {"default": True}),
                "lock_audio": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "On preserves pass-1 audio exactly and masks it out of refinement. "
                            "Off lets sampler 2 refine/remix audio too."
                        ),
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "advanced": True,
                        "tooltip": "Used only when optional negative conditioning is connected on the native fallback path.",
                    },
                ),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "precision": (["fp32", "fp16", "bf16"], {"default": "fp16"}),
            },
            "optional": {
                "audio_latent": (
                    "LATENT",
                    {
                        "tooltip": (
                            "For H3 Continuum, connect the matching audio_latents output. "
                            "Leave disconnected for native joint AV LATENT input."
                        )
                    },
                ),
                "refine_state": (
                    "H3_CONTINUUM_REFINE_STATE",
                    {
                        "tooltip": (
                            "Preferred H3 Continuum path: exact per-chunk fresh MODEL wrapper + "
                            "positive CONDITIONING captured by Continuum. Takes precedence over "
                            "manual model/positive/negative fallback connections."
                        )
                    },
                ),
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "Native/non-Continuum fallback. Ignored when refine_state is connected."
                        )
                    },
                ),
                "positive": (
                    "CONDITIONING",
                    {
                        "tooltip": (
                            "Native/non-Continuum fallback. Ignored when refine_state is connected."
                        )
                    },
                ),
                "negative": (
                    "CONDITIONING",
                    {
                        "tooltip": (
                            "Optional CFG compatibility for native fallback only. Ignored when refine_state is connected."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "upscale_and_refine"
    CATEGORY = "video/MinimaxH3"
    DESCRIPTION = (
        "Learned LBH 3D upscale followed by the actual low-sigma MiniMax H3 refinement pass. "
        "For H3 Continuum, connect parallel video_latents + audio_latents + refine_state. "
        "The LATENT output is final/decode-ready; no external BasicGuider, DisableNoise, "
        "or SamplerCustomAdvanced is required."
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
        refine_model, refine_positive = _resolve_refine_contract(
            refine_state, model, positive
        )
        # Continuum refine_state is an exact positive-only sampler contract. If an
        # upgraded workflow still has old manual conditioning wires attached, do
        # not let a stale negative silently change the handoff into CFG.
        refine_negative = None if refine_state is not None else negative
        clean, positive_out, negative_out = build_clean_h3_upscale(
            latent,
            refine_positive,
            model_name=model_name,
            mode=mode,
            scale=scale,
            width=width,
            height=height,
            megapixels=megapixels,
            align=align,
            keep_proportion=keep_proportion,
            device=device,
            precision=precision,
            lock_audio=bool(lock_audio),
            audio_latent=audio_latent,
            negative=refine_negative,
        )
        refine_model = _model_with_refinement_contract(refine_model)
        refined = run_h3_refinement(
            clean,
            model=refine_model,
            positive=positive_out,
            negative=negative_out,
            noise=noise,
            sampler=sampler,
            sigmas=sigmas,
            cfg=float(cfg),
            lock_audio=bool(lock_audio),
        )
        return (refined,)


# Retain the development-branch class key so workflows created while this PR was
# under review do not become missing-node graphs. The public display name and
# behavior are the completed integrated refinement node.
NODE_CLASS_MAPPINGS = {
    "MinimaxH3LatentUpscaler3DRefineHandoff": MinimaxH3LatentUpscaler3DRefine,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MinimaxH3LatentUpscaler3DRefineHandoff": "MiniMax H3 Latent Upscaler + Refine (3D)",
}
