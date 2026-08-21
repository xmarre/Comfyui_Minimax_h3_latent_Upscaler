from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from enum import Enum
from pathlib import Path

import pytest
import torch

NODES = Path(__file__).resolve().parents[1] / "nodes"
PACKAGE = "h3_refine_support_tests"
if PACKAGE not in sys.modules:
    pkg = types.ModuleType(PACKAGE)
    pkg.__path__ = [str(NODES)]
    sys.modules[PACKAGE] = pkg

support = importlib.import_module(f"{PACKAGE}.minimax_h3_refine_support")
REFINE_SPEC = importlib.util.spec_from_file_location(
    f"{PACKAGE}.refine", NODES / "minimax_h3_refine.py"
)
assert REFINE_SPEC and REFINE_SPEC.loader
refine = importlib.util.module_from_spec(REFINE_SPEC)
sys.modules[REFINE_SPEC.name] = refine
REFINE_SPEC.loader.exec_module(refine)


class FakeNested:
    is_nested = True

    def __init__(self, members):
        self.members = list(members)

    def unbind(self):
        return tuple(self.members)

    def to(self, device):
        return FakeNested([member.to(device) for member in self.members])


def make_joint(h=4, w=6, t=3, audio_t=9, value=2.0, audio_dtype=torch.float32):
    return {
        "samples": FakeNested(
            [
                torch.full((1, 24, t, h, w), value),
                torch.full((1, 32, 2, audio_t), value + 3.0, dtype=audio_dtype),
            ]
        )
    }


def make_split(h=4, w=6, t=3, audio_t=9, value=2.0):
    return (
        {"samples": torch.full((1, 24, t, h, w), value)},
        {"samples": torch.full((1, 32, 2, audio_t), value + 3.0)},
    )


def make_cond(*, keyframe=False, refs=False, container=list, tail=()):
    meta = {"marker": "keep"}
    if keyframe:
        meta["minimax_keyframes"] = [
            {
                "resolved_frame_index": 0,
                "latent": torch.ones(1, 24, 1, 4, 6),
                "audio_latent": torch.ones(1, 32, 2, 3),
            }
        ]
    if refs:
        meta["minimax_refs"] = [
            {
                "kind": "image",
                "latent_h": 4,
                "latent_w": 6,
                "latent": torch.ones(1, 24, 1, 4, 6),
            }
        ]
    entry = [torch.ones(1, 2, 3), meta, *tail]
    return [tuple(entry) if container is tuple else entry]


def fake_upscale(monkeypatch, *, out_h=8, out_w=12, dtype=None):
    def impl(video, *args, **kwargs):
        del args, kwargs
        return torch.full(
            (video.shape[0], video.shape[1], video.shape[2], out_h, out_w),
            7.0,
            dtype=dtype or video.dtype,
            device=video.device,
        )

    monkeypatch.setattr(refine, "run_lbh_video_upscale", impl)


def build_clean(
    monkeypatch,
    *,
    latent=None,
    audio_latent=None,
    positive=None,
    negative=None,
    lock_audio=True,
    out_h=8,
    out_w=12,
    upscale_dtype=None,
):
    fake_upscale(monkeypatch, out_h=out_h, out_w=out_w, dtype=upscale_dtype)
    return refine.build_clean_h3_upscale(
        latent or make_joint(),
        positive or make_cond(),
        model_name="m.safetensors",
        mode="scale by multiplier",
        scale=2.0,
        width=1280,
        height=704,
        megapixels=1.0,
        align=32,
        keep_proportion=True,
        device="cuda",
        precision="bf16",
        lock_audio=lock_audio,
        audio_latent=audio_latent,
        negative=negative,
    )


def test_joint_av_resolves():
    latent = make_joint()
    video, audio, template, split = support._resolve_h3_sources(latent, None)
    assert video.shape == (1, 24, 3, 4, 6)
    assert audio.shape == (1, 32, 2, 9)
    assert template is latent["samples"]
    assert not split


def test_continuum_split_av_resolves():
    video_latent, audio_latent = make_split()
    video, audio, template, split = support._resolve_h3_sources(video_latent, audio_latent)
    assert video.shape == (1, 24, 3, 4, 6)
    assert audio.shape == (1, 32, 2, 9)
    assert template is None
    assert split


def test_plain_video_without_audio_has_actionable_error():
    video_latent, _ = make_split()
    with pytest.raises(ValueError, match="audio_latents output"):
        support._resolve_h3_sources(video_latent, None)


def test_joint_plus_audio_is_rejected():
    _, audio_latent = make_split()
    with pytest.raises(ValueError, match="left disconnected"):
        support._resolve_h3_sources(make_joint(), audio_latent)


def test_invalid_split_audio_is_rejected():
    video_latent, _ = make_split()
    bad = {"samples": torch.zeros(1, 32, 1, 9)}
    with pytest.raises(ValueError, match="Bx32x2xT"):
        support._resolve_h3_sources(video_latent, bad)


@pytest.mark.parametrize("container", [list, tuple])
def test_conditioning_tail_and_container_type_are_preserved(container):
    cond = make_cond(keyframe=True, container=container, tail=("tail", 42))
    out = support.resize_h3_target_conditioning(cond, 8, 12)
    assert isinstance(out[0], container)
    assert list(out[0][2:]) == ["tail", 42]


def test_keyframe_uses_target_grid_but_reference_grid_is_untouched():
    cond = make_cond(keyframe=True, refs=True)
    refs_before = cond[0][1]["minimax_refs"]
    out = support.resize_h3_target_conditioning(cond, 10, 14)
    assert out[0][1]["minimax_keyframes"][0]["latent"].shape[-2:] == (10, 14)
    assert out[0][1]["minimax_refs"] is refs_before
    assert refs_before[0]["latent"].shape[-2:] == (4, 6)


def test_split_masks_are_rejoined_and_video_mask_is_resized():
    video_latent, audio_latent = make_split()
    video_latent["noise_mask"] = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    audio_latent["noise_mask"] = torch.full((1, 1, 2, 9), 0.25)
    video = torch.zeros(1, 24, 3, 7, 11)
    audio = audio_latent["samples"]
    mask = support.prepare_h3_noise_mask(
        video_latent,
        audio_latent,
        video,
        audio,
        FakeNested([video, audio]),
        split_input=True,
        lock_audio=False,
    )
    video_mask, audio_mask = mask.unbind()
    assert video_mask.shape[-2:] == (7, 11)
    assert torch.all(audio_mask == 0.25)


def test_lock_audio_forces_zero_audio_mask():
    video_latent, audio_latent = make_split()
    video = torch.zeros(1, 24, 3, 8, 12)
    audio = audio_latent["samples"]
    mask = support.prepare_h3_noise_mask(
        video_latent,
        audio_latent,
        video,
        audio,
        FakeNested([video, audio]),
        split_input=True,
        lock_audio=True,
    )
    video_mask, audio_mask = mask.unbind()
    assert torch.all(video_mask == 1)
    assert torch.count_nonzero(audio_mask) == 0


def test_without_masks_and_unlocked_audio_keeps_mask_absent():
    video_latent, audio_latent = make_split()
    video = torch.zeros(1, 24, 3, 8, 12)
    audio = audio_latent["samples"]
    assert (
        support.prepare_h3_noise_mask(
            video_latent,
            audio_latent,
            video,
            audio,
            FakeNested([video, audio]),
            split_input=True,
            lock_audio=False,
        )
        is None
    )


def test_clean_joint_upscale_changes_only_video_spatial_grid(monkeypatch):
    latent = make_joint()
    original_audio = latent["samples"].unbind()[1].clone()
    out, _, _ = build_clean(monkeypatch, latent=latent, lock_audio=True)
    video, audio = out["samples"].unbind()
    assert video.shape[-2:] == (8, 12)
    assert torch.equal(audio, original_audio)


def test_clean_continuum_split_upscale_rejoins_exact_audio(monkeypatch):
    video_latent, audio_latent = make_split()
    original_audio = audio_latent["samples"].clone()
    out, _, _ = build_clean(
        monkeypatch,
        latent=video_latent,
        audio_latent=audio_latent,
        lock_audio=True,
    )
    video, audio = out["samples"].unbind()
    assert video.shape[-2:] == (8, 12)
    assert torch.equal(audio, original_audio)


def test_odd_learned_output_is_not_physically_padded(monkeypatch):
    out, _, _ = build_clean(monkeypatch, out_h=9, out_w=13)
    assert out["samples"].unbind()[0].shape[-2:] == (9, 13)


def test_clean_upscale_resizes_keyframe_to_internal_even_grid(monkeypatch):
    _, positive, _ = build_clean(
        monkeypatch,
        positive=make_cond(keyframe=True),
        out_h=9,
        out_w=13,
    )
    assert positive[0][1]["minimax_keyframes"][0]["latent"].shape[-2:] == (10, 14)


def test_clean_upscale_does_not_resize_references(monkeypatch):
    positive = make_cond(refs=True)
    refs_before = positive[0][1]["minimax_refs"]
    _, out, _ = build_clean(monkeypatch, positive=positive, out_h=10, out_w=16)
    assert out[0][1]["minimax_refs"] is refs_before


def test_negative_is_optional_and_resized_when_connected(monkeypatch):
    _, _, negative_none = build_clean(monkeypatch, negative=None)
    assert negative_none is None

    _, _, negative = build_clean(
        monkeypatch,
        negative=make_cond(keyframe=True),
        out_h=9,
        out_w=13,
    )
    assert negative[0][1]["minimax_keyframes"][0]["latent"].shape[-2:] == (10, 14)


def test_clean_split_masks_remain_aligned_after_upscale(monkeypatch):
    video_latent, audio_latent = make_split()
    video_latent["noise_mask"] = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    audio_latent["noise_mask"] = torch.full((1, 1, 2, 9), 0.25)
    out, _, _ = build_clean(
        monkeypatch,
        latent=video_latent,
        audio_latent=audio_latent,
        lock_audio=False,
        out_h=7,
        out_w=11,
    )
    video_mask, audio_mask = out["noise_mask"].unbind()
    assert video_mask.shape[-2:] == (7, 11)
    assert torch.all(audio_mask == 0.25)


def test_clean_upscale_preserves_audio_dtype(monkeypatch):
    latent = make_joint(audio_dtype=torch.float64)
    out, _, _ = build_clean(
        monkeypatch,
        latent=latent,
        lock_audio=True,
        upscale_dtype=torch.float32,
    )
    video, audio = out["samples"].unbind()
    assert video.dtype == torch.float32
    assert audio.dtype == torch.float64


def test_lbh_delegate_forwards_existing_node_controls(monkeypatch):
    calls = {}

    class Mode(str, Enum):
        SCALE = "scale by multiplier"

    class Node:
        @staticmethod
        def execute(latent, model_name, mode_config, align, keep_proportion, device, precision):
            calls.update(
                model_name=model_name,
                mode_config=mode_config,
                align=align,
                keep_proportion=keep_proportion,
                device=device,
                precision=precision,
            )
            return ({"samples": torch.zeros(1, 24, 3, 8, 12)},)

    fake = types.SimpleNamespace(UpscaleMode=Mode, MinimaxH3LatentUpscaler3D=Node)
    monkeypatch.setattr(support, "_lbh_module", lambda: fake)
    out = support.run_lbh_video_upscale(
        torch.zeros(1, 24, 3, 4, 6),
        "m",
        "scale by multiplier",
        2.0,
        1280,
        704,
        0.7,
        32,
        False,
        "cuda",
        "bf16",
    )
    assert out.shape == (1, 24, 3, 8, 12)
    assert calls["model_name"] == "m"
    assert calls["align"] == 32
    assert calls["keep_proportion"] is False
    assert calls["precision"] == "bf16"


def test_lbh_delegate_rejects_temporal_shape_change(monkeypatch):
    class Mode(str, Enum):
        SCALE = "scale by multiplier"

    class Node:
        @staticmethod
        def execute(*args, **kwargs):
            return ({"samples": torch.zeros(1, 24, 4, 8, 12)},)

    monkeypatch.setattr(
        support,
        "_lbh_module",
        lambda: types.SimpleNamespace(UpscaleMode=Mode, MinimaxH3LatentUpscaler3D=Node),
    )
    with pytest.raises(RuntimeError, match="incompatible shape"):
        support.run_lbh_video_upscale(
            torch.zeros(1, 24, 3, 4, 6),
            "m",
            "scale by multiplier",
            2.0,
            1,
            1,
            1.0,
            32,
            True,
            "cpu",
            "bf16",
        )
