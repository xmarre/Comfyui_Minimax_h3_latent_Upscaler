from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import torch

NODES = Path(__file__).resolve().parents[1] / "nodes"
PACKAGE = "h3_upstream_sync_tests"

try:
    import folder_paths
except ImportError:
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.folder_names_and_paths = {}
    folder_paths.models_dir = "/tmp"

    def add_model_folder_path(name, path):
        folder_paths.folder_names_and_paths[name] = ([path], set())

    folder_paths.add_model_folder_path = add_model_folder_path
    folder_paths.get_folder_paths = lambda name: folder_paths.folder_names_and_paths[name][0]
    sys.modules["folder_paths"] = folder_paths

if PACKAGE not in sys.modules:
    pkg = types.ModuleType(PACKAGE)
    pkg.__path__ = [str(NODES)]
    sys.modules[PACKAGE] = pkg

lbh = importlib.import_module(f"{PACKAGE}.minimax_h3_latent_upscaler_3d")
refine = importlib.import_module(f"{PACKAGE}.minimax_h3_refine")


def test_alignment_grid_satisfies_requested_and_vae_grids():
    assert lbh._alignment_grid(32) == 32
    assert lbh._alignment_grid(24) == 48
    for align in (1, 16, 24, 32, 48, 64):
        grid = lbh._alignment_grid(align)
        assert grid % align == 0
        assert grid % lbh.VAE_DOWNSAMPLE == 0


def test_locked_aspect_ratio_aligns_both_axes_without_large_distortion():
    source_w, source_h = 77, 43
    width, height = lbh._aligned_pixel_size(
        1232,
        704,
        source_w,
        source_h,
        32,
        True,
    )
    assert width % 32 == 0
    assert height % 32 == 0
    assert width % lbh.VAE_DOWNSAMPLE == 0
    assert height % lbh.VAE_DOWNSAMPLE == 0
    source_ratio = source_w / source_h
    output_ratio = width / height
    assert abs(output_ratio / source_ratio - 1.0) < 0.02


def test_unlocked_alignment_rounds_axes_independently_on_common_grid():
    width, height = lbh._aligned_pixel_size(1271, 703, 80, 44, 32, False)
    assert (width, height) == (1280, 704)
    assert width % 32 == 0
    assert height % 32 == 0


def test_long_video_forward_is_not_silently_temporally_chunked():
    model = lbh.LatentResizer3D(
        in_channels=24,
        in_blocks=1,
        out_blocks=1,
        channels=32,
        dropout=0.0,
        attn=False,
        temporal_every=1,
        temporal_kernel=5,
    ).eval()
    calls = []
    hook = model.conv_in.register_forward_hook(lambda *_args: calls.append(1))
    try:
        with torch.inference_mode():
            output = model(
                torch.randn(1, 24, 20, 2, 2),
                scale=1.0,
                target_size=(20, 3, 3),
            )
    finally:
        hook.remove()

    assert output.shape == (1, 24, 20, 3, 3)
    assert len(calls) == 1
    assert not hasattr(model, "_forward_seg")


def test_cached_model_is_rehomed_after_optional_offload():
    calls = []

    class FakeModel:
        def to(self, device):
            calls.append(str(device))
            return self

    key = "cached.safetensors::cpu::fp16"
    fake = FakeModel()
    previous = lbh.MODEL_CACHE.get(key)
    lbh.MODEL_CACHE[key] = fake
    try:
        assert lbh.load_model("cached.safetensors", torch.device("cpu"), "fp16") is fake
    finally:
        if previous is None:
            lbh.MODEL_CACHE.pop(key, None)
        else:
            lbh.MODEL_CACHE[key] = previous

    assert calls == ["cpu"]


def test_integrated_refiner_offloads_only_selected_cached_model(monkeypatch):
    calls = []

    class FakeModel:
        def __init__(self, label):
            self.label = label

        def to(self, device):
            calls.append((self.label, str(device)))
            return self

    selected = FakeModel("selected")
    untouched = FakeModel("other")
    fake_lbh = types.SimpleNamespace(
        MODEL_CACHE={
            "selected.safetensors::cuda::bf16": selected,
            "other.safetensors::cuda::bf16": untouched,
        }
    )
    monkeypatch.setattr(refine, "_lbh_module", lambda: fake_lbh)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(("cuda", "empty")))

    assert refine._offload_cached_lbh_model("selected.safetensors", "cuda", "bf16")
    assert calls == [("selected", "cpu"), ("cuda", "empty")]


def test_integrated_refiner_does_not_offload_cpu_execution(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not refine._offload_cached_lbh_model("selected.safetensors", "cpu", "bf16")
