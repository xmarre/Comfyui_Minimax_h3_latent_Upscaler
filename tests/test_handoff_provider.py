from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

NODES = Path(__file__).resolve().parents[1] / "nodes"
PACKAGE = "h3_handoff_provider_tests"
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(NODES)]
    sys.modules[PACKAGE] = package

provider_module = importlib.import_module(f"{PACKAGE}.minimax_h3_handoff_provider")


def test_provider_contract_is_versioned_immutable_and_configuration_only():
    provider = provider_module.H3LatentUpscalerProvider(
        model_name="h3-upscaler.safetensors",
        device="cpu",
        precision="bf16",
        offload_after_upscale=True,
    )

    assert provider.api_version == 1
    assert provider.kind == "minimax_h3_learned_latent_upscaler"
    assert provider.model_name == "h3-upscaler.safetensors"
    assert provider.inference_device == "cpu"
    with pytest.raises(AttributeError):
        provider.precision = "fp32"


def test_provider_configuration_defaults_match_progressive_workflow():
    provider = provider_module.H3LatentUpscalerProvider(model_name="h3-upscaler.safetensors")

    assert provider.device == "cuda"
    assert provider.precision == "bf16"
    assert provider.offload_after_upscale is False


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"model_name": "(place models here)"}, ValueError, "checkpoint"),
        ({"model_name": "m", "device": "mps"}, ValueError, "cuda or cpu"),
        ({"model_name": "m", "precision": "int8"}, ValueError, "fp32, fp16, or bf16"),
        ({"model_name": "m", "offload_after_upscale": 1}, TypeError, "must be boolean"),
    ],
)
def test_provider_configuration_fails_closed(kwargs, error, message):
    with pytest.raises(error, match=message):
        provider_module.H3LatentUpscalerProvider(**kwargs)


def test_provider_delegates_exact_target_once_without_audio_or_input_mutation(
    monkeypatch,
):
    calls = []

    def exact(video, **kwargs):
        calls.append((video.clone(), kwargs))
        return torch.full(
            (2, 24, 3, 8, 10), 7.0, dtype=video.dtype, device=video.device
        )

    monkeypatch.setattr(
        provider_module,
        "_lbh_module",
        lambda: types.SimpleNamespace(upscale_clean_video_exact=exact),
    )
    provider = provider_module.H3LatentUpscalerProvider(
        model_name="m.safetensors",
        device="cpu",
        precision="fp32",
    )
    video = torch.randn(2, 24, 3, 4, 6)
    before = video.clone()

    output = provider.upscale_clean_video(video, target_h=8, target_w=10)

    assert output.shape == (2, 24, 3, 8, 10)
    assert torch.equal(video, before)
    assert len(calls) == 1
    assert torch.equal(calls[0][0], before)
    assert calls[0][1] == {
        "model_name": "m.safetensors",
        "target_h": 8,
        "target_w": 10,
        "device": "cpu",
        "precision": "fp32",
        "offload_after_upscale": False,
    }


def test_provider_never_silently_falls_back_from_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    provider = provider_module.H3LatentUpscalerProvider(model_name="m.safetensors")

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        provider.upscale_clean_video(
            torch.randn(1, 24, 1, 2, 2), target_h=4, target_w=4
        )


def test_provider_node_exposes_stable_side_input_type(monkeypatch):
    preferred = provider_module.PREFERRED_H3_LATENT_UPSCALER_MODEL
    monkeypatch.setattr(
        provider_module,
        "_lbh_module",
        lambda: types.SimpleNamespace(scan_models=lambda: ["m.safetensors", preferred]),
    )
    node = provider_module.MinimaxH3LatentUpscaler3DProvider
    schema = node.INPUT_TYPES()

    assert list(schema["required"]) == [
        "model_name",
        "device",
        "precision",
        "offload_after_upscale",
    ]
    assert schema["required"]["model_name"][1]["default"] == preferred
    assert schema["required"]["device"][1]["default"] == "cuda"
    assert schema["required"]["precision"][1]["default"] == "bf16"
    assert schema["required"]["offload_after_upscale"][1]["default"] is False
    assert node.RETURN_TYPES == ("H3_LATENT_UPSCALER",)
    built = node().build("m.safetensors", "cpu", "fp32", False)[0]
    assert isinstance(built, provider_module.H3LatentUpscalerProvider)


def test_provider_node_does_not_invent_missing_preferred_checkpoint(monkeypatch):
    monkeypatch.setattr(
        provider_module,
        "_lbh_module",
        lambda: types.SimpleNamespace(scan_models=lambda: ["m.safetensors"]),
    )

    schema = provider_module.MinimaxH3LatentUpscaler3DProvider.INPUT_TYPES()

    assert schema["required"]["model_name"] == (["m.safetensors"],)
