from __future__ import annotations

import copy
import importlib.util
import sys
import types
from enum import Enum
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes" / "minimax_h3_refine.py"
PACKAGE = "h3_refine_node_tests"
if PACKAGE not in sys.modules:
    pkg = types.ModuleType(PACKAGE)
    pkg.__path__ = [str(MODULE_PATH.parent)]
    sys.modules[PACKAGE] = pkg
SPEC = importlib.util.spec_from_file_location(f"{PACKAGE}.refine", MODULE_PATH)
assert SPEC and SPEC.loader
refine = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = refine
SPEC.loader.exec_module(refine)


class FakeNested:
    is_nested = True

    def __init__(self, members):
        self.members = list(members)

    def unbind(self):
        return tuple(self.members)

    def to(self, device):
        return FakeNested([member.to(device) for member in self.members])


class FakeNoise:
    seed = 123

    def __init__(self):
        self.calls = 0
        self.shapes = None

    def generate_noise(self, latent):
        self.calls += 1
        video, audio = latent["samples"].unbind()
        self.shapes = (tuple(video.shape), tuple(audio.shape))
        return FakeNested([torch.ones_like(video), torch.ones_like(audio)])


class FakeModelSampling:
    def __init__(self, sigma_max=1.0):
        self.sigma_max = torch.tensor(float(sigma_max))


class FakeModelPatcher:
    def __init__(self, model_options=None, model_sampling=None):
        self.model_options = copy.deepcopy(model_options or {})
        self.model_sampling = model_sampling or FakeModelSampling()

    def clone(self):
        return FakeModelPatcher(self.model_options, self.model_sampling)

    def get_model_object(self, name):
        if name != "model_sampling":
            raise KeyError(name)
        return self.model_sampling


class FakeCFGGuider:
    last = None

    def __init__(self, model):
        self.model = model
        self.model_patcher = model
        self.conds = None
        self.cfg = 1.0
        FakeCFGGuider.last = self

    def inner_set_conds(self, conds):
        self.conds = conds

    def set_conds(self, positive, negative):
        self.conds = {"positive": positive, "negative": negative}

    def set_cfg(self, cfg):
        self.cfg = float(cfg)

    def sample(
        self,
        noise,
        latent_image,
        sampler,
        sigmas,
        *,
        denoise_mask,
        callback,
        disable_pbar,
        seed,
    ):
        del sampler, sigmas, denoise_mask, callback, disable_pbar, seed
        video, audio = latent_image.unbind()
        noise_video, noise_audio = noise.unbind()
        return FakeNested([video + noise_video, audio + noise_audio])


def install_fake_comfy(monkeypatch):
    comfy = types.ModuleType("comfy")
    comfy.samplers = types.ModuleType("comfy.samplers")
    comfy.samplers.CFGGuider = FakeCFGGuider
    comfy.sample = types.ModuleType("comfy.sample")
    comfy.sample.fix_empty_latent_channels = lambda _model, latent, *_ratios: latent
    comfy.model_management = types.ModuleType("comfy.model_management")
    comfy.model_management.intermediate_device = lambda: torch.device("cpu")
    comfy.utils = types.ModuleType("comfy.utils")
    comfy.utils.PROGRESS_BAR_ENABLED = False
    latent_preview = types.ModuleType("latent_preview")
    latent_preview.prepare_callback = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.samplers", comfy.samplers)
    monkeypatch.setitem(sys.modules, "comfy.sample", comfy.sample)
    monkeypatch.setitem(sys.modules, "comfy.model_management", comfy.model_management)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy.utils)
    monkeypatch.setitem(sys.modules, "latent_preview", latent_preview)


def make_joint():
    return {
        "samples": FakeNested(
            [
                torch.full((1, 24, 3, 4, 6), 2.0),
                torch.full((1, 32, 2, 9), 5.0),
            ]
        )
    }


def test_schema_is_complete_sampler_not_handoff(monkeypatch):
    class Mode(str, Enum):
        SCALE = "scale by multiplier"

    fake = types.SimpleNamespace(UpscaleMode=Mode, scan_models=lambda: ["m.safetensors"])
    monkeypatch.setattr(refine, "_scan_models", lambda: fake.scan_models())
    monkeypatch.setattr(refine, "_upscale_modes", lambda: tuple(member.value for member in Mode))
    schema = refine.MinimaxH3LatentUpscaler3DRefine.INPUT_TYPES()
    required = schema["required"]
    optional = schema["optional"]
    assert "noise" in required
    assert "sampler" in required
    assert "sigmas" in required
    assert "refine_state" in optional
    assert "model" in optional
    assert "positive" in optional
    assert "negative" in optional
    assert "guider" not in required and "guider" not in optional
    assert refine.MinimaxH3LatentUpscaler3DRefine.RETURN_TYPES == ("LATENT",)
    assert refine.MinimaxH3LatentUpscaler3DRefine.RETURN_NAMES == ("latent",)


def test_refine_state_replaces_manual_model_and_positive():
    model = object()
    positive = [[torch.ones(1), {}]]
    resolved_model, resolved_positive = refine._resolve_refine_contract(
        {"api": 1, "model": model, "positive": positive},
        None,
        None,
    )
    assert resolved_model is model
    assert resolved_positive is positive


def test_refine_state_takes_precedence_over_stale_manual_inputs():
    state_model = object()
    state_positive = [[torch.ones(1), {"source": "continuum"}]]
    manual_model = object()
    manual_positive = [[torch.zeros(1), {"source": "manual"}]]

    resolved_model, resolved_positive = refine._resolve_refine_contract(
        {"api": 1, "model": state_model, "positive": state_positive},
        manual_model,
        manual_positive,
    )

    assert resolved_model is state_model
    assert resolved_positive is state_positive


def test_invalid_refine_state_does_not_fall_back_to_manual_inputs():
    with pytest.raises(ValueError, match="Unsupported H3 Continuum refine_state API"):
        refine._resolve_refine_contract(
            {"api": 999, "model": object(), "positive": [[torch.ones(1), {}]]},
            object(),
            [[torch.zeros(1), {}]],
        )


def test_refinement_contract_clones_model_and_preserves_continuum_source():
    source = FakeModelPatcher(
        {
            "transformer_options": {
                "h3_continuum": {
                    "api": 1,
                    "active": True,
                    "min_actual_prefix_steps": 2,
                },
                "unrelated": {"value": 7},
            }
        }
    )
    before = copy.deepcopy(source.model_options)

    marked = refine._model_with_refinement_contract(source)

    assert marked is not source
    assert source.model_options == before
    request = marked.model_options["transformer_options"][refine.H3_REFINEMENT_REQUEST_KEY]
    assert request == {
        "api": 1,
        "active": True,
        "min_actual_prefix_steps": 0,
        "sigma_reference": 1.0,
    }
    assert marked.model_options["transformer_options"]["h3_continuum"]["min_actual_prefix_steps"] == 2
    assert marked.model_options["transformer_options"]["unrelated"] == {"value": 7}


def test_refinement_contract_rejects_invalid_sigma_reference():
    with pytest.raises(ValueError, match="finite and positive"):
        refine._model_with_refinement_contract(
            FakeModelPatcher(model_sampling=FakeModelSampling(float("nan")))
        )


def test_integrated_refinement_actually_samples_and_restores_locked_audio(monkeypatch):
    install_fake_comfy(monkeypatch)
    latent = make_joint()
    clean_audio = latent["samples"].unbind()[1].clone()
    noise = FakeNoise()
    positive = [[torch.ones(1), {}]]

    out = refine.run_h3_refinement(
        latent,
        model=object(),
        positive=positive,
        negative=None,
        noise=noise,
        sampler=object(),
        sigmas=torch.tensor([0.25, 0.1, 0.0]),
        cfg=1.0,
        lock_audio=True,
    )

    video, audio = out["samples"].unbind()
    assert noise.calls == 1
    assert torch.all(video == 3.0)
    assert torch.equal(audio, clean_audio)
    assert FakeCFGGuider.last.conds == {"positive": positive}


def test_complete_node_generates_noise_on_enlarged_grid(monkeypatch):
    install_fake_comfy(monkeypatch)

    def fake_upscale(video, *_args, **_kwargs):
        return torch.full(
            (video.shape[0], video.shape[1], video.shape[2], 8, 12),
            7.0,
            dtype=video.dtype,
            device=video.device,
        )

    monkeypatch.setattr(refine, "run_lbh_video_upscale", fake_upscale)
    latent = make_joint()
    noise = FakeNoise()
    model = FakeModelPatcher(
        {
            "transformer_options": {
                "h3_continuum": {
                    "api": 1,
                    "active": True,
                    "min_actual_prefix_steps": 2,
                }
            }
        }
    )
    positive = [[torch.ones(1), {}]]
    stale_model = object()
    stale_positive = [[torch.zeros(1), {"source": "stale"}]]
    stale_negative = [[torch.zeros(1), {"source": "stale-negative"}]]
    (out,) = refine.MinimaxH3LatentUpscaler3DRefine().upscale_and_refine(
        latent,
        noise,
        object(),
        torch.tensor([0.25, 0.0]),
        "m.safetensors",
        "scale by multiplier",
        2.0,
        1280,
        704,
        1.0,
        32,
        True,
        True,
        7.5,
        "cpu",
        "fp32",
        refine_state={"api": 1, "model": model, "positive": positive},
        model=stale_model,
        positive=stale_positive,
        negative=stale_negative,
    )

    video, audio = out["samples"].unbind()
    assert noise.calls == 1
    assert noise.shapes == ((1, 24, 3, 8, 12), (1, 32, 2, 9))
    assert video.shape == (1, 24, 3, 8, 12)
    assert torch.all(video == 8.0)
    assert torch.all(audio == 5.0)
    assert FakeCFGGuider.last.model is not model
    marked_options = FakeCFGGuider.last.model.model_options["transformer_options"]
    assert marked_options["h3_continuum"]["min_actual_prefix_steps"] == 2
    assert marked_options[refine.H3_REFINEMENT_REQUEST_KEY]["min_actual_prefix_steps"] == 0
    assert marked_options[refine.H3_REFINEMENT_REQUEST_KEY]["sigma_reference"] == 1.0
    assert refine.H3_REFINEMENT_REQUEST_KEY not in model.model_options["transformer_options"]
    assert FakeCFGGuider.last.conds == {"positive": positive}
    assert FakeCFGGuider.last.cfg == 1.0


def test_optional_negative_uses_cfg_guider(monkeypatch):
    install_fake_comfy(monkeypatch)
    latent = make_joint()
    positive = [[torch.ones(1), {"p": 1}]]
    negative = [[torch.zeros(1), {"n": 1}]]
    refine.run_h3_refinement(
        latent,
        model=object(),
        positive=positive,
        negative=negative,
        noise=FakeNoise(),
        sampler=object(),
        sigmas=torch.tensor([0.2, 0.0]),
        cfg=2.5,
        lock_audio=False,
    )
    assert FakeCFGGuider.last.conds == {"positive": positive, "negative": negative}
    assert FakeCFGGuider.last.cfg == 2.5


@pytest.mark.parametrize(
    ("start", "error"),
    [
        (1.0, "partial-denoise"),
        (1.1, "partial-denoise"),
        (-0.01, "partial-denoise"),
        (float("inf"), "finite"),
        (float("nan"), "finite"),
    ],
)
def test_invalid_refinement_start_rejected_before_noise(monkeypatch, start, error):
    install_fake_comfy(monkeypatch)
    noise = FakeNoise()
    with pytest.raises(ValueError, match=error):
        refine.run_h3_refinement(
            make_joint(),
            model=object(),
            positive=[[torch.ones(1), {}]],
            negative=None,
            noise=noise,
            sampler=object(),
            sigmas=torch.tensor([start, 0.0]),
            cfg=1.0,
            lock_audio=True,
        )
    assert noise.calls == 0


def test_empty_sigmas_returns_clean_without_sampling_and_drops_source_grid_hints(monkeypatch):
    install_fake_comfy(monkeypatch)
    latent = make_joint()
    latent["downscale_ratio_spacial"] = 32
    latent["downscale_ratio_temporal"] = 4
    noise = FakeNoise()
    out = refine.run_h3_refinement(
        latent,
        model=object(),
        positive=[[torch.ones(1), {}]],
        negative=None,
        noise=noise,
        sampler=object(),
        sigmas=torch.tensor([]),
        cfg=1.0,
        lock_audio=True,
    )
    assert noise.calls == 0
    assert out["samples"] is latent["samples"]
    assert "downscale_ratio_spacial" not in out
    assert "downscale_ratio_temporal" not in out
