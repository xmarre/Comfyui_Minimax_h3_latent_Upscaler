from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes" / "minimax_h3_refine_sequence.py"
PACKAGE = "h3_refine_sequence_tests"
if PACKAGE not in sys.modules:
    pkg = types.ModuleType(PACKAGE)
    pkg.__path__ = [str(MODULE_PATH.parent)]
    sys.modules[PACKAGE] = pkg
SPEC = importlib.util.spec_from_file_location(f"{PACKAGE}.sequence", MODULE_PATH)
assert SPEC and SPEC.loader
sequence = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sequence
SPEC.loader.exec_module(sequence)


class FakeNested:
    is_nested = True

    def __init__(self, members):
        self.members = list(members)

    def unbind(self):
        return tuple(self.members)

    def to(self, device):
        return FakeNested([member.to(device) for member in self.members])


def _joint(video, audio, video_mask=None, audio_mask=None):
    latent = {"samples": FakeNested([video, audio])}
    if video_mask is not None:
        latent["noise_mask"] = FakeNested([video_mask, audio_mask])
    return latent


def test_registered_refine_node_consumes_and_emits_chunk_lists():
    cls = sequence.MinimaxH3LatentUpscaler3DRefineSequence
    assert cls.INPUT_IS_LIST is True
    assert cls.OUTPUT_IS_LIST == (True,)
    assert sequence.NODE_CLASS_MAPPINGS["MinimaxH3LatentUpscaler3DRefineHandoff"] is cls


def test_carry_replaces_only_fully_protected_video_prefix():
    current_video = torch.zeros((1, 24, 5, 3, 4))
    previous_video = torch.zeros_like(current_video)
    previous_video[:, :, -2:] = 9.0
    current_audio = torch.full((1, 32, 2, 8), 5.0)
    previous_audio = torch.full_like(current_audio, 7.0)
    video_mask = torch.ones((1, 1, 5, 3, 4))
    video_mask[:, :, :2] = 0.0
    # A spatially partial zero at t=2 is not a continuation slot and must not
    # extend the protected prefix.
    video_mask[:, :, 2, :1, :1] = 0.0
    audio_mask = torch.zeros((1, 1, 2, 8))

    current = _joint(current_video, current_audio, video_mask, audio_mask)
    previous = _joint(previous_video, previous_audio)
    video_steps, audio_steps = sequence._carry_previous_refined_prefix(
        current,
        previous,
        carry_audio=False,
    )

    assert video_steps == 2
    assert audio_steps == 0
    assert torch.all(current_video[:, :, :2] == 9.0)
    assert torch.all(current_video[:, :, 2:] == 0.0)
    assert torch.all(current_audio == 5.0)


def test_unlocked_audio_can_carry_its_exact_native_masked_prefix():
    video = torch.zeros((1, 24, 4, 2, 2))
    previous_video = torch.zeros_like(video)
    audio = torch.zeros((1, 32, 2, 7))
    previous_audio = torch.zeros_like(audio)
    previous_audio[..., -3:] = 4.0
    video_mask = torch.ones((1, 1, 4, 2, 2))
    audio_mask = torch.ones((1, 1, 2, 7))
    audio_mask[..., :3] = 0.0

    current = _joint(video, audio, video_mask, audio_mask)
    previous = _joint(previous_video, previous_audio)
    video_steps, audio_steps = sequence._carry_previous_refined_prefix(
        current,
        previous,
        carry_audio=True,
    )

    assert video_steps == 0
    assert audio_steps == 3
    assert torch.all(audio[..., :3] == 4.0)
    assert torch.all(audio[..., 3:] == 0.0)


def test_sequence_refinement_carries_actual_post_refine_tail_before_next_sample(monkeypatch):
    seen_clean_videos = []

    def fake_build(current_latent, positive, **_kwargs):
        index = int(current_latent["chunk"])
        video = torch.full((1, 24, 5, 2, 2), float(index))
        audio = torch.full((1, 32, 2, 7), 3.0)
        video_mask = torch.ones((1, 1, 5, 2, 2))
        if index == 2:
            video_mask[:, :, :2] = 0.0
        audio_mask = torch.zeros((1, 1, 2, 7))
        return _joint(video, audio, video_mask, audio_mask), positive, None

    def fake_sample(clean, **_kwargs):
        video, audio = clean["samples"].unbind()
        seen_clean_videos.append(video.clone())
        out_video = video.clone()
        if len(seen_clean_videos) == 1:
            # Simulate sampler 2 changing chunk 1's tail.  Chunk 2 must carry
            # this post-refine value, not its independently upscaled prefix.
            out_video[:, :, -2:] = 11.0
        return {"samples": FakeNested([out_video, audio.clone()])}

    monkeypatch.setattr(sequence, "build_clean_h3_upscale", fake_build)
    monkeypatch.setattr(sequence, "_model_with_refinement_contract", lambda model: model)
    monkeypatch.setattr(sequence, "run_h3_refinement", fake_sample)

    states = [
        {"api": 1, "model": object(), "positive": [[torch.ones(1), {"chunk": 1}]]},
        {"api": 1, "model": object(), "positive": [[torch.ones(1), {"chunk": 2}]]},
    ]
    (outputs,) = sequence.MinimaxH3LatentUpscaler3DRefineSequence().upscale_and_refine(
        [{"chunk": 1}, {"chunk": 2}],
        [object()],
        [object()],
        [torch.tensor([0.2, 0.0])],
        ["model.safetensors"],
        ["scale by multiplier"],
        [1.5],
        [1280],
        [704],
        [1.0],
        [32],
        [True],
        [True],
        [1.0],
        ["cpu"],
        ["fp32"],
        refine_state=states,
    )

    assert len(outputs) == 2
    assert len(seen_clean_videos) == 2
    assert torch.all(seen_clean_videos[1][:, :, :2] == 11.0)
    assert torch.all(seen_clean_videos[1][:, :, 2:] == 2.0)
