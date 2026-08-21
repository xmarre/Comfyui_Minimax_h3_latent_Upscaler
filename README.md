<p align="center">
  <a href="./README.md"><strong>English</strong></a> ·
  <a href="./README_zh.md">中文</a>
</p>

<div align="center">

# ComfyUI Minimax H3 Latent Upscaler

**Neural Latent Upscaler for Minimax H3 Video Generation**  
Learned · High-fidelity · 2D & 3D Variants

</div>

## 📰 News

- [2026-08-20] 🧩 **Integrated MiniMax H3 refinement**: the H3-aware 3D node now performs the complete learned-upscale + low-sigma H3 sampling pass internally. H3 Continuum V3.4 interop uses exact per-chunk `refine_state` from the companion Continuum implementation; no external BasicGuider, DisableNoise, or SamplerCustomAdvanced is required.
- [2026-08-19] 🚀 **3D node overhaul**: all three resize modes (`scale by multiplier`, `target dimensions`, `megapixels`) merged into a single node; fixed aspect-ratio mismatch in certain modes and edge artifacts at specific sizes.
- [2026-08-18] 🔥 **Precision selector**: both 2D and 3D nodes support `fp32` / `fp16` / `bf16` inference.
- [2026-08-17] 🎉 **Initial release**: Minimax H3 Latent Upscaler 2D + 3D nodes.

This project upscales **MiniMax H3** 24-channel video latents with a trained neural network instead of naive interpolation. It can skip the expensive VAE decode → pixel upscale → VAE encode round-trip and supports a two-stage generation strategy: generate at lower resolution, perform the learned latent upscale, then optionally run a short H3 refinement pass at the target resolution.

> The learned upscale saves time, not VRAM. Any H3 refinement pass still executes the transformer on the target-resolution latent grid.

## Nodes

Three nodes are registered under `video/MinimaxH3`:

- **Minimax H3 Latent Upscaler (2D)** — lightweight learned spatial upscale with temporal layers.
- **Minimax H3 Latent Upscaler (3D)** — fully 3D learned upscale with scale, target-dimensions, and megapixel modes.
- **MiniMax H3 Latent Upscaler + Refine (3D)** — complete MiniMax H3 two-stage path: learned video upscale, AV reconstruction, exact H3 conditioning/masks, fresh enlarged-grid noise, and the actual second sampling pass.

The standalone 2D/3D nodes remain ordinary `LATENT → LATENT` upscalers. The integrated refine node is for workflows that intentionally perform a second H3 pass.

---

## 📸 Examples

**Video upscale comparison**

<video src="examples/Minimax_h3_latent_Upscaler_001.mp4" controls width="640"></video>

**Image upscale comparison**

![](examples/Minimax_h3_latent_Upscaler_002.jpg)

---

## 📁 Project Structure

```text
Comfyui_Minimax_h3_latent_Upscaler/
├── .github/workflows/tests.yml
├── examples/
│   ├── Minimax_h3_latent_Upscaler_001.mp4
│   └── Minimax_h3_latent_Upscaler_002.jpg
├── workflow_templates/
│   └── minimax_h3_r2v_Latent Upscaler example workflow.json
├── nodes/
│   ├── __init__.py
│   ├── minimax_h3_latent_upscaler_2d.py
│   ├── minimax_h3_latent_upscaler_3d.py
│   ├── minimax_h3_refine_support.py       # H3 AV/mask/conditioning helpers
│   └── minimax_h3_refine.py               # complete H3 learned-upscale + refinement
├── tests/
│   ├── test_h3_refine_support.py
│   └── test_h3_refine_node.py
├── README.md
├── README_zh.md
└── __init__.py
```

The model weights are not included in this repository.

---

## 🚀 Key Features

- **Learned latent upscaling** instead of bilinear/bicubic latent interpolation.
- **Two learned backbones**: fast 2D and temporally coherent 3D.
- **Integrated H3 refinement**: the H3-aware node performs the second sampler internally and returns a final decode-ready LATENT.
- **Exact H3 Continuum V3.4 interop** with the companion `ComfyUI-H3-Continuum` refinement-state output.
- **Native AV handling**: only video is spatially upscaled; audio is preserved or deliberately refined.
- **Exact target-conditioning geometry**: target `minimax_keyframes` follow H3's padded target grid while independent `minimax_refs` retain their own latent/RoPE grids.
- **Native-Masked continuation safety**: exact per-chunk video/audio denoise masks are retained and the protected prefix is not accidentally re-denoised.
- **Memory-conscious checkpoint loading**: direct safetensors loading to the selected device/precision, meta-device construction, `load_state_dict(assign=True)`, reduced input cloning, and model caching.
- **Flexible output sizing**: multiplier, target dimensions, or megapixels on the 3D nodes.
- **Flexible precision/device**: CUDA/CPU and fp32/fp16/bf16.

Inference forces learned-upscaler attention off (`attn=False`) for speed/stability. Loaded learned models are cached by `(name, device, precision)`.

---

## 📦 Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.git
```

A normal ComfyUI installation already provides the main runtime dependencies (`torch`, `einops`, `safetensors`). Restart ComfyUI after installing or updating the node.

### Model placement

Put the learned upscaler checkpoint in:

```text
ComfyUI/models/latent_upscale_models/
```

Pre-trained checkpoints are available from:

https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler

The loader auto-detects the stored architecture.

---

## 🧩 Usage

### Standalone latent upscale

```text
MiniMax H3 latent
→ Minimax H3 Latent Upscaler (2D or 3D)
→ VAE Decode
```

This path does **not** run another H3 transformer pass.

### Integrated native H3 refinement

For a native joint H3 AV LATENT:

```text
clean low-resolution joint H3 AV latent
             │
             ▼
MiniMax H3 Latent Upscaler + Refine (3D)
  + MODEL
  + positive CONDITIONING
  + RandomNoise
  + KSamplerSelect
  + partial-denoise SIGMAS
             │
             ▼
      final H3 LATENT
             │
             ▼
         VAE Decode
```

`negative` remains optional for deliberate CFG-style workflows on the explicit native fallback path. Native MiniMax H3 normally uses positive-only BasicGuider semantics; the integrated node constructs that guider internally when `negative` is not connected.

### H3 Continuum V3.4 refinement

Use the companion H3 Continuum PR/release that exposes `refine_state`:

https://github.com/xmarre/ComfyUI-H3-Continuum/pull/15

Correct wiring:

```text
H3 Continuum Sampler V3.4
  video_latents -----> MiniMax H3 Latent Upscaler + Refine.latent
  audio_latents -----> MiniMax H3 Latent Upscaler + Refine.audio_latent
  refine_state ------> MiniMax H3 Latent Upscaler + Refine.refine_state

RandomNoise ---------> MiniMax H3 Latent Upscaler + Refine.noise
KSamplerSelect ------> MiniMax H3 Latent Upscaler + Refine.sampler
partial SIGMAS ------> MiniMax H3 Latent Upscaler + Refine.sigmas

MiniMax H3 Latent Upscaler + Refine.latent
  -------------------> VAE Decode / downstream assembly
```

**Do not add an external BasicGuider, DisableNoise, or SamplerCustomAdvanced.** The integrated node performs that sampling stage itself.

The Continuum `video_latents`, `audio_latents`, and `refine_state` outputs are parallel lists. ComfyUI maps corresponding chunk indices together.

A valid Continuum `refine_state` is authoritative. If an upgraded workflow still has old `model`, `positive`, or `negative` fallback wires attached, those manual conditioning inputs are ignored. Invalid/malformed `refine_state` still fails closed instead of silently falling back to the stale connections.

### What `refine_state` contains

A correct Continuum second pass needs more than the sampled tensors. For each chunk, `refine_state` supplies:

- a fresh MODEL clone preserving the exact chunk-specific Continuum model options/context hint and a fresh Continuum APPLY_MODEL wrapper;
- the exact positive CONDITIONING object that sampler 1 actually received.

If Native Masked continuation used an AV denoise mask, Continuum also attaches the exact video/audio mask members to the matching split LATENT outputs. The integrated upscaler resizes only the target video mask to the enlarged grid and preserves the audio mask.

This is why a separate generic `MiniMaxH3ImageToVideo` conditioning node or unrelated raw H3 MODEL is not considered equivalent to Continuum's real per-chunk state.

### Run Storage

Raw runtime MODEL/CONDITIONING state is deliberately not persisted in H3 Continuum Run Storage. If `refine_state` is requested and Continuum reuses an old chunk prefix, Continuum fails closed rather than pairing new runtime state with reused latents.

For an exact refinement run use either:

- `Run Storage = Off`, or
- regenerate from Chunk 1 so every output chunk is sampled in the current execution.

### Sampling semantics

The integrated node follows ComfyUI's normal advanced-sampler contract:

1. perform the learned video upscale;
2. rebuild clean high-resolution joint H3 AV state;
3. generate independent fresh noise directly on that enlarged AV grid;
4. build the positive-only H3 guider for Continuum, or an optional CFG guider on the explicit native fallback when `negative` is connected;
5. call the supplied ComfyUI `SAMPLER` with the clean latent, generated noise, supplied `SIGMAS`, and denoise mask;
6. let ComfyUI perform the model's normal `model_sampling.noise_scaling(...)` internally;
7. return the sampler result as the final LATENT.

There is **no manual pre-noising/inverse-noise handoff** and therefore no `DisableNoise` stage.

Because a full-noise start gives the clean learned latent zero weight for H3's CONST parameterization, the refinement node requires:

```text
0 <= sigmas[0] < 1
```

A full-denoise schedule beginning at `1.0` is rejected. Use a partial-denoise second-pass schedule.

The exact optimal refinement schedule is workload-dependent. A short pass is the intended use; even a short 2× spatial refinement can still be expensive because doubling latent H and W produces roughly four times as many video tokens for every H3 transformer step. Benchmark the second pass on your hardware rather than treating the learned upscaler itself as the dominant cost.

### Audio control

`lock_audio` has two states:

- **True** — preserve pass-1 audio exactly, zero audio refinement noise, use an audio denoise mask of zero, and restore the clean pass-1 audio after sampling.
- **False** — generate normal H3 audio noise and allow sampler 2 to refine/remix audio while preserving any existing audio denoise mask.

Audio never enters the learned spatial upscaler.

### Conditioning geometry

- `minimax_keyframes` are target-grid conditions and are resized to H3's internally padded even latent H/W.
- `minimax_refs` are independent reference blocks with their own latent dimensions/RoPE grids and are deliberately left unchanged.
- conditioning metadata is cloned rather than mutated;
- list/tuple container type and additional entry fields are preserved.

---

## Node Reference — 2D

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `latent` | LATENT | — | Input MiniMax H3 latent |
| `model_name` | dropdown | auto | Checkpoint in `latent_upscale_models/` |
| `scale` | FLOAT | 2.0 | Spatial upscale factor, 1.0–4.0 |
| `device` | dropdown | cuda | cuda / cpu |
| `precision` | dropdown | fp32 | fp32 / fp16 / bf16 |

Output: final learned-upscaled `LATENT`.

## Node Reference — 3D

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `latent` | LATENT | — | Input MiniMax H3 latent |
| `model_name` | dropdown | auto | Checkpoint in `latent_upscale_models/` |
| `mode` | dropdown | scale by multiplier | multiplier / target dimensions / megapixels |
| `scale` | FLOAT | 2.0 | 1.0–4.0 in multiplier mode |
| `width` / `height` | INT | 1280 / 704 | target pixel size |
| `megapixels` | FLOAT | 1.0 | target megapixel budget |
| `align` | INT | 32 | pixel-grid alignment |
| `keep_proportion` | BOOLEAN | True | preserve aspect ratio where supported |
| `device` | dropdown | cuda | cuda / cpu |
| `precision` | dropdown | fp16 | fp32 / fp16 / bf16 |

Output: final learned-upscaled `LATENT`.

## Node Reference — MiniMax H3 Latent Upscaler + Refine (3D)

The integrated node includes the same learned 3D sizing/model/device/precision controls and adds:

| Parameter | Type | Required? | Description |
| :--- | :--- | :---: | :--- |
| `noise` | NOISE | yes | fresh enlarged-grid refinement noise |
| `sampler` | SAMPLER | yes | actual sampler executed internally |
| `sigmas` | SIGMAS | yes | partial-denoise second-pass schedule |
| `audio_latent` | LATENT | no* | matching audio stream for split H3/Continuum input |
| `refine_state` | H3_CONTINUUM_REFINE_STATE | no** | preferred authoritative Continuum model + conditioning contract |
| `model` | MODEL | no** | native/non-Continuum fallback MODEL; ignored when `refine_state` is connected |
| `positive` | CONDITIONING | no** | native/non-Continuum fallback positive conditioning; ignored when `refine_state` is connected |
| `negative` | CONDITIONING | no | optional native-fallback CFG input; ignored when `refine_state` is connected |
| `cfg` | FLOAT | yes | advanced CFG value for the native fallback when `negative` is used |
| `lock_audio` | BOOLEAN | yes | preserve or refine pass-1 audio |

\* `audio_latent` is required at runtime when `latent` contains only a plain 24-channel video stream. Leave it disconnected when `latent` already contains native joint `[video,audio]` samples.

\** For native/non-Continuum use, connect `model + positive`. For Continuum, connect `refine_state`; a valid `refine_state` takes precedence if old manual fallback wires are still present.

**Output:** one final decode-ready `LATENT`.

---

## 🧪 Model / Architecture

- **Latent format:** 24-channel MiniMax H3 video latent, normalized with the checkpoint's training channel statistics for learned inference.
- **Default detected architecture:** `in_channels=24`, `in_blocks=12`, `out_blocks=12`, `base_channels=512`, `dropout=0.1`, `temporal_every=2`, `temporal_kernel=5`, `attn=False`.
- **Interpolation:** 2D uses bilinear feature interpolation; 3D uses trilinear.
- **Temporal handling:** learned variants preserve T and scale only H×W.
- **H3 DiT grid:** H3 pads target video H/W to its 2×2 patch grid internally and crops back to the requested latent shape. The refinement node leaves the learned output shape unchanged and adjusts target keyframe conditioning instead of adding physical latent cells.

### Validation scope

Synthetic tests cover native/split AV validation, learned-upscaler delegation, exact target geometry, keyframe/reference behavior, denoise-mask reconstruction, Continuum refinement-state resolution and precedence over stale manual fallback wires, actual internal guider/sampler invocation, optional native-fallback CFG behavior, partial-denoise guards, and exact locked-audio restoration.

The branch includes a Python 3.10–3.13 GitHub Actions matrix for compile and CPU/mock tests. The integrated path has also been exercised successfully in a real MiniMax H3 CUDA workflow with the LBH learned checkpoint; exact quality and performance remain workload- and hardware-dependent.

---

## 📊 Training Data

The upscaler was trained on **~80,000 paired samples** (low-resolution latent + high-resolution target), weighted heavily toward video and 2× scaling.

| Modality | Pairs | Share |
| :--- | :--- | :--- |
| Video clips | ~70,000 | ~87.5% |
| 2K images | ~8,000 | ~10% |

Approximate scale distribution:

| Scale | Share |
| :--- | :--- |
| 2× | 40% |
| 1.5× | 10% |
| 2.5× | 10% |
| 3× | 10% |
| 4× | 10% |
| arbitrary 1.0×–4.0× | 10% |

---

## 🙏 Acknowledgments

This node follows the neural-latent-upscaling approach pioneered by [ComfyUi_NNLatentUpscale](https://github.com/Ttl/ComfyUi_NNLatentUpscale) by **Ttl**. The model architecture also draws on the **LTX 2.3 Spatial Upscaler** (`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`).

The H3 refinement integration was independently implemented after studying [Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler](https://github.com/Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler) and current ComfyUI MiniMax H3 sampling code. This repository continues to use the LBH learned upscaler/checkpoints and does not depend on Tr1dae's or Mamad8's learned-upscaler packages.