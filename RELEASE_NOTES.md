# MiniMax H3 Latent Upscaler v0.1.0

v0.1.0 is the first formal automated release of the learned MiniMax H3 latent upscaler repository and includes the complete integrated high-resolution refinement path.

## Learned latent upscaling

The existing 2D and 3D learned upscalers remain available. The 3D node supports multiplier, target-dimension and megapixel sizing, precision selection, latent-grid alignment and host-memory optimization.

## Integrated MiniMax H3 upscale + refine

The new `MiniMax H3 Latent Upscaler + Refine (3D)` performs the complete second-pass operation internally:

1. accept native joint H3 AV state or split Continuum video/audio latents;
2. run only the 24-channel video latent through the learned 3D upscaler;
3. rebuild the high-resolution joint AV latent;
4. resize target keyframes and video denoise masks while preserving independent reference grids;
5. generate fresh noise directly on the enlarged H3 grid;
6. construct the appropriate guider;
7. execute the supplied ComfyUI sampler over the supplied low-sigma schedule;
8. optionally restore pass-1 audio exactly;
9. return a decode-ready H3 LATENT.

No external `BasicGuider`, `DisableNoise` or `SamplerCustomAdvanced` reconstruction is required.

## Exact H3 Continuum handoff

H3 Continuum v3.4.1 can supply `video_latents`, `audio_latents` and `refine_state` per chunk. A valid `refine_state` is authoritative and carries the exact per-chunk Continuum MODEL wrapper plus positive conditioning used by sampler 1.

Stale manual `model`, `positive` and `negative` fallback wires are ignored when a valid `refine_state` is connected, preventing upgraded workflows from crashing or silently changing the positive-only Continuum contract. Malformed refinement state still fails closed.

Native Masked denoise masks remain aligned with the matching chunk. Only the target video mask is resized to the enlarged grid; the audio mask is preserved.

## Short-refinement interoperability

The refiner clones the exact sampler-1 MODEL and marks only sampler 2 with an `h3_refinement` API-v1 contract containing the full H3 sigma reference. This lets coordinated Spectrum and DiffAid releases distinguish a short low-sigma refinement from ordinary Continuum generation without mutating the source MODEL.

With Spectrum MiniMax H3 v0.2.17 and DiffAid v1.0.7, a stable three-step refinement can use:

```text
actual -> forecast -> actual
```

while genuine external-patch transitions remain hard safety barriers.

## Runtime validation

The full CUDA path was validated using H3 Continuum exact `refine_state`, DiffAid, Untwisting RoPE metadata, native ER-SDE and Spectrum on sampler 2.

The validated 0.7 MP native -> 1.75x learned-upscale -> three-step refinement reported `2 actual + 1 forecast` per refined chunk and produced user-confirmed impeccable media quality. The Refine node dropped from roughly 302.5 s with three native target-resolution NFEs to roughly 212.7 s with the middle NFE forecast in the compared run.

## Reliability

The release includes Python 3.10-3.13 compile/test CI and regression coverage for exact refinement-state resolution, stale fallback precedence, malformed-state fail-closed behavior, native and split AV validation, enlarged-grid noise generation, conditioning geometry, masks, locked-audio restoration, CFG fallback behavior, empty sigma schedules and real internal sampler invocation.

This release is coordinated with H3 Continuum v3.4.1, Spectrum MiniMax H3 v0.2.17 and ComfyUI-DiffAid-Patches v1.0.7.
