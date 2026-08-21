# MiniMax H3 Latent Upscaler v0.1.1

v0.1.1 is a backward-compatible maintenance release that consolidates the post-v0.1.0 upstream review, CI hardening, alignment fixes, learned-model offload controls, and the final sequence-aware H3 Continuum handoff fix.

## Selective LBH upstream sync

LBH's 2026-08-21 upstream changes were reviewed individually rather than merged wholesale.

Adopted with redesign:

- **Dual-axis output alignment** now uses a common pixel grid compatible with both the requested `align` value and H3's 16× VAE grid: `lcm(align, 16)`.
- **`keep_proportion=True` remains supported.** The node chooses a nearby valid aligned W/H pair instead of solving alignment by silently distorting the source aspect ratio.
- **Optional learned-model offload** is available through `offload_after_upscale`, but remains `False` by default to avoid unnecessary CPU↔GPU transfers on repeated/high-VRAM workflows.
- **Cached model restore** moves an optionally offloaded learned upscaler back to the requested device on the next use.

Deliberately not adopted:

- fixed 16-frame temporal chunking with only `temporal_kernel // 2` overlap, because repeated Conv3d/TemporalConv layers plus GroupNorm make it non-equivalent to whole-sequence execution and allow chunk-boundary changes;
- forced CPU offload after every upscale;
- out-of-place normalization/denormalization changes that do not improve arithmetic precision but do add full-latent temporary allocations;
- removal of `keep_proportion`.

## Sequence-aware Continuum offload fix

The sequence-aware `MiniMax H3 Latent Upscaler + Refine (3D)` path now correctly accepts and propagates the inherited `offload_after_upscale` setting.

For list-valued H3 Continuum execution the node now:

1. resolves the per-chunk learned-upscaler model/device/precision settings;
2. performs the learned upscale;
3. optionally offloads only that learned-upscaler cache entry before sampler 2;
4. preserves the exact post-refine continuation carry into the next chunk;
5. leaves the default behavior unchanged when `offload_after_upscale=False`.

This closes the gap where the base integrated node exposed the control but the sequence-aware list execution path did not accept/propagate it correctly.

## Alignment and conditioning reliability

The 3D learned upscale path guarantees that both final output axes satisfy the common requested/H3 VAE alignment grid. The integrated refiner continues to preserve H3 target-conditioning semantics:

- target `minimax_keyframes` follow the enlarged target grid;
- independent `minimax_refs` keep their own latent/RoPE geometry;
- Native-Masked video/audio masks remain paired with the correct Continuum chunk;
- sequence refinement carries the actual post-refine tail rather than a stale low-resolution prefix.

## CI and compatibility hardening

The repository no longer relies on the old isolated copied-test workaround. GitHub Actions now tests against reviewed native ComfyUI source revisions with the repository root imported directly.

The matrix validates:

- Python 3.10, 3.11, 3.12 and 3.13;
- multiple reviewed ComfyUI revisions;
- Ruff on the maintained integration/test surfaces;
- `compileall`;
- native ComfyUI source-contract tests;
- the full refinement regression suite;
- dual-axis/common-grid alignment;
- whole-sequence temporal execution;
- cached-model restore/offload behavior;
- sequence-aware per-chunk offload propagation.

## Documentation corrections

The README now documents the selective-upstream policy, exact alignment semantics and offload tradeoffs. The installation command also correctly clones this `xmarre` fork rather than the LBH upstream repository.

## Upgrade notes

No workflow migration is required. Existing workflows keep the same default runtime behavior. `offload_after_upscale` remains disabled unless explicitly enabled.
