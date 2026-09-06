<p align="center">
  <a href="./README.md">English</a> ·
  <a href="./README_zh.md"><strong>中文</strong></a>
</p>

<div align="center">

# ComfyUI Minimax H3 Latent Upscaler-Plus

> **Plus 分支：** 这是 `xmarre` 维护的 [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) Plus 分支。它保留上游项目基础，同时维护可能有意与上游不同的功能、集成和修复。

**MiniMax H3 视频生成的神经网络 Latent 放大器**  
Learned · 高保真 · 2D / 3D

</div>

## 📰 更新

- [2026-08-20] 🧩 **集成式 MiniMax H3 精修**：H3 专用 3D 节点现在内部完成完整的 learned upscale + 低噪声 H3 第二次采样。H3 Continuum V3.4 通过精确的逐 chunk `refine_state` 提供 MODEL / CONDITIONING 状态；不再需要外接 BasicGuider、DisableNoise 或 SamplerCustomAdvanced。
- [2026-08-19] 🚀 **3D 节点重构**：倍率、目标尺寸、megapixels 三种缩放模式合并到同一节点。
- [2026-08-18] 🔥 **精度选择**：2D/3D 支持 fp32 / fp16 / bf16。

本项目使用训练好的神经网络直接放大 **MiniMax H3 24 通道视频 latent**，避免简单 bilinear/bicubic latent 插值，也可避免昂贵的 VAE decode → 像素放大 → VAE encode 往返流程。

> Learned upscale 能节省时间，但不会降低第二次 H3 精修在目标分辨率上的峰值显存需求。

## 节点

`video/MinimaxH3` 下包含：

- **Minimax H3 Latent Upscaler (2D)** — 较轻量的 learned spatial upscale。
- **Minimax H3 Latent Upscaler (3D)** — 完整 3D learned upscale，支持倍率、目标尺寸和 megapixels。
- **MiniMax H3 Latent Upscaler + Refine (3D)** — 完整两阶段 H3 路径：learned video upscale、AV 重组、条件/掩码适配、目标网格新噪声，以及实际第二次 H3 sampling。

独立的 2D/3D 节点仍然只是普通 `LATENT → LATENT` 放大器。只有需要第二次 H3 精修时才使用集成 Refine 节点。

---

## 📦 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus.git Comfyui_Minimax_h3_latent_Upscaler
```

模型放到：

```text
ComfyUI/models/latent_upscale_models/
```

预训练权重：

https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler

更新或安装后重启 ComfyUI。

---

## 🧩 使用方法

### 普通 latent upscale

```text
MiniMax H3 latent
→ Minimax H3 Latent Upscaler (2D / 3D)
→ VAE Decode
```

这条路径不会再执行 H3 transformer。

### 原生 H3 两阶段精修

```text
低分辨率 clean joint H3 AV latent
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
       最终 H3 LATENT
              │
              ▼
          VAE Decode
```

MiniMax H3 原生 fallback 路径通常只使用 positive conditioning。`negative` 保留为可选 CFG 兼容输入；未连接 negative 时节点内部使用 positive-only BasicGuider 语义。

### H3 Continuum V3.4

需要配套的 H3 Continuum `refine_state` 支持：

https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/15

正确连接：

```text
H3 Continuum Sampler V3.4
  video_latents -----> MiniMax H3 Latent Upscaler + Refine.latent
  audio_latents -----> MiniMax H3 Latent Upscaler + Refine.audio_latent
  refine_state ------> MiniMax H3 Latent Upscaler + Refine.refine_state

RandomNoise ---------> MiniMax H3 Latent Upscaler + Refine.noise
KSamplerSelect ------> MiniMax H3 Latent Upscaler + Refine.sampler
partial SIGMAS ------> MiniMax H3 Latent Upscaler + Refine.sigmas

MiniMax H3 Latent Upscaler + Refine.latent
  -------------------> VAE Decode / 后续 assembly
```

**不要再添加外部 BasicGuider、DisableNoise 或 SamplerCustomAdvanced。** 第二次采样已经在 Refine 节点内部完成。

Continuum 的 `video_latents`、`audio_latents`、`refine_state` 都是逐 chunk 的并行 list output，ComfyUI 会按相同 index 映射。

有效的 Continuum `refine_state` 是权威输入。如果升级后的旧 workflow 仍然保留 `model`、`positive` 或 `negative` fallback 连接，这些手工 conditioning 输入会被忽略，而不会再因为同时连接而报错，也不会把 Continuum 的 positive-only 第二次采样意外改成 CFG。无效或损坏的 `refine_state` 仍然会 fail closed，不会悄悄退回旧连接。

### `refine_state` 为什么必要

Continuum 的第二次采样不能只靠 video/audio tensor 恢复。每个 chunk 的 `refine_state` 提供：

- 保留该 chunk Continuum model options/context hint 的 fresh MODEL clone，并重新安装 fresh Continuum APPLY_MODEL wrapper；
- sampler 1 实际收到的精确 positive CONDITIONING。

Native Masked continuation 如果使用了 joint AV `noise_mask`，Continuum 还会把精确 video/audio mask 放回对应的 split LATENT。Refine 节点只把 video mask resize 到新的高分辨率目标网格，audio mask 保持原语义。

因此单独重新运行一个通用 `MiniMaxH3ImageToVideo` conditioning，或者把无关的 raw H3 MODEL 接到 sampler 2，都不能视为 Continuum sampler 1 的精确等价状态。

### Run Storage

运行期 MODEL / CONDITIONING 状态不会被持久化到 Continuum Run Storage。如果 `refine_state` 已连接，而本次运行复用了旧 chunk prefix，Continuum 会直接报错，而不是把新捕获的状态错误地和旧 latent 配对。

精确 refine 运行应使用：

- `Run Storage = Off`，或
- 从 Chunk 1 重新生成，使所有输出 chunk 都在当前执行中采样。

### 第二次采样语义

集成节点直接遵循 ComfyUI 的正常 advanced-sampler 路径：

1. learned upscaler 只放大 24-channel video latent；
2. 重建 clean 高分辨率 joint H3 `[video,audio]`；
3. 在新的 AV 网格上生成独立 fresh noise；
4. Continuum 使用 positive-only H3 guider；只有显式 native fallback 路径在连接 `negative` 时才使用 CFG guider；
5. 用输入的 `SAMPLER`、`SIGMAS`、clean latent、noise 和 denoise mask 调用采样；
6. ComfyUI sampler 自己执行正常 `model_sampling.noise_scaling(...)`；
7. 节点直接输出最终、可 decode 的 LATENT。

不再存在手工 pre-noise / inverse-noise handoff，也不需要 DisableNoise。

对于 H3 CONST 参数化，full-noise 起点会让 learned-upscaled clean latent 的权重变成 0，因此 Refine 节点要求：

```text
0 <= sigmas[0] < 1
```

以 `1.0` 开始的 full-denoise schedule 会被拒绝。请使用 partial-denoise 第二次 schedule。

第二次采样应保持短小；即使只有少量 steps，2× spatial refine 仍可能很昂贵，因为 latent H/W 同时翻倍会让每个 H3 transformer step 的 video token 数量约变成 4 倍。实际速度应在目标硬件和工作流上测量，不要把 learned upscaler 本身当作主要耗时来源。

### Audio

`lock_audio=True`：

- audio 不进入 learned spatial upscaler；
- audio refinement noise = 0；
- audio denoise mask = 0；
- sampler 2 后恢复 sampler 1 的 clean audio，不把它强制转换为 learned video 的精度。

`lock_audio=False`：

- 生成正常 H3 audio noise；
- 保留已有 audio denoise mask；
- sampler 2 可以 refine/remix audio。

### Conditioning geometry

- `minimax_keyframes` 属于目标视频网格，放大后会更新到 H3 内部 padded-even 目标 H/W；
- `minimax_refs` 是独立 reference block，拥有自己的 latent H/W 和 RoPE grid，故保持不变；
- metadata 通过 clone 更新，不原地修改调用方数据；
- conditioning entry 的 list/tuple 类型和额外字段保持不变。

---

## Refine 节点接口

| 参数 | 类型 | 必需？ | 说明 |
| :--- | :--- | :---: | :--- |
| `latent` | LATENT | 是 | native joint H3 或 split video latent |
| `noise` | NOISE | 是 | 新目标网格 fresh noise |
| `sampler` | SAMPLER | 是 | 节点内部实际执行的 sampler |
| `sigmas` | SIGMAS | 是 | partial-denoise 第二次 schedule |
| `audio_latent` | LATENT | 否* | split H3/Continuum 的对应 audio latent |
| `refine_state` | H3_CONTINUUM_REFINE_STATE | 否** | Continuum 推荐且权威的 MODEL + CONDITIONING 路径 |
| `model` | MODEL | 否** | 非 Continuum / native fallback；连接 `refine_state` 时忽略 |
| `positive` | CONDITIONING | 否** | 非 Continuum / native positive fallback；连接 `refine_state` 时忽略 |
| `negative` | CONDITIONING | 否 | native fallback 的可选 CFG 输入；连接 `refine_state` 时忽略 |
| `cfg` | FLOAT | 是 | 仅 native fallback 且连接 negative 时有意义 |
| `lock_audio` | BOOLEAN | 是 | 锁定或精修 sampler-1 audio |

\* 主 `latent` 是 plain 24-channel video 时，`audio_latent` 在运行时必须连接；native joint AV 输入则不要连接。

\** 非 Continuum 路径连接显式 `model + positive`；Continuum 路径连接 `refine_state`。如果旧 workflow 仍保留手工 fallback 连接，有效的 `refine_state` 会优先并忽略它们。

**输出：一个最终可直接 VAE Decode 的 `LATENT`。**

---

## 🚀 Learned upscaler 特性

- 24-channel MiniMax H3 latent。
- 2D / 3D 两种网络。
- 3D 支持倍率、目标尺寸、megapixels。
- 自动检测 checkpoint 架构。
- safetensors 直接加载到目标 device/precision。
- meta-device 构建 + `load_state_dict(assign=True)`，减少 CPU 峰值。
- 按 `(name, device, precision)` 缓存 learned model。
- 支持 CUDA / CPU、fp32 / fp16 / bf16。
- 推理时强制关闭 learned-upscaler attention (`attn=False`) 以提高速度与稳定性。

H3 本身会把目标 video latent H/W pad 到 2×2 DiT patch grid 后再 crop 回请求尺寸。Refine 节点不会物理增加 latent cell，而只更新需要匹配目标网格的 keyframe conditioning。

---

## 🧪 验证

CPU/mock regression tests 覆盖：

- native/split AV validation；
- learned 3D delegation；
- target geometry；
- keyframe/reference semantics；
- mask reconstruction；
- Continuum exact refine-state 与其对旧手工 fallback 连接的优先级；
- 无效 refine-state 的 fail-closed 行为；
- 内部 guider/sampler 实际调用；
- native fallback optional negative CFG；
- partial-denoise guard；
- `lock_audio=True` 的精确 audio 恢复。

分支包含 Python 3.10–3.13 GitHub Actions matrix。集成路径也已在真实 MiniMax H3 + LBH learned checkpoint 的 CUDA 工作流中成功执行；具体画质与耗时仍取决于工作流和硬件。

---

## 📊 训练数据

Learned upscaler 使用约 **80,000 组 paired samples** 训练，其中视频占主要部分，2× 是最常见倍率。

| 数据类型 | 数量 | 占比 |
| :--- | :--- | :--- |
| 视频片段 | ~70,000 | ~87.5% |
| 2K 图片 | ~8,000 | ~10% |

大致倍率分布：2× 约 40%；1.5× / 2.5× / 3× / 4× 各约 10%；另有约 10% 的 1.0×–4.0× 任意小数倍率用于提高连续倍率泛化。

---

## 🙏 致谢

本项目延续 **Ttl** 的 [ComfyUi_NNLatentUpscale](https://github.com/Ttl/ComfyUi_NNLatentUpscale) 神经 latent upscale 思路，模型架构也参考了 **LTX 2.3 Spatial Upscaler**。

H3 refine integration 在研究 [Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler](https://github.com/Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler) 与当前 ComfyUI MiniMax H3 sampling 代码后独立实现。本仓库仍使用 LBH 的 learned upscaler/checkpoint，不依赖 Tr1dae 或 Mamad8 的 learned-upscaler package。