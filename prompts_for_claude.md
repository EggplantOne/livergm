# 指导 MONAI GenerativeModels 开发的 Prompt

以下 prompt 按顺序使用，每个在新对话或同一对话中依次发送。

---

## Prompt 0：项目背景介绍（首次对话时先发这个）

```
我正在基于 MONAI GenerativeModels（当前目录）开发一个 3D 肝脏血管分割增强框架。整体思路借鉴了一个叫 CVI 的 2D 项目（4-phase pipeline），我现在要把它扩展到 3D。

核心目标：用扩散模型生成合成的 3D 肝脏血管 mask 和对应的 CT 图像，用合成数据增强下游分割模型的训练。

整体 pipeline 分 4 个阶段：
- Phase 1：训练 3D MaskGenerator，从噪声生成逼真的 3D 血管 mask（用 AutoencoderKL + DiffusionModelUNet，都设 spatial_dims=3）
- Phase 2：训练 3D ImageGenerator，根据 3D mask 生成对应的 CT 图像（用 ControlNet + ControlNetLatentDiffusionInferer 做 mask→CT 的条件生成）
- Phase 3：训练 3D 分割器（3D UNet），用真实数据+合成数据一起训练
- Phase 4：DCL 联合优化（后续再做）

我的数据：
- 3D 肝脏 CT 图像和对应的血管标注 mask，NRRD 格式
- 数据量较少（few-shot 场景），所以需要合成数据增强

当前目录是 MONAI GenerativeModels 的代码库，里面有我需要的所有 3D 组件：
- generative/networks/nets/autoencoderkl.py — 3D VAE
- generative/networks/nets/diffusion_model_unet.py — 3D 扩散 UNet
- generative/networks/nets/controlnet.py — 3D ControlNet
- generative/inferers/inferer.py — ControlNetLatentDiffusionInferer
- generative/networks/schedulers/ — DDPM/DDIM 调度器
- tutorials/generative/3d_ldm/ — 3D LDM 教程
- tutorials/generative/3d_autoencoderkl/ — 3D VAE 教程
- model-zoo/models/brain_image_synthesis_latent_diffusion_model/ — 脑部 MRI 预训练模型配置参考

请先熟悉当前代码库的结构和关键组件，然后我们开始逐步开发。
```

---

## Prompt 1：Phase 1 — 训练 3D VAE（AutoencoderKL）

```
现在开始 Phase 1 的第一步：训练 3D AutoencoderKL，用于将 3D 血管 mask 压缩到潜空间。

请参考以下内容：
- tutorials/generative/3d_autoencoderkl/3d_autoencoderkl_tutorial.py 的训练流程
- model-zoo/models/brain_image_synthesis_latent_diffusion_model/configs/inference.json 的模型配置
- generative/networks/nets/autoencoderkl.py 的实现

我需要你帮我写一个训练脚本，要求：
1. 数据加载：读取 NRRD 格式的 3D 血管 mask，裁剪/resize 到合适的 patch size（比如 64x64x64 或 128x128x128，根据显存决定）
2. 模型：AutoencoderKL，spatial_dims=3，参考脑部 MRI 模型的配置但适当调整
3. 训练目标：重建损失 + KL 损失，让 VAE 学会压缩和重建 3D mask
4. 保存训练好的 VAE 权重，后续 Phase 1 的扩散模型和 Phase 2 都要复用

先分析一下教程代码和脑部 MRI 的配置，然后给出训练脚本。
```

---

## Prompt 2：Phase 1 — 训练 3D 扩散模型（生成 mask）

```
Phase 1 第二步：在训练好的 3D VAE 潜空间中训练扩散模型，学会从噪声生成 3D 血管 mask。

请参考：
- tutorials/generative/3d_ldm/3d_ldm_tutorial.py 的 LDM 训练流程
- generative/networks/nets/diffusion_model_unet.py 的 DiffusionModelUNet
- generative/inferers/inferer.py 的 LatentDiffusionInferer
- generative/networks/schedulers/ 的 DDPMScheduler 和 DDIMScheduler

我需要你帮我写一个训练脚本，要求：
1. 加载上一步训练好的 3D VAE 权重，冻结 VAE
2. 用 DiffusionModelUNet(spatial_dims=3) 作为去噪网络
3. 用 LatentDiffusionInferer 处理训练和采样
4. 训练：将 mask 编码到潜空间 → 加噪 → UNet 预测噪声 → MSE 损失
5. 推理：从随机噪声开始 → DDIM 采样去噪 → VAE 解码 → 生成 3D mask
6. 保存扩散模型权重

同时写一个推理脚本，能加载训练好的 VAE + 扩散模型，生成若干 3D mask 并保存为 NRRD 文件。
```

---

## Prompt 3：Phase 2 — 训练 3D ControlNet（mask→CT 条件生成）

```
现在开始 Phase 2：训练 mask→CT 的条件生成模型。

这一步需要：
1. 一个新的 3D AutoencoderKL，用于压缩 CT 图像（不是 mask）到潜空间。这个 VAE 需要单独训练，因为 CT 图像和 mask 的数据分布不同。或者如果数据量不够，也可以复用 Phase 1 的 VAE，你分析一下哪种更合适。
2. 一个 3D ControlNet，接收 mask 作为条件输入
3. 一个 3D DiffusionModelUNet，在 CT 的潜空间中做去噪
4. 用 ControlNetLatentDiffusionInferer 处理训练和采样

请参考：
- generative/networks/nets/controlnet.py 的 ControlNet 实现
- generative/inferers/inferer.py 的 ControlNetLatentDiffusionInferer
- 2D ControlNet 教程（如果有的话）作为流程参考

训练流程：
- 输入：真实 CT + 对应的 mask
- CT → VAE.encode() → latents
- mask → ControlNet 条件输入
- 给 latents 加噪 → ControlNet + UNet 预测噪声 → MSE 损失
- 训练 ControlNet 和 UNet，冻结 VAE

推理流程：
- 输入一个 3D mask（可以是 Phase 1 生成的）
- 从随机噪声开始 → ControlNet + UNet 去噪 → VAE 解码 → 生成 CT
- 保存为 NRRD

请帮我写训练脚本和推理脚本。
```

---

## Prompt 4：合成数据生成 + 整合

```
Phase 1 和 Phase 2 都训练完成后，我需要一个脚本来批量生成合成数据：

1. 用 Phase 1 的模型生成 N 个 3D 血管 mask
2. 用 Phase 2 的模型根据这些 mask 生成对应的 CT 图像
3. 将生成的 mask-CT 对保存为 NRRD 文件，组织成和真实数据相同的目录结构
4. 输出一个汇总信息（生成了多少对，保存在哪里）

这些合成数据后续会和真实数据合并，用于训练 3D 分割模型（Phase 3）。
```
