# 官方 DiT 适配源码索引与 pc_convert 对照

日期：2026-11-10。用途：为“官方已经实现什么、pc_convert 实际接了什么、还欠什么证据”提供固定版本的一手依据。

**结论：官方已经有特定 FLUX/DiT 架构的适配和完整生图样例。pc_convert 的通用 `litert_torch.convert` 调用不会自动获得这些模型包装、分片、位置编码和调度逻辑。已有 SoC target，也不会自动获得 MT6897 的完整生图验收。**

本次使用公开 GitHub API 读取源码、解析远端 commit，以及阅读官方硬件文档；没有下载模型、安装软件、转换模型或执行设备测试。文中的“官方测量”指对应样例作者报告，不能填写到“本项目实测”一栏。

## 1. 固定比较对象，避免同名版本混用

| 对象 | 固定版本 / commit | 用途 |
|---|---|---|
| `google-ai-edge/litert-torch` | `v0.10.0` → `90cbbaa61ab3f029f1273b108909a5b4d9cc2c12` | 官方模型注册、模型重写、分片导出与量化 |
| `google-ai-edge/litert-samples` | `f9f0e000630e9cdd6d263c4cd80495e4231a2dc8` | 本次读取时 main；commit 时间 `2026-11-10T04:03:29Z`；完整生图 host 与作者测量 |
| `google-ai-edge/LiteRT` | `v2.3.0` → `4352e261d3f1b3cda00345db5dbd75859d3bf0d3` | MT6897 编译 target 与 NeuroPilot 版本映射 |

commit 来自 GitHub 远端 ref/commit API，不是依据本地 wheel 文件名猜测。可复查：[Torch tag ref](https://api.github.com/repos/google-ai-edge/litert-torch/git/ref/tags/v0.10.0)、[samples 固定 commit](https://github.com/google-ai-edge/litert-samples/commit/f9f0e000630e9cdd6d263c4cd80495e4231a2dc8)、[LiteRT tag ref](https://api.github.com/repos/google-ai-edge/LiteRT/git/ref/tags/v2.3.0)。[Torch 正式 release](https://github.com/google-ai-edge/litert-torch/releases/tag/v0.10.0) 是发布记录，release 文案未穷举代码中的所有生图功能。

以下源码行号均为 GitHub API 返回的实际文件 1-based 行号，永久链接锁定 commit。代码为 Apache-2.0 官方源码的必要节录；`...` 为删节，表格中的 shape 是依据相邻 source 计算，未读取实际模型输出文件。

### 最重要的两条路线区别

1. **官方 Torch `Flux2Klein` 前端**：3 个文本编码器分片＋FP16 embedding bin＋8 个 DiT 分片＋VAE decoder。已确认源码存在及连接方式；本次未取得这条 12 图分片路线的实际输出与设备记录。
2. **官方 Bonsai Image 4B 样例**：text encoder、DiT、VAE 三个 `.tflite` 图＋host sampling loop。样例 README 报告了 CPU/XNNPACK 生图和 Apple Metal DiT 的运行结果。它使用 FLUX.2-klein 架构，但其性能记录不能移借给上述 8 分片 DiT 路线。

这两个对象同时属于官方公开仓库，结构和验收证据不同，比较时必须保持分开。

## 2. 官方已接通模型级前端，远不止一个 convert 调用

### 2.1 注册与 checkpoint 目录识别

`litert_torch/generative/export_hf/model_ext/exportables.py:217–224` 注册的是 `get_image_gen_model_cls`；源码中接受 Bonsai 和 FLUX2-Klein 两个模型类。该入口不提供任意 DiT 自动适配。

```python
if model_type == 'bonsai_flux2':
  return bonsai_flux2_lib.BonsaiFlux2
elif model_type in ('flux2_klein', 'Flux2KleinPipeline'):
  return flux2_klein_lib.Flux2Klein
```

[永久来源：模型注册](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/exportables.py#L217-L224)

`core/export_lib.py:279–318` 对 `TEXT_TO_IMAGE` 提前分流，读取 Diffusers 多组件目录的 `model_index.json`，识别 `_class_name`，加载对应 model class 与 tokenizer；`1217–1229` 调用模型自己的 `export()` 并收集全部 artifact。上游清楚区分“多组件 pipeline”与“根目录一个 Transformers config”的入口。

```python
model_cls = model_ext_exportables.get_image_gen_model_cls(model_type)
model = model_cls(local_model_dir, export_config=export_config)
# ... export_image_gen_models ...
artifacts = image_gen_model.export(export_config)
```

[永久来源：加载分流](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/core/export_lib.py#L279-L318)、[导出分流](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/core/export_lib.py#L1217-L1229)

**对照意义**：本地若只加载一个 TorchScript/自定义 Module 再 `litert_torch.convert(model, sample_inputs)`，则模型架构、checkpoint 装载和 forward 兼容性仍由那个 Module 负责；上述 `get_image_gen_model_cls → model.export` 分支不会凭空执行。

### 2.2 官方按 Diffusers 参数名重写模型，加载真实权重有严格合同

`bonsai_flux2/modeling_flux2.py` 实现自己的 PyTorch DiT 与 VAE，文件头说明保留 Diffusers 模块层级和参数名，并将 RoPE 频率计算保持为 FP32。`35–65` 支持单个及分片 safetensors，浮点权重转换 FP32；`758–793` 从模型 config 构造，再严格装载 state dict。

```python
state_dict[k] = v.to(torch.float32) if v.is_floating_point() else v
# ...
model.load_state_dict(state_dict, strict=True)
return model.to(torch_dtype)
```

[永久来源：权重读取](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/bonsai_flux2/modeling_flux2.py#L15-L65)、[构造及严格装载](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/bonsai_flux2/modeling_flux2.py#L758-L793)

`use_random_weights=True` 是明确的另外一条代码分支；随机权重结构验证与真实 checkpoint 转换不应共用一个“模型已支持”标签。

## 3. FLUX2-Klein 分片结构、输入合同与 shape

### 3.1 分片数量和边界

`flux2_klein/flux2_klein.py:58–61` 给出 3 个 text encoder shard、2 个 double-stream shard、4 个 single-stream shard。DiT 再加 initial 和 final，总共 8 个图；加 text encoder 3 图及 VAE 为 12 个 `.tflite` 图，embedding 是独立 `.bin`。

```python
_TEXT_EMBED_FILENAME = "qwen_embed_fp16.bin"
_NUM_TE_SHARDS = 3
_NUM_DIT_DOUBLE_SHARDS = 2
_NUM_DIT_SINGLE_SHARDS = 4
```

[永久来源：分片常量](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/flux2_klein/flux2_klein.py#L55-L68)

Double/single blocks 按连续区间平均切分，边界是 `i*n//num_shards` 和 `(i+1)*n//num_shards`；该分法没有读取 MT6897 的 L1 大小、SRAM 或最大图容量。它是已有架构分片，不能称为 MediaTek 硬件自适应切图。

```python
start = (i * n) // num_shards
end = ((i + 1) * n) // num_shards
shards.append(nn.ModuleList(items[start:end]))
```

[永久来源：连续分片算法](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/flux2_klein/flux2_klein.py#L37-L52)

### 3.2 符号与输入顺序

令输出图像边长为 `S`，`N=(S/16)^2` 为 image tokens，`L` 为 text tokens，`J=N+L`，`C` 为 DiT in_channels，`E` 为 joint_attention_dim，`D=head_dim×num_heads`。`S` 必须是正的 16 倍数；`L` 默认 256，可由 `extra_kwargs.max_seq_len` 或特定 prefill_lengths 覆盖。`S=512` 时 `N=1024`，若 `L=256` 则 `J=1280`，这些数不等同于 LLM KV cache length。

表中名称是 **PyTorch forward 的参数名及顺序**；本次未生成各 shard flatbuffer，故不冒充实际 tensor 命名清单。各 sample_inputs 都使用 FP32；weight 量化不等于这些浮点边界自动变成 A16。

| 图 | forward 输入顺序 | 固定 sample shape | 输出含义 |
|---|---|---|---|
| Text encoder shard ×3 | hidden_states, attention_mask, cos, sin | `[1,L,Ht]`、`[1,heads_t,L,L]`、两份 `[1,L,head_dim_t]` | 该层段 hidden states；host 需要保留各选定层输出 |
| DiT initial ×1 | hidden_states, encoder_hidden_states, timestep | `[1,N,C]`、`[1,L,E]`、`[1]` | image、text、mod_img、mod_txt、mod_single、temb 共6项 |
| DiT double block ×2 | image, text, cos, sin, mod_img, mod_txt | `[1,N,D]`、`[1,L,D]`、两份 `[1,J,1,head_dim]`、两份 `[1,6D]` | image、text |
| DiT single block ×4 | joint, cos, sin, mod_single | `[1,J,D]`、两份 `[1,J,1,head_dim]`、`[1,3D]` | joint |
| DiT final ×1 | joint, temb | `[1,J,D]`、`[1,D]` | 去掉前面的 text tokens，norm_out + proj_out |
| VAE decoder ×1 | latents | `[1,latent_channels,S/8,S/8]` | RGB 图像 tensor |

直接决定 shape 的原文：`flux2_klein.py:171–179、216–223`。

```python
grid = image_size // 16
img_tokens = grid * grid
joint_tokens = seq_len + img_tokens
# double_inputs 节录
torch.ones(1, joint_tokens, 1, head_dim, dtype=torch.float32),
torch.zeros(1, joint_tokens, 1, head_dim, dtype=torch.float32),
```

[永久来源：全部 DiT sample inputs](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/flux2_klein/flux2_klein.py#L145-L262)、[Text encoder shape](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/flux2_klein/flux2_klein.py#L98-L131)、[forward 合同](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/flux2_klein/modeling.py#L37-L160)、[VAE shape](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/bonsai_flux2/bonsai_flux2.py#L528-L540)

initial 的 timestep 还有 `×1000` 语义，绝不能只看 dtype/shape：

```python
temb = self.time_guidance_embed(timestep * 1000.0, None)
return image, text, mod_img, mod_txt, mod_single, temb
```

[永久来源：initial 计算](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/flux2_klein/modeling.py#L84-L90)

**对照意义**：三输入合同只恰好覆盖此官方实现的 initial 图；完整 DiT 包括后续 6/4/2 输入分片以及6输出初始结果的路由。一个名为 `dit` 的三输入泛化接口，不能靠名字获得这些计算。

## 4. 官方已有的 attention、RoPE 和 weight 适配

### 4.1 Text encoder 保留中间层，独立 embedding 和 4D attention

`BonsaiFlux2` 默认取 Qwen3 第 9/18/27 层 hidden state；单图 `PromptEmbedder` 将它们沿 feature 维组合。Klein 分片使用这些边界切 3 段，embedding table 单独写 FP16 bin，各段输入仍是 FP32 hidden state。它导出的是 prompt encoder，用不上文本生成 LM head。

`bonsai_flux2.py:69–90` 显式设置 eager attention，替换 repeat_kv 为不经 expand/broadcast_to 的 ≤4D cat/reshape：

```python
return torch.cat([hidden_states] * n_rep, dim=2).reshape(
    batch, num_key_value_heads * n_rep, slen, head_dim
)
# configure_qwen3_for_gpu_export
modeling_qwen3.repeat_kv = _repeat_kv_4d
```

[永久来源：Qwen attention 适配](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/bonsai_flux2/bonsai_flux2.py#L69-L126)、[Klein embedding/分片](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/flux2_klein/flux2_klein.py#L73-L143)

### 4.2 RoPE 有实际计算接线

官方重写 `modeling_flux2.py:93–108` 在 FP32 计算频率；`111–141` 用 real/imag 成对旋转及4D reshape；double/single attention 前实际应用到 query/key。整体图输入 img_ids/txt_ids 后在图内算 cos/sin，分片图则显式接收 cos/sin 并传给每个 block。下面是后者的实际传递点：

```python
text, image = block(
    hidden_states=image,
    encoder_hidden_states=text,
    temb_mod_img=mod_img,
    temb_mod_txt=mod_txt,
    image_rotary_emb=(cos, sin),
)
```

[永久来源：分片实际接线](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/flux2_klein/modeling.py#L100-L117)、[FP32/4D RoPE](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/bonsai_flux2/modeling_flux2.py#L93-L141)

官方 samples 另有 GPU 专用脚本：按真实 512×512 position IDs 预计算 RoPE，并**替换 forward 让这些常量参与 tracing**；成对旋转使用4D matmul。`export_dit_gpu.py:69–95` 的关键两处：

```python
flux2mod.Flux2PosEmbed.forward = lambda self, ids: (
    (COS_TXT, SIN_TXT) if ids.shape[0] == TXT else (COS_IMG, SIN_IMG))
# rank4_rope
return (x.float() * cos + (x.float() @ P) * sin).to(x.dtype)
```

[永久来源：GPU RoPE 常量接入](https://github.com/google-ai-edge/litert-samples/blob/f9f0e000630e9cdd6d263c4cd80495e4231a2dc8/models/bonsai/bonsai_image_4b/converted/export_dit_gpu.py#L66-L95)

因此“生成一个独立 rope.bin”和“图的 forward 使用这个表”是两个完成状态。官方这两条路径都能指出消费者；对本地也应追到实际 tensor/forward 的消费者。

### 4.3 Attention 与 VAE 的算子形状约束

官方 DiT 的 Q/K/V permute 成 `[B,heads,seq,head_dim]`，调用非 causal SDPA；单流保留 `to_qkv_mlp_proj` 合并投影，再拆 q/k/v 与 MLP 分支。这个模型重写没有声称把所有 attention 手工改成 MediaTek 专用 kernel。

```python
attn_out = F.scaled_dot_product_attention(
    q_t, k_t, v_t, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
)
```

[永久来源：双流 attention](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/bonsai_flux2/modeling_flux2.py#L318-L372)、[单流投影/attention](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/bonsai_flux2/modeling_flux2.py#L402-L448)

VAE 的 `Flux2GroupNorm` 专门保持4D，避免权重是单位常量时出现 BROADCAST_TO；weight/bias 恢复 `[1,C,1,1]`。这是前端建模适配，仍交由公共 converter。

```python
x_g = input.reshape(n, g, c // g, h * w)
mean = x_g.mean(dim=(2, 3), keepdim=True)
var = ((x_g - mean) ** 2).mean(dim=(2, 3), keepdim=True)
```

[永久来源：VAE GroupNorm](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/bonsai_flux2/modeling_flux2.py#L857-L870)

### 4.4 官方量化方案主要面向其现有 runtime 合同

`bonsai_flux2.py:230–273` 的固定 helper：text FC 为 dynamic W4 block128，embedding lookup 为 W8 channelwise；DiT 非 block FC 默认 W8 channelwise，匹配 `.*TransformerBlock_.*` 的 FC 为 W4 block32。量化后修复全零块 scale，避免 XNNPACK 拒绝。Klein 也调用这些 helper；已独立拆出的 FP16 embedding bin 则不经过图内 EMBEDDING_LOOKUP 量化。

```python
regex=".*TransformerBlock_.*",
operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
num_bits=4,
granularity=qtyping.QuantGranularity.BLOCKWISE_32,
```

[永久来源：量化 recipe](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/bonsai_flux2/bonsai_flux2.py#L230-L273)、[零 scale 修复](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/bonsai_flux2/bonsai_flux2.py#L165-L227)

这些 helper 使用 dynamic config，没有在此路径实现 MT6897 A16W8 activation 校准。量化字符串在 `_should_quantize` 中主要作为开关使用，不能以为它会自动选择与任意字符串同名的 DiT recipe。helper 的正则命中与最终 tensor 位宽仍需 artifact 证据。

## 5. 官方完整生图样例补齐了哪些主机工作

固定 samples commit 下的 `models/bonsai/bonsai_image_4b/python/generate.py` 是 189 行 host loop。它使用已发布的**三图**，没有 torch/diffusers runtime 依赖。

### 5.1 按真实输入名/参数顺序绑定，保持 shape 与 dtype

Graph runner 读取 CompiledModel 的输入元数据，将 `args_<n>` 按位置排序；逐项按 runtime dtype 转换并检查 shape，创建 buffer 后复用。作者强调不能按 shape 认输入：text encoder 的 ids/mask 同 shape，256图像的 img_ids/txt_ids 也同 shape。

```python
details = self.model.get_input_tensor_details(self.sig)
self.inputs = sorted(details.items(), key=lambda kv: argpos(kv[0]))
# ...
a = np.asarray(t, dtype=d["dtype"])
```

[永久来源：Graph runner](https://github.com/google-ai-edge/litert-samples/blob/f9f0e000630e9cdd6d263c4cd80495e4231a2dc8/models/bonsai/bonsai_image_4b/python/generate.py#L34-L73)

这份样例的每图签名为 `serving_default`、位置参数输入名为 `args_0...`，README 提供实际 API 调用和 output shape `[1,256,7680]`。这是该三图 artifact 的证据，不能无检查复制到 pc_convert 自己导出的图。

[永久来源：实际输入输出 API 示例](https://github.com/google-ai-edge/litert-samples/blob/f9f0e000630e9cdd6d263c4cd80495e4231a2dc8/models/bonsai/bonsai_image_4b/README.md#L49-L70)

### 5.2 Tokenizer、positions、seed、scheduler、latent、VAE

host 顺序是：Qwen chat template（关闭 thinking）→固定 256 token 编码→text encoder→构造 image/text 4轴 position IDs→按 seed 采样初始噪声→FlowMatch-Euler 多步→latent BatchNorm affine 与 unpatchify→VAE→RGB/PNG。

`flowmatch_sigmas` 根据 image token 数和 steps 计算 empirical mu，使用 exponential time-shift，再补 sigma=0；循环将每次 DiT 输出速度 v 积分回 latent。

```python
v = dit(lat, embeds, sigmas[k:k + 1], img_ids, txt_ids)
lat = lat + (sigmas[k + 1] - sigmas[k]) * v
```

[永久来源：scheduler](https://github.com/google-ai-edge/litert-samples/blob/f9f0e000630e9cdd6d263c4cd80495e4231a2dc8/models/bonsai/bonsai_image_4b/python/generate.py#L76-L86)、[完整 host 流程](https://github.com/google-ai-edge/litert-samples/blob/f9f0e000630e9cdd6d263c4cd80495e4231a2dc8/models/bonsai/bonsai_image_4b/python/generate.py#L140-L184)

单图 DiT 的调用有**5项输入**：latents、text embeddings、sigma/timestep、img_ids、txt_ids。最后 latent packed-channel affine 必须先于2×2解包；单写一个 `.tflite` 或一个 `.bin` 都没有替代这些数学步骤。

```python
z = lat.astype(np.float32) * bn_scale + bn_shift
z = z.reshape(grid, grid, LATENT_CH, 2, 2)
return z.transpose(2, 0, 3, 1, 4).reshape(1, LATENT_CH, 2 * grid, 2 * grid)
```

[永久来源：latent 解包](https://github.com/google-ai-edge/litert-samples/blob/f9f0e000630e9cdd6d263c4cd80495e4231a2dc8/models/bonsai/bonsai_image_4b/python/generate.py#L89-L98)

VAE 导出还读取其 BatchNorm running variance/mean，将 scale、shift 写进生图 metadata；不能随便用0/1占位值替代已训练模型的常量。`BonsaiFlux2` 的 metadata 包含 seq_len、img_size、packed_ch、prompt_dim、default_steps=4 和 latent affine 参数。

```python
self._latent_bn_scale = (
    torch.sqrt(running_var.float() + vae.bn.eps).cpu().tolist()
)
self._latent_bn_shift = running_mean.float().cpu().tolist()
```

[永久来源：VAE BN 参数](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/bonsai_flux2/bonsai_flux2.py#L518-L526)、[生图 metadata](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/bonsai_flux2/bonsai_flux2.py#L576-L612)

## 6. 证据分级：官方有样例，不等于本地已接通

| 对照项 | 官方已有能力 / 证据 | 对 pc_convert 的正确解释 |
|---|---|---|
| 标准 FLUX pipeline 加载 | 模型注册、model_index 分流、真实权重 strict load | 通用 convert 不会自动调用这层，需明确复用入口或自备兼容 Module |
| DiT 分片 | initial + double×2 + single×4 + final | 本地3输入接口覆盖不了全部分片；必须有前后层和多输出路由 |
| Text encoder | Qwen3 hidden-state 取层、3段编码、FP16 embedding table | 不能把一个预填 text embedding tensor 当 tokenizer/编码器已完成 |
| RoPE | 整体图 ids→FP32三角函数；分片显式cos/sin；GPU样例真替换forward | 独立bin只算产物存在，还需证明图/host消费 |
| Attention/VAE形状 | 4D repeat_kv、RoPE、GroupNorm；SDPA非causal | 框架里有实现不等于本地模型调用了实现 |
| 量化 | 官方动态W4/W8 + 零scale修复 | 不能当作已具备MT6897静态A16W8方案或精度验收 |
| 完整生图 | 3图＋tokenizer＋scheduler＋latent affine/unpatchify＋PNG | 导出DiT子图和AOT成功仍欠完整执行主机链 |
| MT6897 | 官方目标与NeuroPilot版本映射 | 只说明能选择后端目标，不能证明特定DiT算子/shape/精度/性能 |
| LiteRT编译器是否修改 | 上述适配都能定位到模型、wrapper、量化及host层，最后仍调用公共convert | 本地增加CLI/整理FlatBuffer/包装AOT不足以证明修改了compiler本体；结论须看实际diff/来源 |

### 6.1 已从官方来源确认

- 官方 v0.10.0 tag 中确实有 FLUX2-Klein/Bonsai ImageGen 导出，适配内容不依赖对本地 wheel 名字的信任。
- 固定 samples commit 的 README 报告：Mac M4 Max 的 CPU/XNNPACK 路径使用 LiteRT 2.2.0；Apple Metal DiT 路径使用另一个 GPU-shaped 图、其余组件在 CPU。文档明确每个计时行之前先生成了图像，Android没有在该页测量。
- 这些是作者报告的成功记录；本次只核查来源，没有复现实验。

文档的作者测量原句：

> Every row produced the fox image before its times were recorded.

[永久来源：测量范围、版本及限制](https://github.com/google-ai-edge/litert-samples/blob/f9f0e000630e9cdd6d263c4cd80495e4231a2dc8/models/bonsai/bonsai_image_4b/README.md#L78-L88)

### 6.2 仍不能写成“已验证”

- 整个本地 wheel/底层编译器/插件是否与官方发行物一致。已有[四文件来源比对](LiteRT_Torch_0.10.0_四文件对照.md)确认 `_convert/core.py`、`_convert/litert_converter.py`、`model_ext/exportables.py`、`flux2_klein/flux2_klein.py` 与官方 v0.10.0 字节相同；该结论保留在这四个文件范围内。
- pc_convert 导出的真实 DiT 是否保留正确参数、输入语义与所有位置编码，完整 denoising loop 是否能生成合格图像。
- 官方 8 分片 DiT 与本地 artifact 的实际 tensor 命名、regex量化命中、数值误差及设备性能。
- MT6897 上该具体图的编译成功、委派分区、实际 NPU 执行与输出质量。

官方样例的 CPU、Apple GPU 成功记录只能证明对应图/设备/运行链；报告不得把它们换成 MT6897 或本项目记录。

## 7. MT6897 target 的准确含义

当前 [Google MediaTek 页面](https://developers.google.com/edge/litert/next/mediatek) 列出 Dimensity 8300/MT6897，并说明 CompiledModel 支持 AOT 和 on-device compilation；页面更新时间为 2026-05-28，本次读取于 2026-11-10。

LiteRT v2.3.0 固定源码 `litert/python/aot/vendors/mediatek/target.py:36、48–54` 包含 MT6897，且只有 MT6993 映射 v9，其余映射 v8：

```python
MT6897 = "MT6897"
# ...
case SocModel.MT6993:
  return "v9"
case _:
  return "v8"
```

[永久来源：MT6897 target 与 NP 映射](https://github.com/google-ai-edge/LiteRT/blob/4352e261d3f1b3cda00345db5dbd75859d3bf0d3/litert/python/aot/vendors/mediatek/target.py#L24-L54)

**报告用语建议**：“MT6897 是官方公开编译目标；本地正在组合既有转换器与 MediaTek AOT。完整 DiT 的适配、数值和设备执行仍须逐项提供证据。” 不能把列在 SoC 清单上解读成任意 DiT 所有算子、所有 shape、所有量化格式都已有保证。

### 7.1 与 #6886 支持答复的冲突尚未澄清

本次有界复查 [LiteRT issue #6886](https://github.com/google-ai-edge/LiteRT/issues/6886)及其 comments，得到三种不同层级的事实：

1. **产品文档与编译配置**：上面的公开 SoC 清单及 v2.3.0 target 确有 MT6897。
2. **具体复现**：报告者在 Lenovo IdeaTab Pro TB373FU、Android API 34、LiteRT 2.1.0、LiteRT-LM Android 0.10.0，加载 `gemma-4-E2B-it.litertlm`，报告 NPU registration 失败及 MediaTek dispatch SIGSEGV。ROM 属性缺失/命名空间错位是报告者提出的原因；本次没有独立证明这个根因。[原始版本与配置](https://github.com/google-ai-edge/LiteRT/issues/6886#issuecomment-4249449336)
3. **支持答复**：2026-10-15，`aravindmurali22`（GitHub `author_association=CONTRIBUTOR`）答复 MT6897 当前不支持 LiteRT NPU，并建议使用 CPU/GPU。回复措辞是 SoC 范围，**没有限定只对 Gemma4**。[该条原答复](https://github.com/google-ai-edge/LiteRT/issues/6886#issuecomment-6200222696)

支持答复中的关键原文：

> The MediaTek Dimensity 8300 (MT6897) SoC is currently not supported for NPU acceleration in LiteRT.

所以报告必须保留“**官方公开 target/SoC 列表与仓库支持答复存在未澄清冲突**”。可复现事实覆盖原来的版本/ROM/模型组合；它不足以宣告所有 MT6897 DiT 都必然失败。反过来，也不能只引用 SoC 清单而保证这台设备可用，或把原答复缩写成“只有该 Gemma 模型不支持”。

拟定部署线 **PC AOT → Android CompiledModel + MediaTek Dispatch，DiT 主干争取 NPU、text encoder/scheduler/VAE 先 CPU** 能减少首轮变量；其中主干 NPU 仍为待验目标。把其余组件放 CPU 不能解除发生于 NPU 初始化阶段的 dispatch/ROM 兼容风险。

## 8. 对本项目的工程结论

把本地成果定位为**转换/编译流程的项目适配与组装**，按真实调用链明确接上了哪些组件。对生图模型优先复用官方已存在的模型包装、RoPE合同和host计算，把剩余工作收敛到 MT6897 所需的算子/shape/量化与端上执行差异。不要以“重写了LiteRT编译器”解释没有来源证据的能力，也不要以“官方完全不支持DiT”掩盖尚未接入的上游现成模块。

