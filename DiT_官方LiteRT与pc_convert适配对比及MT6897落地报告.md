# DiT：官方 LiteRT 与 pc_convert 的适配边界，以及 MT6897 NPU 生图落地

**源码对照与工程报告 · 2026-11-10**

## 结论：官方已有特定 DiT 适配，本包还没有交付 MT6897 完整生图链

1. **“官方 LiteRT 不支持 DiT”这个概括需要更新。** 官方 LiteRT Torch `v0.10.0` 已有 Bonsai/FLUX.2-klein 模型适配；官方还有完整图像生成示例。支持某个模型导出、在 CPU 上生图、在 MT6897 NPU 上生图，是三个需要分别举证的能力。
2. **本包抽查的四个关键 LiteRT Torch 文件与官方同版本逐字节相同。** 覆盖转换入口、转换桥接、模型分发和 FLUX.2-klein 导出。当前能定位的项目适配主要在 `convert_dit.py` 等外围脚本；没有证据把这四个文件里的 DiT 能力归为作者私改。该结论仅覆盖抽查文件。
3. **本地增加了固定输入模板、导出兼容处理、两类 TFLite 图规整、量化编排、MediaTek AOT 和设备诊断。** 缺口集中在模型专属接口、真实校准、多输入运行器和完整生图循环；RoPE 文件生成了，消费路径还空着。
4. **MT6897 需要重新目标编译，并先确认目标设备能初始化/执行 NPU。** 官方 SoC 清单与 issue 中的支持答复存在未澄清冲突，详见第 6.1 节。改 `--soc` 只能选择编译目标；现有 manifest/部署脚本中仍有 MT6991、原作者容器和单输入探针的假设。
5. **最短完整路线：PC AOT，端上 LiteRT CompiledModel + MediaTek Dispatch；DiT 去噪主干用 NPU，文本编码、scheduler、VAE 首版用 CPU。** 验收必须拿到端上从 prompt 到 PNG 的完整结果。先跑一个 block 是必要时的定位手段，不能把任务缩成“一个 block 跑通”。

本次实际完成：源码阅读、四文件直接内容比较、官方资料核对、当前项目证据盘点。没有安装依赖、下载模型权重、执行转换或连接 MT6897。以下明确区分事实与待实施方案。

## 1. 对照基线与证据用语

### 1.1 固定版本，避免拿不同年代的“官方”互相比较

| 对象 | 本报告采用的基线 |
|---|---|
| 本地项目 | `D:\misc\pc_convert\pc_convert`，本次分析时的文件内容 |
| 随包工具链 | LiteRT 2.3.0；LiteRT Torch 0.10.0；AI Edge Quantizer 0.10.1 |
| 官方 Torch | `v0.10.0`，commit `90cbbaa61ab3f029f1273b108909a5b4d9cc2c12` |
| 官方 LiteRT AOT | `v2.3.0`，commit `4352e261d3f1b3cda00345db5dbd75859d3bf0d3` |
| 官方样例 | litert-samples，commit `f9f0e000630e9cdd6d263c4cd80495e4231a2dc8` |
| 当前硬件证据 | 原 README 主要描述 MT6991 小模型及另一款 App 的历史实验；本包 MT6897 DiT 生图未验证 |

本文使用四种证据标签：

- **源码已确认**：本次直接读到了对应实现，或完成了文件内容比较。
- **官方已有**：官方源码/文档包含该能力；本次没有代替官方复跑其样例。
- **原作者记录**：原包 README 声称曾执行；当前交付目录没有相应原始结果供复查。
- **待验证/待实现**：需要真正模型、编译器、目标设备或新增代码才能完成。

### 1.2 四文件来源对比，能证明到哪一层

比较方法是读取 wheel 指定成员，与官方同 tag 原文件直接比较 bytes，再生成文本 diff；四份 diff 均为空。没有安装/执行 wheel，也没有做整个依赖包的来源审计。

| wheel 内成员，前缀均为 `litert_torch/` | 行数 | 结果 |
|---|---:|---|
| `_convert/core.py` | 310 | 与官方相同 |
| `_convert/litert_converter.py` | 258 | 与官方相同 |
| `generative/export_hf/model_ext/exportables.py` | 224 | 与官方相同 |
| `generative/export_hf/model_ext/flux2_klein/flux2_klein.py` | 315 | 与官方相同 |

完整比较说明与原文在 [四文件源码对照](evidence/LiteRT_Torch_0.10.0_四文件对照.md)，直接比较记录在 [comparison.json](evidence/litert_torch_0_10_0_comparison.json)。

**应当怎么讲：**项目做了模型/工具链适配；抽查的转换入口与 FLUX 适配沿用官方内容；`litert_converter` 二进制、MediaTek 插件和整套 wheel 的来源没有由四文件抽查证明。三层分开讲，结论就不会跑偏。

## 2. 官方已经提供哪些 DiT 能力

### 2.1 官方模型注册已包含 FLUX.2-klein

官方 `litert_torch/generative/export_hf/model_ext/exportables.py:217–224`：

```python
def get_image_gen_model_cls(model_type: str):
  """Gets ImageGen model class by model type or pipeline class name."""
  if model_type == 'bonsai_flux2':
    return bonsai_flux2_lib.BonsaiFlux2
  elif model_type in ('flux2_klein', 'Flux2KleinPipeline'):
    return flux2_klein_lib.Flux2Klein
  else:
    raise ValueError(f'Unsupported ImageGen model type: {model_type}')
```

这些分支在本地 wheel 与官方文件中完全一致。因此，“包里出现 FLUX2 就说明作者给 LiteRT 内核加了 DiT”没有源码依据。[官方模型分发源码](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/exportables.py#L217-L224)

### 2.2 官方也需要模型适配，通用 convert 不会自动理解所有 DiT

官方 `flux2_klein/flux2_klein.py:55–61`：

```python
class Flux2Klein(bonsai_flux2.BonsaiFlux2):
  """FLUX.2-klein multi-shard text-to-image model."""

  _TEXT_EMBED_FILENAME = "qwen_embed_fp16.bin"
  _NUM_TE_SHARDS = 3
  _NUM_DIT_DOUBLE_SHARDS = 2
  _NUM_DIT_SINGLE_SHARDS = 4
```

该导出器拆出 3 个文本编码器分片、独立 FP16 embedding、8 个 DiT 分片（initial + 2 double + 4 single + final）和 VAE。分片涉及真实的权重映射、输入合同和运行调度；把单图文件切成八段做不到这些。[官方分片导出实现](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/flux2_klein/flux2_klein.py)

同文件 `189–193`，模型适配完成后仍调用公共转换接口：

```python
      fp32_path = os.path.join(output_dir, f"{stem}_fp32.tflite")
      litert_torch.convert(mod.eval(), inputs).export(fp32_path)
      if quantize and self._should_quantize(export_config):
        int4_path = os.path.join(output_dir, f"{stem}_int4b32.tflite")
        bonsai_flux2.quantize_dit(fp32_path, int4_path)
```

**技术路线：模型重表达/包装与拆分 → PyTorch 导出 → LiteRT Torch lowering/转换 → TFLite → 按后端量化与编译。** 本项目没有另造一套 DiT 编译理论。官方 INT4 分片产物也不能直接视作 MT6897 可接受的 NPU 配方；需核查该目标的算子、量化粒度、激活与 bias 类型。

### 2.3 官方完整生图样例与上述分片导出是两份证据

官方 Bonsai Image 4B / FLUX.2-klein 图像生成样例采用文本编码器、DiT、VAE 三图及 host 侧 FlowMatch-Euler 采样循环，提供 CPU/XNNPACK 等运行说明。它证明官方工程体系内已经有特定 DiT 的完整生图路径；本报告没有在本机复跑。不能将样例的运行结果转借给八分片导出，更不能转借给 MT6897 NPU。[固定版本的官方样例说明](https://github.com/google-ai-edge/litert-samples/blob/f9f0e000630e9cdd6d263c4cd80495e4231a2dc8/samples/litert/image_generation/README.md)

旧 Stable Diffusion 1.5 示例主要涉及 UNet，单独拿它作 DiT 支持证明会混淆架构。[官方 SD 示例](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/examples/stable_diffusion/README.md)

### 2.4 真正的接口差异：三输入只覆盖官方分片的第一片

令 `N=(S/16)^2` 为图像 token 数，`L` 为文本 token 数，`J=N+L`，`D=head_dim×num_heads`，`C/E` 分别取模型配置的输入通道/文本条件维度。官方默认 `L=256`；512×512 时 `N=1024、J=1280`。下表是源代码的 **forward 参数顺序和 sample shape**，本次没有生成 flatbuffer 来确认实际运行时名称。

| 官方分片 | 输入顺序与关键 shape | 输出/接线 |
|---|---|---|
| DiT initial | hidden_states `[1,N,C]`、encoder_hidden_states `[1,L,E]`、timestep `[1]` | image、text、mod_img、mod_txt、mod_single、temb，6 个输出 |
| double ×2 | image `[1,N,D]`、text `[1,L,D]`、cos/sin 各 `[1,J,1,head_dim]`、mod_img/mod_txt 各 `[1,6D]` | image、text；真实消费 cos/sin |
| single ×4 | joint `[1,J,D]`、cos、sin、mod_single `[1,3D]` | joint |
| final | joint `[1,J,D]`、temb `[1,D]` | 移除文本 token 后的图像预测 |

官方 `flux2_klein/flux2_klein.py:171–179` 及 double sample 输入段 `216–223`，节选：

```python
    grid = image_size // 16
    img_tokens = grid * grid
    joint_tokens = seq_len + img_tokens
# double_inputs 中的 cos / sin 样例
        torch.ones(1, joint_tokens, 1, head_dim, dtype=torch.float32),
        torch.zeros(1, joint_tokens, 1, head_dim, dtype=torch.float32),
```

[完整分片输入源码](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/flux2_klein/flux2_klein.py#L145-L262)

官方 `flux2_klein/modeling.py:84–90` 还明确了时间输入的单位转换，节选：

```python
    temb = self.time_guidance_embed(timestep * 1000.0, None)
# ...
    return image, text, mod_img, mod_txt, mod_single, temb
```

[initial 的真实 forward](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/flux2_klein/modeling.py#L84-L90)。把本地固定 `500` 生搬到这种归一化时间入口，数值语义就错了。官方三图样例的整体 DiT 则有 **5 路输入**，也不符合本地三输入假设。

### 2.5 官方已经做过的模型改写，值得直接复用

- **权重与模型结构：**按 Diffusers 的模块/参数层级重表达 DiT/VAE，支持单文件与分片 safetensors，按配置构建并严格装载；用 `model_index.json` 识别多组件 pipeline。
- **文本编码：**Qwen3 中间层特征组合、独立 embedding 和分片边界；eager attention 与 ≤4D 的 repeat_kv 写法，避免某些广播表达。
- **DiT/RoPE：**FP32 频率计算、4D 旋转及非因果 SDPA；分片 forward 将 `(cos,sin)` 真正送入 block。官方 GPU 专用样例还将预计算位置常量替换进 forward，区别于仅输出附件。
- **VAE：**4D GroupNorm 表达，保存 latent BatchNorm 的 scale/shift；host 在 VAE 前执行 affine 与正确 unpatchify。
- **量化：**官方 helper 的动态 W4/W8、block32/block128 等方案及 XNNPACK 零 scale 修复，是其现有后端合同；不等于 MT6897 静态 W8A16 校准已解决。

这些适配的逐段原文、永久链接和行号整理在 [官方 DiT 适配源码索引](evidence/官方DiT适配源码索引.md)。它们位于模型/导出/host 层，最后仍调用公共 converter。当前本地脚本没有完整接入这些上游模块。

官方三图样例 `models/bonsai/bonsai_image_4b/python/generate.py:140–184` 中的核心迭代，节选：

```python
        v = dit(lat, embeds, sigmas[k:k + 1], img_ids, txt_ids)
        lat = lat + (sigmas[k + 1] - sigmas[k]) * v
```

这里把 DiT 预测的速度按 sigma 差积分回 latent，下一轮再消费它；本地的 32 个独立随机样本没有这一状态更新。[完整官方 host 流程](https://github.com/google-ai-edge/litert-samples/blob/f9f0e000630e9cdd6d263c4cd80495e4231a2dc8/models/bonsai/bonsai_image_4b/python/generate.py#L140-L184)

## 3. 官方与本地逐层对比

| 层 | 官方已有 | pc_convert 具体做了什么 | 当前断口 |
|---|---|---|---|
| 架构与权重 | 特定模型 adapter、模块包装/权重映射 | 接受用户 `build()`、指定类或 Hub `.transformer` | 未调用已有 FLUX 专项 export；未提供默认 `model_def.py` |
| 图导出 | Torch export、转换 passes、TFLite 导出 | 固定三输入，兼容导出 API 与文件落盘 | 不覆盖任意 DiT 的参数顺序、packing、mask、guidance 等 |
| 图规整 | 通用转换/算子表示 | opcode 回填；老式 RESHAPE 补 shape tensor | 无通用 attention/RoPE/动态 shape 改写；RESHAPE 只遍历首子图 |
| 位置编码 | 模型 adapter 中按真实公式处理 | 额外生成二维 RoPE `.bin` | 没有注入/加载到导出图或运行器 |
| 量化 | Quantizer、模型专属方案 | 配方选择、校准调用、量化统计、失败回退 | 真轨迹条件未收全；回退产物不能冒充原定静态配方 |
| 厂商编译 | MediaTek Target、AOT plugin、SDK | 按 `--soc` 调用 AOT、输出报告/产物 | MT6897 真实 DiT 分区与编译未验证；失败可能仍返回 0 |
| 设备执行 | CompiledModel/Dispatch、TensorBuffer/Run | 创建探针；直接 Neuron 单 I/O 小模型探针 | 缺正式多 I/O runner；缺图之间调度与资源复用 |
| 完整生图 | 特定模型完整样例 | 清单、推送、诊断辅助 | tokenizer/text encoder、scheduler、CFG、VAE、PNG 输出未接通 |
| 交付证据 | 样例/文档与具体平台记录 | README 有 MT6991 小图历史记录 | 无当前真实 DiT → MT6897 → 图片的原始证据 |

## 4. 结合本地代码看适配与断口

以下文件统一位于 `D:\misc\pc_convert\pc_convert`。节选保留原文；省略号只标识不连续摘录。

### 4.1 加载 Hub transformer，没有自动走官方 FLUX 导出器

`D:\misc\pc_convert\pc_convert\convert_dit.py:239–245,265–268`：

```python
    if args.from_hub:
        try:
            from diffusers import DiffusionPipeline
        except ImportError:
            raise SystemExit("--from-hub 需要 diffusers：pip install diffusers")
        log(f"  从 HF 取 {args.from_hub} 的 transformer")
        model = DiffusionPipeline.from_pretrained(args.from_hub).transformer
# ...
    sd = _read_state_dict(src)
    _inject(model, sd, args.allow_partial)
    model.eval()
    return model
```

Hub 分支之后仍读取位置参数 `source` 并注入权重。所以当前 `--from-hub` 并非只给一个模型 ID 就能完成的入口；还需符合 `_read_state_dict` 的文件及名称映射。普通路径要求用户补 `model_def.py` 的 `build()`，当前包未提供该文件。

**要补：**选定一个 checkpoint/revision，明确组件清单和权重加载方式，复用对应官方 adapter 或编写等价 wrapper；逐项核对完整权重覆盖。`--allow-partial` 只能用于诊断，不得带着未对齐的参数进入图像质量验收。

### 4.2 三输入模板带着特定模型假设

`D:\misc\pc_convert\pc_convert\convert_dit.py:318–324,342,345–346`：

```python
    L = args.seq_len
    lh, lw = args.height // 8, args.width // 8
    hs = torch.randn(1, L, args.hidden_dim, dtype=torch.float32)
    tb = torch.randn(1, args.text_len, args.text_dim, dtype=torch.float32)
    t = torch.full((1,), 500.0, dtype=torch.float32)
    return ({"hidden_states": hs, "encoder_hidden_states": tb, "timestep": t},
            (args.latent_ch, lh, lw, L))
# ...
    sample_args = tuple(inputs.values())
# ...
        conv_result = litert_torch.convert(model, sample_args, strict_export=False,
                                           output_file_path=str(out_path))
```

默认 512×512，`L=(H/16)×(W/16)=1024`，图像输入 `[1,1024,2048]`，文本输入 `[1,77,2048]`，timestep `[1]`。这是该脚本的示例合同，无法代表所有 DiT。原模型可能接受 `[B,C,H,W]` latent、尚未投影的 packed patch、多个位置 ID、pooled 条件或 guidance。

尤其注意：字典被转为 **tuple** 后按位置调用；参数名字不会自动匹配原 `forward`。`strict_export=False` 也不会替开发者修正模型语义。

**要补：**显式定义 input/output 名称、顺序、shape、dtype、layout、量化参数和含义；将 patchify/unpatchify、位置编码、条件输入与真实模型绑定。静态 shape 固定长度，timestep 的数值仍应在每轮变化，通常无需每一步重编。

### 4.3 导出文件落盘处理是有价值的 API 兼容修补

`D:\misc\pc_convert\pc_convert\convert_dit.py:351–355`：

```python
        if not out_path.is_file():
            content = getattr(conv_result, "model_content", None)
            if callable(content):
                out_path.write_bytes(content())
                log(f"  （convert() 未落盘，已从返回的 LiteRTModel 写出）")
```

它处理了当前默认转换分支只返回内存模型的交接问题。这属于外围接口兼容；算子 lowering 能力仍来自 LiteRT Torch 与其转换器依赖。序列化出文件与数值正确要分两步确认。

### 4.4 两类图规整，覆盖面很具体

`D:\misc\pc_convert\pc_convert\convert_dit.py:429–433`：

```python
    n_backfill = 0
    for oc in m.operatorCodes:
        if not getattr(oc, "builtinCode", 0) and getattr(oc, "deprecatedBuiltinCode", 0):
            oc.builtinCode = oc.deprecatedBuiltinCode
            n_backfill += 1
```

这给受影响的旧式算子编码补字段，解决当前后续读取器的兼容问题；并非所有合法 TFLite 都必须经过这种补丁。

同文件 `444–449,461–463`：

```python
            opt = op.builtinOptions
            shape = list(getattr(opt, "newShape", None) or [])
            if not shape:
                continue          # option 里也没有 shape，无从物化，留给上游
            raw = np.array(shape, dtype=np.int32).tobytes()
            buf = S.BufferT(); buf.data = list(raw)
# ...
            op.inputs = [ins[0], len(sg.tensors) - 1] + \
                [i for i in (op.inputs or [])[1:] if i is not None and i >= 0]
            n_reshape += 1
```

它把单输入 RESHAPE 的 options 形状物化为第二个 INT32 输入，供后续编译消费。当前只遍历第一个子图的 RESHAPE；没有处理任意不支持算子、动态维度和注意力融合。

同文件 `475–478`：

```python
    # 改动后必须验证输出等价：round-trip 丢字段的话，宁可不改
    if not _verify_equivalence(src, out):
        log("  !! 规整化后模型输出与原模型不一致 —— 放弃改动，用原模型继续")
        return src, dict(stats, reverted=True)
```

失败回原图能避免交出已知错误改写，但也可能保留原本阻止后端编译的表示问题。正式流水线应记录该状态，由后续编译结果决定能否继续，不能看到“继续执行”就当修复生效。

### 4.5 RoPE 只做了文件，没有接线

`D:\misc\pc_convert\pc_convert\convert_dit.py:228–230,1273–1275,1284`：

```python
    name = f"rope_{grid_h}x{grid_w}.bin"
    (out_dir / name).write_bytes(tbl.tobytes())
    return name, tbl.nbytes, grid_h, grid_w
# ...
            gh, gw = args.height // 16, args.width // 16
            name, nbytes, sy, sx = build_rope_tables(gh, gw, args.head_dim, out_dir)
            log(f"  ✓ {name}  {nbytes} B  (grid {sy}x{sx}, L={sy * sx}, head_dim={args.head_dim})")
# ...
            artifacts.append(out_dir / name)
```

后面的模型加载与 `to_tflite()` 没有接收此表；设备程序也没有绑定它。`compile_plan.json` 出现文件名只说明有附件，无法改变计算图。

**要补：**按所选模型准确实现 RoPE 的轴、base、位置编号、维度排列和精度，选择图常量折叠或真实输入端口；随后对照参考输出。文件大小相同不能证明表内容相同。源码日志中“APU 无三角函数所以必须烘表”的说法不能当作所有硬件/编译版本的事实，实际改写由模型与编译支持决定。

### 4.6 “真实校准”仍混入随机文本和固定时间步

`D:\misc\pc_convert\pc_convert\convert_dit.py:799–808`：

```python
    for f in files[:256]:
        hs = np.load(f).astype(np.float32)
        if hs.ndim == 2:
            hs = hs[None]
        if hs.shape != (1, L, D):
            raise SystemExit(f"{f.name} shape={hs.shape}，期望 (1,{L},{D})")
        samples.append({
            "hidden_states": hs,
            "encoder_hidden_states": rng.standard_normal((1, T, TD)).astype(np.float32),
            "timestep": np.array([500.0], dtype=np.float32)})
```

当前只从文件恢复了 hidden states；与它对应的文本条件和 timestep 已经丢失。DiT 不同噪声阶段的激活分布会变化，这样校准得到的 scale/zero point 不能视为覆盖真实运行分布。随机校准固定 seed 只提高重复性，不会让随机数据变得有代表性。

**要补：**从参考 pipeline 同一次前向中保存完整命名输入，覆盖多个 prompt/seed 及去噪初、中、后期；模型要求的 sigma、mask、位置 ID、guidance/CFG 条件都要一起保存。建议每条记录用命名张量集合加元数据，禁止单独 latent 配随机条件。`--calib-steps=32` 表示当前采样记录数量，不控制图片的 32 步采样。

同文件 `906–910` 的回退值得单独警惕：

```python
            log(f"  校准/量化失败（{type(e).__name__}: {str(e)[:120]}）→ 退化 dynamic_wi8_afp32 先保产物")
            qt = quantizer.Quantizer(str(model_tflite))
            qt.load_quantization_recipe(rc.dynamic_wi8_afp32())
            qm = qt.quantize()
            out = out_dir / f"dit_{args.recipe}_fallback_w8dyn.tflite"
```

请求 W8A16，最终可能拿到权重 INT8、激活 FP32 的候选。后端兼容性和数值合同都变了。生产转换应让失败阶段明确失败；诊断模式可保留回退，但 manifest 必须记 **实际** 配方，不能只沿用请求值。

### 4.7 `--soc MT6897` 已接 AOT；目标之外的交接仍需修正

`D:\misc\pc_convert\pc_convert\convert_dit.py:1071–1085`，节选：

```python
    from ai_edge_litert.aot.aot_compile import aot_compile
    from ai_edge_litert.aot.core import aot_types
    from ai_edge_litert.aot.vendors.mediatek import target as T

    soc = getattr(T.SocModel, args.soc, None)
# ...
    tgt = T.Target(soc)
    log(f"  target={tgt!r}  np={tgt.recommended_np_version}  {tgt.flatten()}")

    try:
        res = aot_compile(str(tflite), output_dir=str(out_dir),
                          config=[aot_types.CompilationConfig(target=tgt)],
                          keep_going=False)
```

这是复用官方 MediaTek AOT 接口。本地包为 MT6897 提供 target，推荐 SDK 大版本为 v8；需生成该目标的新产物，并保留分区报告、编译配置和完整 AOT `.tflite`。具体 SDK/adapter/runtime 小版本还需与设备兼容。

同文件 `1090–1093,1321,1347`：

```python
    except Exception as e:
        log(f"  ✗ aot_compile 失败：{type(e).__name__}: {e}")
        log(f"     {str(e)[:400]}")
        return None
# ...
        aot_partition(quant, args, out_dir, artifacts)
# ...
    return 0
```

主程序未消费 AOT 返回状态。生成量化文件、写出计划、进程返回 0，三件事合在一起仍不能证明 AOT 成功。

同文件 `1156–1158`：

```python
            "vendor_lib": "/vendor/lib64/libneuron_runtime.8.so -> mt6991/",
            "acceleration_gate": "LiteRtCompiledModelIsFullyAccelerated() 必须为 true，"
                                 "否则静默回 CPU",
```

计划中还有硬编码 MT6991 路径，以及对加速布尔值过强的解释。应根据本次目标和运行器生成配置；执行证据还要包括实际 backend、Run、输出和设备活动。`IsFullyAccelerated` 单独不能证明跑在指定 NPU。

## 5. 设备程序离完整 DiT 运行还差什么

### 5.1 LiteRT checker 只到创建/查询，没有 Run

`D:\misc\pc_convert\pc_convert\litert_npu_check.cpp:97–104,121–123`：

```cpp
    LiteRtCompiledModel *cm = NULL;
    r = createCompiled(env, model, opts, &cm);
    printf("[4] LiteRtCreateCompiledModel -> %d cm=%p\n", r, (void*)cm);
    if (r || !cm) { destroyOpts(opts); destroyEnv(env); return 7; }

    // 5) 关键判据
    bool fully = false;
    r = isFully(cm, &fully);
// ...
    if (destroyCompiled) destroyCompiled(cm);
    destroyOpts(opts); destroyEnv(env);
    return 0;
```

没有读取模型签名、创建 TensorBuffer、绑定输入、提交 Run 和读取输出；查询失败或 `fully=false` 也仍可能返回 0。它可以作为版本限定的装载探针，不能当推理引擎。新 runner 应使用配套发行版的正式头文件，替换当前手写 ABI/option ID。

### 5.2 Neuron 探针有 compute，接口仅支持单输入/单输出

`D:\misc\pc_convert\pc_convert\aot_on_apu.cpp:165–170`：

```cpp
        r = fp_sin(ex, 0, NULL, Xin, (size_t)nin * inEb);
        if (r) { say("  setInput -> %d (第 %d 个)", r, s); return 7; }
        r = fp_sout(ex, 0, NULL, Ymdla, (size_t)nout * outEb);
        if (r) { say("  setOutput -> %d", r); return 7; }
        r = fp_cmp(ex);
        if (r) { say("  compute -> %d (第 %d 个)", r, s); return 7; }
```

两个 index 都固定为 0。`nin/nout` 可以描述不同单张量大小，但不能自动表达 DiT 的多输入合同。前面的加载路径是 `restoreFromCompiledNetwork`，恢复已有编译网络；当前文件没有 README 所描述的 `Compilation_finish` 现场编译流程。

同文件 `15,143–147`：

```cpp
#define NSAMP 32
// ...
    for (int s = 0; s < NSAMP; s++) {
        void *Xin = malloc((size_t)nin * inEb);
        void *Ymdla = malloc((size_t)nout * outEb);
        memset(Xin, 0, (size_t)nin * inEb);
        memset(Ymdla, 0, (size_t)nout * outEb);
```

32 次读取独立的随机基准输入，没有 latent 反馈、时间步序列、scheduler、CFG 或 VAE。它的 32 次计算与 32 步扩散采样没有对应关系。逐轮分配还没有逐轮释放，放大到图像张量会浪费内存；正式 runner 应在循环外建立模型和可复用 buffers。

### 5.3 基准、字节码提取、部署都带小模型/原环境假设

`D:\misc\pc_convert\pc_convert\apu_baseline.py:74`：

```python
    rd, od = it.get_input_details()[0], it.get_output_details()[0]
```

CPU 参考路径也只处理第一路输入输出。其后备模拟器仅有 FC/ReLU 分支，未知算子没有通用实现，不能用来判断真实 DiT 的数值正确性。

`D:\misc\pc_convert\pc_convert\verify_on_apu.sh:68–72`：

```python
m = fu.read_model(Path(src)); sg = m.subgraphs[0]
co = bytes(bytearray(sg.operators[0].customOptions))
off = size = None
for i in range(len(co) - 16):
    a = struct.unpack_from('<Q', co, i)[0]; b = struct.unpack_from('<Q', co, i + 8)[0]
```

这条启发式提取路径假设首子图/首算子包含目标字节码。多分区和 CPU/NPU 混合图需要完整调度，正式路线应保留并加载完整 AOT `.tflite`。

`D:\misc\pc_convert\pc_convert\install_on_device.sh:14,23`：

```bash
if [ -n "$SER" ]; then ADB=(adb -s "$SER"); else ADB=(adb); fi
# ...
adb(){ "${ADB[@]}" "$@"; }
```

包装函数会再解析到自己，形成递归；可用 `command "${ADB[@]}" "$@"` 或改函数名修复。该脚本本身只推送文件，也没有安装运行器和启动图片生成。`litert_stage.sh` 另依赖原作者 root 容器、特定第三方 APK 路径；正式集成应使用合法可分发、版本配套的官方 runtime/dispatch。

错误状态还应一并接通：AOT/静态量化失败、精度超阈、query 失败、compute 失败、push 失败都需要明确传递。`verify_on_apu.sh` 只有 `set -uo pipefail`，没有显式拦住部分失败链；使用每次独立输出目录可避免旧 `.dla` 被误消费。本轮仅记录这些问题，没有修改脚本。

## 6. 把真实 DiT 搬到 MT6897：推荐实施路线

### 6.1 先查设备入口：公开支持信息存在冲突

官方 [MediaTek 页面](https://developers.google.com/edge/litert/next/mediatek)列有 Dimensity 8300/MT6897；LiteRT v2.3.0 的 `target.py:36,48–54` 也有对应枚举和 v8 映射，节选：

```python
  MT6897 = "MT6897"
# ...
      case SocModel.MT6993:
        return "v9"
      case _:
        return "v8"
```

[固定版本 target 源码](https://github.com/google-ai-edge/LiteRT/blob/4352e261d3f1b3cda00345db5dbd75859d3bf0d3/litert/python/aot/vendors/mediatek/target.py#L24-L54)

但 [LiteRT issue #6886](https://github.com/google-ai-edge/LiteRT/issues/6886) 中，2026-10-15 的 contributor 支持答复称 MT6897 当前不支持 LiteRT NPU，措辞针对 SoC，未限定某个模型。原复现为 Lenovo IdeaTab Pro TB373FU、Android API 34、LiteRT 2.1.0、LiteRT-LM 0.10.0、Gemma-4-E2B-it，涉及 NPU registration 失败与 Dispatch 崩溃。报告者对 ROM 属性的根因解释，本次没有独立验证。[原始复现配置](https://github.com/google-ai-edge/LiteRT/issues/6886#issuecomment-4249449336)、[支持答复](https://github.com/google-ai-edge/LiteRT/issues/6886#issuecomment-6200222696)

**工程决定：先用目标 MT6897/ROM 与计划采用的配套 runtime/Dispatch，完成小图的初始化和实际 NPU Run，再投入大型 DiT 编译。** 这是设备兼容门槛，随后仍须完成真实模型与整图验收。若此处失败，记录完整版本/错误交给 runtime/厂商兼容性负责人，取得可用组合后继续 NPU 迁移；模型前端、CPU 参考与校准可独立推进。把文本编码/VAE 放到 CPU 无法修复 NPU 初始化失败。

这一冲突既不允许保证“MT6897 肯定能跑”，也不足以宣布所有 MT6897 DiT 必定失败。本文 NPU 路线是明确的实施目标，尚无目标设备运行证据。

### 6.2 先定模型合同，再选量化；不要从现有默认参数倒推模型

**建议以官方已适配的 FLUX.2-klein/Bonsai 路径作为源码参考起点，首版固定 batch=1 和一个原生支持的分辨率。** 官方现成实现能减少权重映射、RoPE 和 scheduler 的猜测；4B 级模型仍需先算清目标设备内存，不能因为有示例就宣称 MT6897 足够装下。

以 4B 参数粗估，仅权重本体就约为 FP16 8 GB、INT8 4 GB、INT4 2 GB（十进制，未计 scale、对齐、文本编码器、VAE、激活与编译工作区）。最终必须从选定 checkpoint 的真实张量和编译器内存报告计算。分片能管理生命周期，但不能凭空消除权重和跨片传输成本。

首轮保持浮点参考，静态量化先考察 **W8A16** 的可编译性和画质；敏感归一化/调制/输出路径按误差定位保留精度。随后再压缩权重或降低激活位宽。官方 CPU 用的 INT4 block 方案不应未经检查直接当 MT6897 最优方案。

### 6.3 七件交付物，少一件都还没到“端上生图”

| 顺序 | 要补的交付物 | 完成条件 |
|---|---|---|
| 1 | 明确 checkpoint/revision、组件配置、adapter、完整 I/O 合同 | 权重无缺失；packing、RoPE、条件和预测参数化与原 pipeline 一致 |
| 2 | 固定 shape 浮点图与真实输入参考 | 从真实权重导出；初/中/后期 timestep 的输出与原模型对齐 |
| 3 | 完整轨迹校准记录、实际量化配方、量化结果 | 同一前向的全部输入共同保存；检查量化误差及完整采样质量 |
| 4 | MT6897 AOT 图和分区/内存报告 | 明确委派主干与 CPU 边界；记录失败算子；不能改名复用 MT6991 图 |
| 5 | Android arm64 正式 LiteRT runner | 正式 API、全部 I/O、shape/dtype/量化匹配、buffer 复用、实际 Run |
| 6 | 本地 tokenizer/text encoder、scheduler、CFG、VAE 和图像输出 | prompt+seed 在设备上生成 PNG；CPU 组件也在设备本地执行 |
| 7 | 可重放的 MT6897 证据包 | 设备/ROM、配置、执行日志、中间 latent、最终图片、时延/内存可对应 |

不要先让所有组件上 NPU。文本编码通常每次 prompt 执行一次，DiT 重复多次；先把去噪主干放上 NPU，收益路径清楚，也容易定位每一层误差。首次正式验收须跑完选定模型的整条去噪主干；CPU/NPU 混合边界及其覆盖要如实列出。

### 6.4 运行循环应该长这样

以下是**待实现伪代码**，用于解释缺失合同，不是本包已经提供的 API：

```python
pipeline = load_exact_components_and_configs()
condition = pipeline.text_encoder_cpu(pipeline.tokenizer(prompt))
latent = pipeline.initial_noise(seed, fixed_shape)
denoiser = load_mt6897_aot_compiled_model_once()
buffers = allocate_all_signature_buffers_once(denoiser)

for timestep in pipeline.scheduler.timesteps:
    x = pipeline.scheduler.scale_model_input(latent, timestep)
    inputs = pipeline.pack_all_dit_inputs(x, condition, timestep)
    prediction = run_all_dit_shards_and_read_outputs(denoiser, buffers, inputs)
    prediction = pipeline.apply_guidance_if_required(prediction)
    latent = pipeline.scheduler.step(prediction, timestep, latent)

pixels = pipeline.vae_decode_cpu_with_exact_scaling(latent)
save_png(pipeline.postprocess(pixels))
```

需要落实的细节：

- **时间步和输出语义：**模型预测 epsilon、v 还是 flow；时间输入是训练 timestep、归一化时间还是 sigma，都从原配置取值，不能统一塞 `500`。
- **CFG：**按模型选择两次条件/无条件前向、固定 batch=2 合并前向，或蒸馏模型自己的 guidance 机制；不能凭别的 App 的 `cfg=1` 删掉逻辑。
- **分片：**每个 shard 的全部输出送到正确的下一片输入；处理共享条件、位置常量、dtype/量化边界及生命周期。
- **VAE：**latent 通道、scale/shift、packing、空间缩放、输出像素范围逐项对齐。主干输出数值“差不多”仍可能解成花图。
- **缓存：**复用 compiled model、buffers 和可证明不变的文本条件/位置常量。DiT 图像 latent 每步变化，不能照搬自回归 LLM 的旧 KV 复用；`--seq-len` 控制 token 数，也不是 APU L1 容量旋钮。

### 6.5 算子不被 MT6897 接受时，修实际阻塞层

先用真实编译报告定位 attention、归一化、调制、动态 reshape、量化轴或 dtype 的具体拒绝点。优先采用语义等价且后端可接受的分解/静态化，或保留少量 CPU 边界；再根据拷贝与内存成本决定是否重划分子图。当前包的 opcode/RESHAPE 补丁仅解决两种表示问题，不能替代这项模型工作。

若关键去噪块不能进入 NPU，就把“该模型/配置的关键块未支持”写成阻塞项；换一份有依据的图表达或量化配置继续验证。不能以创建成功、空分区报告或 CPU 正常输出替代 NPU 目标。

## 7. 已验证、原作者记录与待验证清单

### 7.1 本次已经完成的核对

| 项目 | 状态与范围 |
|---|---|
| 四个 LiteRT Torch 文件与官方 v0.10.0 对照 | **已完成直接内容比较**，四文件相同 |
| 官方模型注册/FLUX 分片实现 | **源码已确认**；不能推及任意 DiT |
| 本地三输入、RoPE 附件、图规整、校准与回退 | **源码已确认**；报告给出具体调用及断口 |
| AOT target 接线、返回码与 MT6991 残留 | **源码已确认**；未执行 AOT |
| checker 无 Run、Neuron 单 I/O 和 32 独立样本 | **源码已确认**；未执行设备程序 |
| 当前项目原始证据盘点 | 未找到 README 引用的模型产物、输入/输出、设备 trace 或生成图片 |

### 7.2 原作者记录保留，但不能升级成当前 DiT/MT6897 实测

`D:\misc\pc_convert\pc_convert\README.md:322–324`：

```text
实测结果（本机，5 算子 MLP）：
`Subgraph 0 fully compiled: 5 / 5 ops offloaded to 1 partitions`，
产物 `dit_npu_MediaTek_MT6991.tflite` 里 `DISPATCH_OP` + `Partition_0` +
```

文件叫 `dit_npu`，计算图仍是小 MLP。后文还记录了该环境 w8a8 余弦约 0.999998、w8a16 约 1.0 及驱动调用等历史结果。当前解包目录没有对应 `aot_report.txt`、`aot_verify.txt`、`apu.strace`、`ref_in.bin/ref_out.bin`、转换后 `.tflite/.dla` 或图片，无法在本次复核这些数值。

同 README `661–663` 明确保留真实 PyTorch 导出的未验证状态：

```text
- **PyTorch → .tflite 导出段没跑过真 torch**，只用桩验到 `import litert_torch`。
  `convert()` 签名从 `litert_torch 0.10.0` wheel 源码读出，回退路径的签名错处
  已按外部复核修正（bug 4），但整条导出路径仍需你在 PC 上首验。
```

原作者对另一个 App 的图像与性能观察，也不能证明该 App 可消费本包导出的模型。其 TFLOPS 与 INT8 TOPS 直接相除的利用率、用权重文件大小/时延代替 DRAM 流量的瓶颈判断口径不足，本报告不采用这些确定性性能结论。

### 7.3 下一阶段需要的四类决定性证据

1. **语义与数值：**同输入分别比较原 PyTorch→浮点 LiteRT、浮点→量化、同量化图 CPU→NPU。覆盖代表性 timestep，区分导出误差、量化误差和后端差异；完整采样再看图像质量。
2. **NPU 执行：**MT6897 分区报告、实际加载的 MediaTek backend、成功 Run 及数值输出；授权调试条件下用厂商 profiler/驱动任务补强。单个布尔值、文件扩展名或退出码均不够。
3. **完整端上图片：**固定模型/配置/prompt/seed，保存初始噪声、关键 latent、最终 PNG；确认端上 text/scheduler/VAE 都真正参与，图像符合 prompt 且与参考质量可比。
4. **资源与性能：**分开记录初始化、文本编码、每步主干、边界传输、scheduler、VAE、总时延和峰值内存；冷启动与模型复用后的运行分开。当前没有可填写的 MT6897 实测数字。

## 8. 面试问答：把边界讲清楚

**问：这是魔改 LiteRT 才支持 DiT 吗？**  
答：抽查的转换入口、桥接、模型分发和 FLUX2 导出文件与官方 v0.10.0 相同，官方本身已有特定 DiT 适配。本地主要封装转换/量化/AOT，并修补部分图表示。模型适配、编译器改动、设备后端支持应分别用源码和运行证据说明。

**问：官方既然有 FLUX，为什么这个脚本还不能直接跑？**  
答：脚本直接取 Diffusers transformer，再用自己的三输入模板调用通用 convert，没有走官方模型专属拆分与接口。随后还缺全输入运行器、调度器、文本编码和 VAE。依赖里存在功能，不等于主程序已经调用。

**问：RoPE 文件大小与原 App 一致，能证明接通吗？**  
答：只能证明字节数一致。还要证明表内容/排列符合该模型，图或 runner 实际读取，并验证输出。当前生成后只加入附件清单。

**问：固定 shape 会不会让 timestep 也固定？**  
答：不会。固定 `[1]` 这样的输入形状后，每步可以填不同数值。只有把时间值错误烘成常量才会破坏迭代合同。

**问：有真实 latent 就能做好 DiT 校准吗？**  
答：要把同一次前向的文本条件、时间步、位置/引导输入一起恢复。当前真实 latent 配随机文本、固定 500，破坏了输入之间的关系；还要覆盖整个噪声区间。

**问：NPU compute 成功为什么还可能出不来图？**  
答：compute 只表示某次网络执行成功。多路条件、预测参数化、scheduler 更新、CFG 和 VAE 缩放任何一处接错，完整输出都可能失败。最终验收是可重放的图片和质量对照。

**问：第一版你会怎么做？**  
答：固定一个官方已有 adapter 的模型，先拿到可信浮点参考与完整轨迹，再做目标兼容的量化和 MT6897 AOT。用正式 LiteRT runner 跑去噪主干，CPU 完成文本编码、scheduler 和 VAE。先让整张图在设备上正确生成，再优化全链时延和更多 NPU 覆盖。

## 9. 阅读入口与发布范围

- [主报告：转换链、MediaTek、Cache 与面试题](PC_CONVERT_MT6897技术报告.md)
- [Qwen2.5-0.5B：量化、prefill/decode、KV cache](Qwen2.5-0.5B_MT6897逐步转换与面试问答.md)
- [其他 LLM、ASR、TTS 扩展可行性](LLM_ASR_TTS扩展可行性分析.md)
- [四文件源码来源对照](evidence/LiteRT_Torch_0.10.0_四文件对照.md)
- [官方 DiT 模型适配、shape、量化与 host 源码索引](evidence/官方DiT适配源码索引.md)

本仓库发布报告、必要源码节选和比较记录；不发布原始离线工具包、MediaTek SDK 二进制、模型权重或凭据。官方源码节选来自 Google AI Edge 项目，相关文件采用 Apache-2.0，版权与许可归原项目；原包节选用于用户提供代码的分析。上述代码问题本轮没有实施修复，报告中的运行循环与交付物明确属于后续实现合同。
