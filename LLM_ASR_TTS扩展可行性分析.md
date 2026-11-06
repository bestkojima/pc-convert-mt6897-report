# pc_convert 能否扩展到其他 LLM、ASR 和 TTS？

**结论：底层工具链可以复用；当前 `convert_dit.py` 不能原封不动通吃这些模型。**

推荐保留 LiteRT Torch → Quantizer → MediaTek AOT → Android Runtime 这条公共链，为各类模型使用已有专项导出器或小型适配层。对 MT6897，优先让计算密集且形状可控的子图进入 NPU，文本/音频处理、状态更新和采样由合适的 CPU/runtime 组件负责。

本篇承接 [主报告](PC_CONVERT_MT6897技术报告.md)，分析日期 **2026-11-06**。版本依据仍为本地 LiteRT Torch 0.10.0、LiteRT 2.3.0、Quantizer 0.10.1。这里只做源码与一手资料分析，未下载模型、执行转换或进行设备推理。

下文 `WHEEL:` 指本地 `D:\misc\pc_convert\pc_convert\offline\pip_pkgs\litert_torch-0.10.0-py3-none-any.whl` 内的源文件；行号以归档内原文件为准。该包相关源码具有 Apache-2.0 声明，节选用于解释实现。在线资料用于补充模型结构和官方部署路线，不能覆盖本地版本的实际行为。

## 1. “能转换”要拆成四个问题

| 关口 | 要成立的事实 | 哪种证据不够用 |
|---|---|---|
| 模型能导出 | 网络结构、权重、算子和输入合同能表示为 LiteRT 图 | 只看到 `.safetensors`、ONNX 文件或 HF model ID |
| 量化可接受 | 真实数据校准后，数值与任务质量可用 | 文件变小、有 INT8 张量 |
| MT6897 可编译 | 目标插件接受该算子/shape/dtype 组合，关键子图有实际卸载 | 通用 CPU LiteRT 能跑、target 枚举包含 MT6897 |
| 设备业务可用 | 匹配 ROM/runtime 的真实 Run、完整状态与前后处理、速度和质量达标 | AOT 生成成功、CompiledModel 创建成功 |

LLM、ASR、TTS 都包含大量矩阵乘、卷积、归一化和 attention，底层编译器有复用价值；但它们的**输入语义、时间轴、状态与输出长度**不同。模型格式相同不能抹掉这些差异。

LiteRT Torch 的通用转换要求模型符合 `torch.export`，接口应能表示为张量输入/输出。遇到框架对象、动态 Python 控制流或定制 CUDA 算子，需要先做适配。官方也把“PyTorch 图捕获失败”和“图 lowering 失败”作为两类问题区分。[官方 PyTorch 转换说明](https://github.com/google-ai-edge/litert-torch/blob/main/docs/pytorch_converter/README.md)

## 2. 当前脚本有哪些硬限制

源码位置：`D:\misc\pc_convert\pc_convert\convert_dit.py`。

### 2.1 `--from-hub` 仅加载 Diffusers 的 transformer

`convert_dit.py:239–245`：

```python
if args.from_hub:
    try:
        from diffusers import DiffusionPipeline
    except ImportError:
        raise SystemExit("--from-hub 需要 diffusers：pip install diffusers")
    log(f"  从 HF 取 {args.from_hub} 的 transformer")
    model = DiffusionPipeline.from_pretrained(args.from_hub).transformer
```

所以把 Qwen、Whisper 或 Piper 的仓库名直接填进去，不能获得正确的对应模型。它没有统一调用各类 `AutoModel`，也没有语音 pipeline 的装配逻辑。

### 2.2 样例输入固定为 DiT 的三个张量

`convert_dit.py:320–324`：

```python
hs = torch.randn(1, L, args.hidden_dim, dtype=torch.float32)
tb = torch.randn(1, args.text_len, args.text_dim, dtype=torch.float32)
t = torch.full((1,), 500.0, dtype=torch.float32)
return ({"hidden_states": hs, "encoder_hidden_states": tb, "timestep": t},
        (args.latent_ch, lh, lw, L))
```

三类模型需要的典型接口则是：

- **文本 LLM**：token IDs/embedding、position、mask、各层 KV cache → logits、新 K/V。
- **ASR**：PCM 或声学特征、有效帧长、mask，流式模型还有状态 → logits/文本 token/新状态。
- **TTS**：文本/音素 IDs、speaker/style、duration 或噪声/时间步 → mel、codec token 或波形。

把 `hidden_dim` 调到某个看起来相近的值，不能使 DiT 的接口自动变成 ASR 或 TTS。

### 2.3 真实校准入口也绑定了 DiT

`convert_dit.py:805–808`：

```python
samples.append({
    "hidden_states": hs,
    "encoder_hidden_states": rng.standard_normal((1, T, TD)).astype(np.float32),
    "timestep": np.array([500.0], dtype=np.float32)})
```

语音模型不能用这套随机文本条件和固定时间步校准。ASR 要真实音频/声学特征，TTS 声码器要真实 mel/隐变量，LLM 要实际 token、位置与多步 cache 状态。

### 2.4 现成 `.tflite` 入口可复用一部分，但仍有边界

输入 `.tflite` 会跳过模型加载/DiT 样例导出，后续可以尝试规整、量化和 AOT；这给其他任务留下了入口。但当前脚本的量化、清单、单子图检查及设备探针仍带 DiT/小 MLP 假设。

**已有正确量化的 TFLite，直接调用 AOT API 更干净。** 无需强行再次经过 `convert_dit.py` 的量化。公共编译 API 见主报告第 6 章。

当前脚本没有 ONNX 或 GGUF 导入器。FunASR/Piper 能导出 ONNX，证明的是那条工具链有入口；它不会自动成为本项目的 MT6897 NPU 模型。优先从可信的原始 PyTorch 模型直接导出，避免无必要的 ONNX → TensorFlow → TFLite 多次转换。

## 3. 应该怎样组织扩展，避免越改越乱

```text
文本 LLM ── HF / Generative 专项导出 ── prefill、decode、aux ──┐
ASR      ── 语音专项导出 / 模型 wrapper ── encoder、decoder ─┤
TTS      ── 专项导出 / 声学与声码器拆分 ── 可控计算子图 ─────┤
                                                         ▼
                                      模型专属校准与量化配方
                                                         ▼
                                      MediaTek MT6897 AOT
                                                         ▼
                                      完整 Runtime + 业务状态机
```

我建议复用库 API，保留少量分任务导出入口；先别把所有任务塞进一个上百个参数的“万能脚本”。公共部分应保持统一：目标选择、产物状态、版本信息、I/O 元数据、错误处理、运行后端记录。任务专属部分保持独立：预处理、样本构造、状态更新、解码与质量指标。

本地依赖里已经发现 Whisper、Qwen3-ASR、Qwen3-TTS 的专项源码。它们能减少从零适配的工作，但每个分支还需检查自己是否接入 NPU 编译、是否完整打包、是否有对应运行器；“仓库里有一个类”只能回答第一层问题。

## 4. 其他文本 LLM：可以扩展，小型 dense 模型优先

### 4.1 已有适配覆盖到哪里

| 家族 | 随包可见的导出入口 | MT6897 的判断 |
|---|---|---|
| Qwen2 / Qwen2.5 dense | HF 通用入口；重写配置含 0.5B、1.5B、3B | 最适合先做完整链路候选 |
| Qwen3 dense | 专项 patch；重写配置含 0.6B、1.7B、4B | 可以继续研究，需处理 Q/K norm、composite 与目标算子 |
| Llama | HF 入口；Llama3.2 1B/3B 重写配置 | 可扩展，按确切 config/许可/模板选择 |
| Gemma | 不同代有专项导出或重写模型 | 各代差异大，不能把“Gemma 支持”视为统一合同 |
| Phi-2 / Phi-3.5 / Phi-4 | 老式 Generative 重写模型 | 有起点，新 HF/NPU 路线需另核对 |
| SmolLM / SmolLM2 | 重写配置含 135M | 适合极小模型研究，中文任务能力另选 |
| Mistral / SmolLM3 | 官方 HF 导出文档列出，使用通用导出路径 | 滑窗、mask、具体结构与后端继续核对 |
| DeepSeek-R1-Distill-Qwen | 本地示例走 Qwen 蒸馏结构 | 可以沿 Qwen 路线研究；完整 DeepSeek MoE/MLA 是另一件事 |

这里列的是**前端适配证据**，没有声称这些模型已在本任务的 MT6897 上执行成功。官方文档同样把普通 HF 导出与 NPU AOT 分为不同步骤。[官方 GenAI 导出说明](https://developers.google.com/edge/litert/conversion/pytorch/genai)

HF 入口通过任务选择模型类，`WHEEL:litert_torch/generative/export_hf/core/export_lib.py:383–390`：

```python
if task == ExportTask.TEXT_GENERATION:
  auto_model_cls = transformers.AutoModelForCausalLM
elif task == ExportTask.IMAGE_TEXT_TO_TEXT:
  auto_model_cls = transformers.AutoModelForImageTextToText
elif task == ExportTask.MULTIMODAL_LM:
  auto_model_cls = transformers.AutoModelForMultimodalLM
else:
  raise ValueError(f'Unsupported task: {task}')
```

实际有无专用 patch、是否需要特殊 cache/embedding 分支，还要跟进该架构。代码里的 `_CAUSAL_TEXT_MODEL_KEYS` 只是因果 mask 优化的名单，不能拿它充当“MT6897 支持模型列表”。

### 4.2 同家族的微调模型能不能复用

Base、Instruct、Coder、Math、蒸馏或全参微调模型，只要结构保持一致，通常可以复用结构前端；仍须检查层数、hidden/head/KV 维度、词表、RoPE、tokenizer 与 chat template。

首轮使用原始浮点 checkpoint。当前 HF 入口明确拒绝带量化配置的 checkpoint，`WHEEL:litert_torch/generative/export_hf/core/export_lib.py:130–135`：

```python
if (
    hasattr(model_config, 'quantization_config')
    and model_config.quantization_config
):
  print(utils.ERROR_MESSAGE)
  raise NotImplementedError('Quantized checkpoint is not supported yet.')
```

因此别先拿 GPTQ/AWQ/bitsandbytes 权重来套此入口。LoRA adapter 通常只有增量，需与正确 base 共同加载/合并，不能当完整权重文件。

### 4.3 有专项导出也可能撞到明确限制

Gemma3n 是一个直接反例，`WHEEL:litert_torch/generative/export_hf/model_ext/exportables.py:45–51`：

```python
if model_config.model_type == 'gemma3n':
  assert (
      not export_config.split_cache
  ), 'Split cache is not supported for Gemma3N.'
  assert (
      export_config.externalize_embedder
  ), 'External embedder is required for Gemma3N.'
```

主报告中的实验 NPU 四阶段默认强制 `split_cache=True`。两条入口直接拼在一起就会冲突，必须用匹配的专属路径或完成适配。

其他要单独处理的结构：

- **MoE**：expert 路由、gather/scatter、动态选择与权重驻留；本包虽有部分 MoE 实现，某些 V2 + expert 导出组合仍显式拒绝。
- **Mamba/SSM 或混合状态模型**：递归/卷积状态与 scan，无法只套普通 attention KV。
- **MLA**：压缩状态、解压投影及位置分量改变缓存合同。
- **长上下文 RoPE 变体**：本包对部分 dynamic/longrope 配置有显式限制。
- **多模态 LLM**：encoder/projector、特殊 token、额外 mask 与采样器都要完整接入。

### 4.4 我选的第一个模型

**Qwen2.5-0.5B-Instruct，原始浮点权重，batch 1、静态 cache 2048、prefill 128。**

选择理由：dense `qwen2` 架构，包里有确切 0.5B 配置；24 层、2 个 KV 头，结构与状态规模较小，支持中文指令场景，适合把导出、模板、prefill/decode、量化和设备执行逐层讲明白。[官方 checkpoint 配置](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json)

这一步追求的是把链路做扎实。中文对话、数学或教学质量需要后续换更合适的模型并比较，0.5B 不能因工程容易就被默认成最终产品模型。下一候选可选 Qwen3-0.6B 或已有明确需求的 Qwen2.5-1.5B。

执行路线沿用主报告第 9 章：原始 checkpoint → LLM 导出 → 真实文本/状态校准 → 静态量化与跨图 KV 尺度对齐 → MT6897 AOT → LiteRT-LM/对应生成 runtime。

## 5. TTS：可以做，但应优先拆声码器和生成状态机

### 5.1 本包有 Qwen3-TTS 导出器，适配范围很具体

`WHEEL:litert_torch/generative/export_hf/model_ext/exportables.py:209–214`：

```python
def get_tts_model_cls(model_type: str):
  """Gets TTS model class by model type."""
  if model_type in ('qwen3_tts', 'qwen3_tts_talker', 'qwen3'):
    return qwen3_tts_lib.Qwen3Tts
  else:
    raise ValueError(f'Unsupported TTS model type: {model_type}')
```

所以已有 `task="text_to_speech"` 任务入口，能复用 Qwen3-TTS 专用代码；Piper、Kokoro、F5-TTS 不会因为选择同一 task 就自动获得适配。注册表里的 `qwen3` 也不把普通文本 Qwen3 权重变成语音模型。

Qwen3-TTS 导出拆成以下子图：

```text
文本/参考音频的 CPU 前处理
    ├─ speaker_encoder：说话人特征
    ├─ embedding / projection：条件与 token 表示
    └─ talker：沿时间轴产生 codec token
            └─ MTP/code predictor：同一时刻其余码本预测
                    └─ codec_decoder：codec token → 波形
                            └─ 分块拼接与播放
```

这意味着 CPU/runtime 仍需负责多图调度、采样、停止条件、K/V 与中间 buffer。NPU 编译其中一张图，不能自动获得整个语音引擎。

### 5.2 当前 TTS 的“一键 AOT”有明确断口

这三段代码连起来很有解释力。

`WHEEL:litert_torch/generative/export_hf/export.py:328–332`：

```python
elif task == ExportTask.TEXT_TO_SPEECH:
  export_tasks.append(export_lib.export_tts_models)
  if export_config.aot_backend is not None:
    export_tasks.append(export_lib.aot_compile_model)
    legacy_compile_triggered = True
```

`WHEEL:litert_torch/generative/export_hf/core/export_lib.py:1209–1214`：

```python
tts_model = source_model_artifacts.model
artifacts = tts_model.export(export_config)
return dataclasses.replace(
    exported_model_artifacts,
    additional_model_paths=artifacts,
)
```

同文件 `1496–1498`：

```python
target = backend.target
source_model = exported_model_artifacts.prefill_decode_model_path
assert source_model is not None, 'Prefill-decode model is not found.'
```

**导出写 `additional_model_paths`，旧 AOT 却读 `prefill_decode_model_path`。** 默认接线没有交接上。TTS 顶层分支也没安排正常的 `.litertlm` 打包动作，强启新的 NPU pipeline 仍会因缺少实际包路径失败；单设 `bundle_litert_lm=True` 无法补齐业务逻辑。

最短可行改造方向是：复用独立 TFLite 子图导出，逐张用公共 MT6897 AOT 编译，再以明确的 TTS runtime 组合。完整支持 `.litertlm` TTS 还需要包内模型 section、编译策略和消费端同时接通。

### 5.3 先锁定 0.6B Base，不能随便换 1.7B

本地导出器把 talker 的 hidden size、层数、KV 头数等写成固定配置，主要尺寸对应 **Qwen3-TTS-12Hz-0.6B-Base**。官方 1.7B 的 hidden/intermediate 和 speaker embedding 维度不同，必须改结构和权重映射；尺寸匹配后还要对齐位置编码与运行语义。[官方 0.6B 配置](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/blob/main/config.json)、[官方 1.7B 配置](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/blob/main/config.json)

权重加载也有风险，`WHEEL:litert_torch/generative/export_hf/model_ext/qwen3_tts/qwen3_tts.py:109–113`：

```python
safetensors_path = os.path.join(self.model_dir, "model.safetensors")
if os.path.exists(safetensors_path):
  self.reader = safetensors.safe_open(safetensors_path, framework="pt")
else:
  self.reader = None
```

缺少该单文件时，MTP 分支会构造随机权重，talker 则不导出。分片 checkpoint 也不能仅靠一个目录名被这里自动识别。正式转换应对必需权重缺失立即失败，不能验收一堆“成功生成”的随机模型文件。

### 5.4 原模型支持流式，不代表导出接口已经保留流式能力

`WHEEL:litert_torch/generative/export_hf/model_ext/qwen3_tts/qwen3_tts.py:340–354` 节选：

```python
class CodecDecode(torch.nn.Module):
  def __init__(self, dec):
    super().__init__()
    self.dec = dec
  def forward(self, codes):
    return self.dec(codes)
out_path = os.path.join(output_dir, "codec_decoder_fp32.tflite")
sample_input = (torch.zeros(1, 16, 64, dtype=torch.int32),)
litert_torch.convert(CodecDecode(decoder).eval(), sample_input).export(out_path)
```

该样例固定为 16 个码本、64 个位置，没有在此接口暴露任意流式状态。原模型的低延迟流式机制，还需要导出对应状态、确定 chunk 与边界处理，不能照抄官方原模型延迟作为这个转换产物的成绩。[Qwen3-TTS 官方项目](https://github.com/QwenLM/Qwen3-TTS)

### 5.5 TTS 候选怎么选

| 路线 | 对本项目的价值 | 主要工作/风险 | 选择意见 |
|---|---|---|---|
| 小型 HiFi-GAN 声码器 | mel → 波形，单次前向，无自回归/ODE 外循环 | ConvTranspose1d、扩张卷积、量化音质与分块边界 | **首个 MT6897 TTS 子图候选** |
| FastSpeech2 + HiFi-GAN | 声学与声码器分工明确 | duration 产生动态长度；G2P、展开和 bucket 管理 | 声码器通过后再迁移声学部分 |
| Qwen3-TTS 0.6B Base | 随包已有多子图导出，支持中文 | 权重与几何检查、AOT/打包断口、采样与流式 runtime | 复用已有导出器的完整 TTS 研究路线，工程量较大 |
| Piper / VITS | 已有端侧 CPU/ONNX 路线及中文 voice | duration/flow、动态长度、算子 lowering；需额外 LiteRT 前端 | 适合作产品与音质对照，整图 NPU 不作为第一步 |
| Kokoro-82M-v1.1-zh | 作者提供轻量中文版本 | LSTM、动态展开、iSTFT/相关解码路径 | 值得研究，但本包没有专项 adapter |
| F5-TTS | 使用 DiT/flow matching，有可拆神经子图 | 声学输入、ODE 循环、动态长度、vocoder | 图像 DiT 的输入模板不能直接复用 |
| CosyVoice | 明确拆 LLM、flow、HiFT | AR + flow + vocoder，多级缓存与流式调度 | 完整能力丰富，首轮移植复杂度高 |

矩阵依据来自各项目维护者的一手资料：[HiFi-GAN 作者源码](https://github.com/jik876/hifi-gan/blob/master/models.py)、[FastSpeech2 论文](https://arxiv.org/abs/2006.04558)、[Piper voice 文档](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md)、[Kokoro 中文 model card](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh)、[F5-TTS 作者项目](https://github.com/SWivid/F5-TTS)、[CosyVoice runtime](https://github.com/QwenAudio/CosyVoice/blob/main/cosyvoice/cli/model.py)。表中的移植顺序是依据结构复杂度给出的工程判断。

### 5.6 声码器为什么值得先做

推荐先取真实声学模型产生的 mel，把 HiFi-GAN 固定帧长子图导出、编译、执行。这样把文本归一化、多音字/G2P、duration 展开、自回归采样等独立问题留在外面，能清楚回答“NPU 加速了哪部分”。

必须匹配原声学模型的采样率、FFT/hop、mel bins、频率范围、log/归一化和 padding。英文 LJSpeech checkpoint 只能代表对应数据路线；中文产品选配套中文声学模型与声码器。可参考维护者发布的 [FastSpeech2 Mandarin AISHELL-3 路线](https://github.com/ming024/FastSpeech2)，这是独立实现的一手资料，需沿其配套配置使用。

普通非因果 HiFi-GAN 切块需要左右上下文与裁切，相邻波形直接硬拼可能产生咔哒声。量化要覆盖真实 mel，听可懂度、韵律、齿音、爆音和块边界；张量余弦接近不能替代听感。

精度上先保留浮点参考，再研究真实校准的 W8A16；不要直接把 DiT 的全局 W4A16 配方套到整条语音链。最终是否下放、哪些层保留精度，以该目标编译器接受的组合和音质结果为准。

## 6. ASR：现成入口比想象中多，先把 encoder 跑通

### 6.1 包内明确注册了五类 ASR

`WHEEL:litert_torch/generative/export_hf/model_ext/exportables.py:193–206`：

```python
def get_speech_model_cls(model_type: str):
  """Gets ASR model class by model type."""
  if model_type == 'parakeet_ctc':
    return parakeet_ctc_lib.ParakeetCTC
  elif model_type == 'parakeet_tdt':
    return parakeet_tdt_lib.ParakeetTDT
  elif model_type == 'whisper':
    return whisper_lib.Whisper
  elif model_type == 'moonshine':
    return moonshine_lib.Moonshine
  elif model_type == 'qwen3_asr':
    return qwen3_asr_lib.Qwen3Asr
  else:
    raise ValueError(f'Unsupported speech model type: {model_type}')
```

ASR 顶层按 `bundle_litert_lm` 分成打包路径与 standalone 路径。首轮推荐 standalone，容易看清声学图和解码图。下面使用本包现有 API，**仅为待执行的浮点导出示例**：

```python
from litert_torch.generative.export_hf.export import export

export(
    model="/models/whisper-tiny",   # 预先准备的多语言 checkpoint 目录
    output_dir="./whisper_tiny_export",
    task="automatic_speech_recognition",
    bundle_litert_lm=False,
    quantization_recipe="none",    # 先建立浮点参考
    input_sec=30.0,
    keep_temporary_files=True,
)
```

`export_asr_models` 会把 `encode` 与 decoder signatures 放入同一个 `asr_model.tflite`。**这个调用没有自动做“只把 encoder 编译到 NPU”。** encoder-only 方案需要进一步独立导出该模块，或在后端支持的分区/签名机制下明确选择，并配好对应运行器。公共 MT6897 AOT 接法沿用主报告第 6 章。

### 6.2 默认导出与 MT6897 的目标配方还有距离

ASR 装载时启用 Transformers override，替换 attention 等实现。`WHEEL:litert_torch/generative/export_hf/core/speech/asr_model.py:194–199`：

```python
def __init__(self, override_transformers: bool = False):
  if not override_transformers:
    return
  # Replace sdpa for GPU inference.
  attn_funs = transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS
  attn_funs["sdpa"] = _sdpa
```

它使用面向 GPU 的 `odml.scaled_dot_product_attention` composite，并有 norm 包装。最终需要查看 lowering 后的图，确认 MediaTek 后端识别或分解了哪些算子。看到 GPU 注释不能直接断言 NPU 必然失败，但也不能跳过这一关。

默认量化配方是 `dynamic_wi8_afp32`。standalone helper 直接调用量化，`WHEEL:litert_torch/generative/export_hf/core/export_lib.py:782–783`：

```python
qt.quantize().export_model(quantized_model_path, overwrite=True)
return quantized_model_path
```

这里没有接入真实音频校准数据。静态 W8A16/W8A8 路线要补音频、encoder 输出和 decoder 前缀样本，单换 recipe 字符串不够。主报告的量化过程可复用，校准数据生成器必须改成语音语义。

### 6.3 Whisper 的现有 decoder 没有显式 KV 接口

`WHEEL:litert_torch/generative/export_hf/model_ext/whisper/whisper.py:31–44`：

```python
def forward(
    self,
    encoder_hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
  """Returns the decoder's logits for each token including the next token."""
  decoder_outputs = self._model.get_decoder()(
      encoder_hidden_states=encoder_hidden_states,
      input_ids=input_ids,
      attention_mask=attention_mask,
  )
  logits = self._model.proj_out(decoder_outputs.last_hidden_state)
  return (logits,)
```

输入只有音频 encoder 特征、文本 token 与 mask，输出只有 logits；没有跨调用 `past_key_values` 输入/输出。应按全前缀解码接口理解它，不能承诺高效逐 token KV 复用。若继续优化 decoder，需要补明确的 self-attention/cross-attention 状态合同，或使用已经具备该合同的另一套导出/runtime。

Whisper 完整转写还包含 log-Mel 前端、语言与任务 token、停止条件、时间戳和长音频窗口处理。首轮选**多语言 `openai/whisper-tiny`**，固定中文任务前缀、固定窗口、greedy decode；中文不能误拿 `.en` 版本。先将声学 encoder 作为 NPU 目标，CPU/已有可用后端维持 decoder 和完整识别。[Whisper 官方项目](https://github.com/openai/whisper)

### 6.4 中文模型怎么选

| 模型 | 本包现状 | 关键难点 | 建议 |
|---|---|---|---|
| Whisper-tiny 多语言 | 已有专项导出 | decoder 无显式 KV；音频窗口与生成控制 | **打通工具链第一站** |
| Whisper-base/small | 同一类可接对应 checkpoint | 计算/内存增加，仍需完成同一状态接口 | tiny 路线成立后扩展 |
| Qwen3-ASR-0.6B-hf | 已有专项导出 | 音频 prompt、重排、位置与 decoder 状态 | 已有 adapter 中的中文后续候选 |
| SenseVoiceSmall | 未在此 ASR registry 注册 | 新增 LiteRT adapter、保留音频前端/长度/任务提示 | **中文短句产品方向优先评估** |
| Paraformer | 无专项注册 | CIF/predictor 产生数据相关 token 长度 | 可做，安排在 SenseVoice 后 |
| Streaming Zipformer/Conformer | 无覆盖各变体的通用接入 | chunk、left context、卷积/attention 状态；transducer 还有 joiner | 连续低延迟需求明确后选具体 checkpoint |

本包 Qwen3-ASR 类对应的精确模型为 **`Qwen/Qwen3-ASR-0.6B-hf`**，使用 Transformers 原生类。standalone decoder 同样只输出 logits，不能假设它已经继承普通 LLM 的完整增量 cache 方案。其他格式的 Qwen3-ASR checkpoint 需单独核对。[官方 HF 模型卡](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf)

**SenseVoiceSmall 的吸引力在结构：encoder + CTC，省掉文本逐 token 的自回归调度。** 官方支持普通话、粤语、英语、日语、韩语，并提供其他格式导出；本项目需补 LiteRT 接口。[官方项目](https://github.com/QwenAudio/SenseVoice)、[官方模型卡](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)

FunASR 的 SenseVoice 导出接口是 `speech,speech_lengths,language,textnorm`，样例 speech 特征维度为 560，还会添加四个任务提示位置。它接收处理后的声学特征，不能把 PCM 直接塞给这个张量。[维护者导出源码](https://raw.githubusercontent.com/modelscope/FunASR/main/funasr/models/sense_voice/export_meta.py)

推荐分工：

```text
CPU：PCM → 重采样 → fbank/LFR/CMVN → padding + 有效帧长
NPU：支持的 prompt/encoder/CTC 计算子图
CPU：CTC blank/repeat 处理 → token/标签映射 → 文本
```

以 VAD 切句、逐句识别，称为“分段识别”。真正的流式 encoder 需要跨 chunk 保留状态。Paraformer 的 predictor/CIF 还会决定数据相关的输出长度，静态部署比固定 encoder+CTC 多一道形状管理；可借鉴官方 ONNX 导出接口，但不能把它直接算作 LiteRT 后端支持。[Paraformer 导出源码](https://raw.githubusercontent.com/modelscope/FunASR/main/funasr/models/paraformer/export_meta.py)

### 6.5 音频前端和长度合同必须保留

第一版把 PCM 解码、重采样、分帧、STFT、mel/fbank、log、CMVN 与 VAD 留在 CPU，NPU 承接声学主干。逐项记录采样率、声道、幅度、窗长/hop、FFT、mel 滤组、center/padding、归一化与 LFR；Whisper 的 log-Mel 和 FunASR 的 fbank/LFR 不能互换。

`input_sec` 只控制生成导出样例音频的长度；processor 仍可能 pad/truncate，最终 graph shape 要看实际张量。`WHEEL:litert_torch/generative/export_hf/core/speech/exportables.py:43–53`：

```python
processor = self.asr_model.get_processor()
sr = processor.get_sampling_rate()
input_sec = getattr(self.export_config, "input_sec", 1.0)
if model_config.model_type.startswith("parakeet"):
  duration = input_sec - 0.01  # Parakeet processor adds extra 10ms.
else:
  duration = input_sec
dummy_audio = np.zeros(int(duration * sr), dtype=np.float32)
processed = processor.process(dummy_audio)
encoder_inputs = self.asr_model.get_encoder_sample_input(processed)
return {"encode": (encoder_inputs, {})}
```

这里的零音频服务于追踪图形状，真实量化校准仍要实际语音。固定 bucket 可以减少动态形状麻烦，但必须保留有效长度/mask，且长度要随 subsampling、prompt 前缀同步更新。否则模型可能把 padding 当语音，“编译成功，识别胡说”就来了。

## 7. 语音模型的 cache size：先说清缓存的对象和单位

### 7.1 不同任务的缓存分开记账

| 对象 | 单位/内容 | 如何设置 |
|---|---|---|
| 文本 LLM KV | token 位置 × 层 × KV 头 | 按最大 prompt + 生成预算及模型上下文上限选，静态图需同步导出 |
| ASR 音频缓冲 | PCM samples、特征帧 | 由采样率、hop、窗口与端点检测决定 |
| 流式 ASR encoder state | attention 历史、卷积历史、subsampling padding、帧计数 | 按具体 checkpoint 的训练/导出 chunk 与 left-context 合同 |
| ASR decoder KV | 已生成文本 self-KV、音频 cross-KV | 仅在导出/runtime 明确携带这些状态时成立 |
| Qwen3-TTS talker KV | 跨时间步的生成状态 | 按文本/参考条件和语音生成长度预算；不能直接当秒数 |
| Qwen3-TTS MTP KV | 同一时刻码本预测的局部序列 | 本包固定 32，与 talker 长度独立 |
| vocoder/flow/播放缓存 | mel/latent overlap、卷积状态、PCM 队列 | 由块大小、感受野、连续性和首包延迟决定 |
| 编译产物缓存 | 编译结果文件或 runtime artifact | 减少准备/冷启动，不增加语义上下文长度 |

SenseVoice 的离线 CTC 解码没有逐 token 文本 KV，但声学激活仍占内存。增大某个 LLM `cache_length`，不会让语音 encoder 自动接收更长音频。

### 7.2 ASR 会重写 split-cache 配置

`WHEEL:litert_torch/generative/export_hf/core/exportable_module_config.py:371–377`：

```python
case ExportTask.AUTOMATIC_SPEECH_RECOGNITION:
  self.export_vision_encoder = False
  self.split_cache = False
  self.externalize_rope = False
  self.externalize_embedder = self.bundle_litert_lm
  self.export_audio_encoder = self.bundle_litert_lm
  self.single_token_embedder = self.bundle_litert_lm
```

所以 LLM 实验 NPU 管线的 split-cache 设置不能原样照搬。`stateful_after` 也要沿代码追：当前 speech exportable 用它截短样例 token 长度、增设 `decode_1` 签名，并没有自动给前述 Whisper wrapper 添加 KV I/O。

### 7.3 Qwen3-TTS 有一个容易误读的默认值改写

`WHEEL:litert_torch/generative/export_hf/model_ext/qwen3_tts/qwen3_tts.py:526–531`：

```python
cache_length=export_config.cache_length
if export_config.cache_length != 4096
else 1024,
prefill_lengths=export_config.prefill_lengths
if export_config.prefill_lengths != [128]
else [32, 128],
```

**顶层填 4096，内层实际变成 1024；显式填 4096 也一样。** 这段代码用“等于默认数值”判断是否保留用户设置，分不清用户明确选择与默认值。真需要 4096，应修正传参机制后重新导出；仅改命令参数得不到预期结果。

MTP 常量在 `WHEEL:litert_torch/generative/export_hf/model_ext/qwen3_tts/exportable_module.py:24–29`：

```python
VOCAB = 2048
CACHE = 32
LAYERS = 5
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
```

这个 cache 处理局部码本预测序列，不能拿 32 估计整句话能说多久。Talker、MTP、codec decoder 三者的长度轴要分开理解。

### 7.4 缓存内存可以算，但不能把估算当峰值 RSS

密集、未压缩且每层保存 K/V 时：

```text
KV_bytes = 2 × batch × layers × positions × KV_heads × head_dim × element_bytes
MiB = bytes / 1024²
```

以 batch=1 计算：

| 配置 | FP32 KV | FP16 KV |
|---|---:|---:|
| Qwen2.5-0.5B：24 层、2048 位置、2 KV 头、head_dim 64 | 48 MiB | 24 MiB |
| Qwen3-TTS talker：28 层、1024 位置、8 KV 头、head_dim 128 | 224 MiB | 112 MiB |
| 同一 talker，4096 位置 | 896 MiB | 448 MiB |
| MTP：5 层、32 位置、8 KV 头、head_dim 128 | 1.25 MiB | 0.625 MiB |

这些是按结构算出的逻辑容量，不含权重、激活、工作区、对齐、双缓冲与数据复制。本包 MTP 的 K/V 样例为 FP32；FP16 列是更换状态 dtype 的容量比较，FP16 权重配方不能证明缓存也已成为 FP16。

ASR 若明确导出 self/cross KV，分别以文本位置数和音频 encoder 位置数代入；共享或重算的实现按实际张量计。流式 encoder 的状态更适合直接求和：

```text
state_bytes = Σ[product(state_i.shape) × state_i.element_bytes]
```

chunk 变大可减少调用次数，也增加等待首块音频的积累时间；left context 变大增加历史状态和计算。先遵守具体模型支持的档位，再在首包延迟、质量与开销之间选择。参考 [icefall 流式 Zipformer 导出](https://raw.githubusercontent.com/k2-fsa/icefall/master/egs/librispeech/ASR/zipformer/export-onnx-streaming.py)；[Sherpa-onnx 部署](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html) 可作另一条 CPU 基线，不能替代本项目的 NPU 证据。

## 8. 我建议的落地顺序与面试答法

### 8.1 三条任务，各挑一个最小完整目标

1. **LLM：Qwen2.5-0.5B-Instruct。** 原始浮点 checkpoint，batch 1、cache 2048、prefill 128；完成模板、量化、prefill/decode 状态和真实 MT6897 生成。
2. **ASR：Whisper-tiny encoder。** 现有 adapter 起步，CPU 保留音频前端与可用 decoder。之后用 SenseVoiceSmall 评估中文短句产品价值。
3. **TTS：小型 HiFi-GAN 声码器。** 真实 mel、固定 chunk、匹配声学模型；先得到可听且连续的 NPU 波形输出，再接声学部分或研究 Qwen3-TTS 多图状态机。

上述顺序是后续实施建议；本次交付是分析报告。若后续启动实施，每条路线先留一个完整正确的 CPU/浮点参考，再比较量化/NPU，避免把模型本身错误和转换错误搅在一起。

### 8.2 评价收益，要看完整任务

| 任务 | 正确性/质量 | 性能与资源 |
|---|---|---|
| LLM | logits/生成对齐、任务准确率、模板与停止行为 | TTFT、decode token/s、端到端延迟、峰值内存 |
| ASR | 中文 CER、英文 WER、数字/专名、静音/噪声 | RTF、首个稳定结果/最终结果延迟、内存与持续运行表现 |
| TTS | 听感、可懂度、韵律、说话人一致性、块接缝；ASR-CER 仅辅助 | 首包延迟、RTF、播放 underrun、内存与热降频 |

ASR 的 RTF 为处理耗时/输入音频时长，TTS 的 RTF 为合成耗时/生成音频时长。RTF 小于 1 表示平均处理速度快于实时，首包仍可能很慢。前后处理、CPU/NPU copy、同步和采样都计入端到端数据。

还要记录确切模型 revision、代码/权重许可、processor/tokenizer 配置、SDK/ROM/runtime 版本、静态 shape、量化配方、签名与状态 dtype。尤其中文 voice、参考音频和说话人克隆，要有相应使用授权。部署兼容性与许可分别判断。

### 8.3 面试时，用这段回答主问题

> 能扩展。公共能力是 PyTorch 导出、量化和 MediaTek AOT；当前 convert_dit.py 的模型加载、输入和校准绑定 DiT，要替换任务前端。包内已有多种 LLM，以及 Whisper、Qwen3-ASR、Qwen3-TTS 的专项适配。我先用小型 Qwen 打通文本生成，ASR 先把 Whisper encoder 放上 NPU，TTS 先放 HiFi-GAN 声码器。这样既保留完整可用业务，又能定位各个计算子图的收益。导出成功、目标编译成功、真机执行成功、业务质量达标，是四份不同的证据。

追问可以这样接：

- **为什么不能直接换权重？** 模型类、张量接口、状态和前后处理都可能不同；相同 `.safetensors` 只表示相同存储容器。
- **已有 ONNX，是否就能上 MT6897？** 可以借鉴接口和基线，但当前链没有 ONNX importer，仍需 LiteRT 前端与目标算子适配。
- **为什么 ASR 优先 encoder？** 计算集中，形状较易控制；decoder 的逐 token 调度、KV 和跨设备往返另行处理。
- **为什么 TTS 优先声码器？** mel→waveform 的合同简单，便于隔离卷积 lowering、量化失真和切块接缝。
- **cache 调大能解决长语音吗？** 先分音频输入长度、流式 encoder 状态、文本/talker KV 和波形缓存；每个对象都有自己的容量和形状合同。
- **发现了哪些真实代码问题？** Qwen3-TTS 导出产物字段与旧 AOT 输入字段没接上；4096 被改写成 1024；现有 Whisper standalone decoder 未暴露跨调用 KV。这些都比笼统说“框架支持语音”更影响落地。

**最后的判断：有值得复用的基础，而且语音前端已经走了一段；下一步应做分任务、分子图的适配。把当前 DiT 脚本包装成“任意模型一键 NPU 转换器”，会把真正的工程缺口藏起来。**
