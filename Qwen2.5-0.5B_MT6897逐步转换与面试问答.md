# Qwen2.5-0.5B-Instruct → MT6897 NPU：逐步转换、量化与 KV Cache 面试专题

**先记住主线：这版采用 Google LiteRT Torch 的模型导出与 MLIR 编译技术，配合 AI Edge Quantizer 做训练后量化，再用 MediaTek NeuroPilot 编译器为 MT6897 生成目标产物，最后由 LiteRT-LM 在 Android 上管理生成过程并调用 NPU。**

本专题固定模型为 **Qwen/Qwen2.5-0.5B-Instruct**，参数采用 **batch 1、cache 2048、prefill 128、校准最多 decode 32 步**。部署示例另设最多生成 128 token。四个数字各管一件事，后文逐一拆开。

分析日期：2026-11-06。依据用户提供的离线包源码与官方模型配置；本次没有执行模型转换或 MT6897 真机推理。流程中的代码用于说明确切接口与候选部署方案，设备端仍需配套 runtime、ROM 和实际模型运行证据。

配套阅读：[总报告](PC_CONVERT_MT6897技术报告.md)、[其他 LLM/ASR/TTS 扩展分析](LLM_ASR_TTS扩展可行性分析.md)。

**2026-11-10 兼容性补充：**MT6897 的公开支持表/target 与仓库支持答复存在未澄清冲突；本专题的 NPU 部署属于待验证方案。设备侧先完成目标 ROM/runtime 的初始化和实际 NPU Run。详细版本、原答复及源码对照见 [DiT 对照报告第 6.1 节](DiT_官方LiteRT与pc_convert适配对比及MT6897落地报告.md)。

读法：按转换操作看第3→4→5→8→9章；理解推理机制重点看第6～7章；面试复习从第10章提问，再回正文找源码依据。

## 1. 这版到底用什么技术转换？

### 1.1 把工程入口、编译技术和运行技术分清

| 层次 | 本包采用的技术 | 负责什么 |
|---|---|---|
| 原始模型 | Hugging Face Transformers + Safetensors | 读取 Qwen 结构、浮点权重、配置与 tokenizer |
| LLM 适配 | LiteRT Torch `generative/export_hf` | 将生成模型包装为 prefill/decode、cache 与辅助图接口 |
| 图捕获 | PyTorch `torch.export`，ExportedProgram/FX 图 | 将 Python 模型前向转成可分析的张量计算图与形状约束 |
| 中间表示与转换 | LiteRT Converter、MLIR，含 StableHLO/TFL 等方言 | 算子分解、布局处理、融合/规整、转换并序列化 TFLite 图 |
| 数值压缩 | AI Edge Quantizer，PTQ | 权重 INT8、代表性数据校准、激活静态 INT16、跨图缓存尺度对齐 |
| 硬件编译 | LiteRT AOT + MediaTek compiler plugin + NeuroPilot SDK | 以 MT6897 为 target，编译可卸载子图并保存目标二进制 |
| 模型容器 | LiteRT-LM builder，`.litertlm` | 组织模型子图、元数据、tokenizer 等运行所需内容 |
| Android 执行 | LiteRT-LM + LiteRT NPU Dispatch + 设备厂商运行库/驱动 | 生成调度、状态管理、NPU 执行、采样与文本输出 |

**本例采用 PyTorch → LiteRT 路线，不经过 ONNX 或 GGUF。** 不需要训练一个“MT6897 专属 Qwen”；训练好的网络经过图转换、数值量化和硬件编译，成为目标 runtime 可消费的部署产物。

```text
config.json + 原始浮点 safetensors + tokenizer/模板
                   ↓ HF Transformers
              Qwen2ForCausalLM
                   ↓ LiteRT Torch LLM wrapper / split-cache
          prefill_128 + decode + aux
                   ↓ torch.export
          ExportedProgram / ATen-FX 图
                   ↓ 分解、FX passes、lowering
         MLIR（StableHLO / TFL 等表示）
                   ↓ LiteRT Converter passes / FlatBuffers
         TFLite 子图 + 初始 W8 权重量化
                   ↓ 打包
             model.litertlm
                   ↓ 真实文本 prefill + decode 校准
       激活范围 + 主图/辅助图 KV 范围对齐
                   ↓ 静态量化
       A16W8 主干 + 必要浮点子图/边界
                   ↓ MediaTek AOT，target=MT6897
          qwen05_mt6897.litertlm
                   ↓ Android LiteRT-LM
        CPU 调度/辅助工作 ↔ NPU 主干计算
```

### 1.2 用源码把“技术栈”落到实处

下文 `LT:` 为 `D:\misc\pc_convert\pc_convert\offline\pip_pkgs\litert_torch-0.10.0-py3-none-any.whl` 中的 `litert_torch/`，`AQ:` 为 Quantizer 0.10.1 wheel 中的 `ai_edge_quantizer/`。行号均为成员源文件行号；相关 LiteRT 源码版权归原作者，采用 Apache-2.0 许可。

`LT:_convert/core.py:105–112`，常规默认分支实际调用 `torch.export`：

```python
elif not strict_export:
  exported_program = torch.export.export(
      sig.module,
      args=args,
      kwargs=kwargs,
      dynamic_shapes=sig.dynamic_shapes,
      strict=False,
  )
```

这一步捕获前向计算；不执行训练，也不生成 MT6897 指令。样例输入确定张量结构与 shape 合同，不能当作量化校准样本。

`LT:_convert/litert_converter.py:164–172`，把捕获结果降低到 MLIR：

```python
lowered_programs = []
for exported_program, sig in zip(exported_programs, signatures):
  # Convert ExportedProgram to Mlir Module.
  with progress.task(f"Lower to MLIR: {sig.name}"):
    lowered = backend.export.exported_program_to_mlir(
        exported_program,
        ir_context=ir_context,
        lowering_context_plugins=[cross_program_inline_consts_ctx],
    )
```

同文件 `224–229`，继续做 LiteRT 转换 passes：

```python
# Run LiteRT converter passes.
with ir_context, progress.task("Run LiteRT Converter Passes"):
  pass_manager = passmanager.PassManager()
  converter_api_ext.run_convert_to_tfl_passes(
      merged_module, pass_manager, config
  )
```

本包的默认 `use_v2=False`；也包含另一条实验 V2 导出桥与 C++ pipeline。本专题保持默认路径。历史资料中的 Torch/XLA 后端不能直接套作这版默认实现；这里可从调用链明确追到 `litert_converter` 的 MLIR 接口。

### 1.3 冻结这份版本表，别把不同年代接口拼起来

| 组件 | 随包版本/条件 |
|---|---|
| PC 平台 | x86_64 Linux、CPython 3.12；当前 Windows 只做分析 |
| PyTorch | 2.14.1+cpu |
| Transformers | 5.22.1 |
| LiteRT Torch | 0.10.0 |
| LiteRT Converter | 0.5.0 |
| AI Edge Quantizer | 0.10.1 |
| AI Edge LiteRT / MediaTek wrapper | 2.3.0 |
| LiteRT-LM builder | 0.19.0 |
| MT6897 对应 SDK 族 | 本地 target 选择 v8，归档含 v8_0_10 |

PC 上安装了 `torch+cpu` 不妨碍生成 NPU 模型：转换端运行的是编译工具；手机端执行的是另一套硬件运行库。两端架构与职责分开。

## 2. 先把这一个 Qwen 的结构讲准确

依据 [Qwen 官方 config.json](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json)：

| 配置 | 数值 | 对部署的意义 |
|---|---:|---|
| `model_type` | `qwen2` | Qwen2.5 权重使用 Qwen2 架构类 |
| Decoder 层数 | 24 | 每层各有 K、V 状态 |
| hidden size | 896 | 残差/embedding 维度 |
| MLP intermediate size | 4864 | FFN 主要矩阵的中间维度 |
| Query heads | 14 | attention 查询头数 |
| KV heads | 2 | GQA，共享 K/V |
| head dimension | 896÷14＝64 | 每个头的向量宽度 |
| vocabulary size | 151936 | embedding 和 logits 维度 |
| 位置上限 | 32768 | 原始配置上限，本次导出容量只选 2048 |
| RoPE theta | 1000000 | 位置编码参数，导出应保留 |
| sliding window | `use_sliding_window=False` | 本例按全注意力缓存讲解 |
| tied embeddings | `True` | 原模型输入 embedding 与输出投影共享权重 |
| 原始配置 dtype | BF16 | 导出默认加载成 FP32；最终 A16W8 是后续量化结果 |

### 2.1 GQA 如何影响图与缓存

忽略 batch/time 维，一个 token 的投影结果为：

```text
x:        [896]
Q = xWq: [14, 64]
K = xWk: [ 2, 64]
V = xWv: [ 2, 64]
```

14 个 Q 头分为 2 组，每组 7 个 Q 头共享一个 KV 头。缓存保存 2 个 KV 头即可；attention 计算时再按组匹配。相对 14 个 KV 头的普通 MHA，相同层数/长度/dtype 下逻辑 KV 容量降为 **1/7**。

RoPE 作用于 Q/K，KV 缓存中的 K 必须与模型的位置编码合同一致。cache 写入位置、有效长度、RoPE position 是相关但不同的量；只移动字节而不维护位置语义，模型照样会生成错误结果。

### 2.2 权重占用与缓存占用要分开算

按官方维度、Q/K/V bias、无输出投影 bias，以及 tied embedding 只计一次，结构参数量为 **494,032,768**。理想纯权重存储量约：

| 每参数存储 | 理想容量 |
|---|---:|
| FP32，4 字节 | 1884.6 MiB |
| BF16/FP16，2 字节 | 942.3 MiB |
| INT8，1 字节 | 471.1 MiB |
| INT4，0.5 字节 | 235.6 MiB |

这张表用于数量级理解。实际文件和运行内存还包括 scales、bias、未量化算子、图/容器、激活、KV、驱动工作区，以及分图后可能新增的常量存储。不能拿“0.5B×1 字节”给整个进程打包票。

## 3. 第一步：准备原始模型和输入合同

### 3.1 转换所需材料

准备模型本地目录，例如 Linux 上 `/work/models/Qwen2.5-0.5B-Instruct`，包含完整的浮点 checkpoint、`config.json`、tokenizer 文件、chat template 与相关生成配置。使用同一 revision，别混搭另一个 Qwen 的 tokenizer。

首轮不使用 GGUF/GPTQ/AWQ 文件，也不以单独 LoRA adapter 代替基础权重。当前 HF 导出入口对带 `quantization_config` 的 checkpoint 有显式拒绝；本例从原始浮点模型开始。

模型加载后仍是一个 PyTorch 网络。`LT:generative/export_hf/core/export_lib.py:421–429`：

```python
else:
  model = auto_model_cls.from_pretrained(
      model_path,
      config=config,
      torch_dtype=model_dtype,
      trust_remote_code=trust_remote_code,
  )
if model_dtype == torch.float32:
  model = model.to(model_dtype)
```

`model_dtype` 默认取 FP32。把原有 BF16 数值扩展成 FP32 便于转换和校准，不会恢复原先 BF16 舍入时已经丢失的信息。

### 3.2 模板和 token 数从一开始就要一致

校准文件使用预先格式化好的 `text` 字段，保留 system、user 和多轮 assistant 历史。示例仅展示一条记录的生成方法，实际数据应包含目标任务、语言、长短文本和不同生成轨迹。

```python
import json
from pathlib import Path
from transformers import AutoTokenizer

model_dir = "/work/models/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
messages = [
    {"role": "system", "content": "你是一位讲解清晰的中文助手。"},
    {"role": "user", "content": "80元的商品打八折，是多少钱？请解释。"},
]
text = tok.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
prompt_ids = tok(text, add_special_tokens=False)["input_ids"]
print("包含模板的 prompt token 数：", len(prompt_ids))

out = Path("/work/qwen05/calibration_text")
out.mkdir(parents=True, exist_ok=True)
with (out / "prompts.jsonl").open("w", encoding="utf-8") as f:
    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
```

汉字数、字符串长度和 token 数不同。容量预算用完整模板编码后的 token 数，包含角色/分隔标记。

选择预格式化 `text` 有具体原因：这版校准器的 `messages` 分支只保留 user 角色。后续必须 `enable_formatting=False`，防止重复套模板；读取任务名设成 `['prompts']`，因为读取器按任务名拼 `prompts.jsonl`，不会扫描目录自动找文件。

## 4. 第二步：导出固定形状的 prefill/decode 与 aux

### 4.1 本例固定参数

| 参数 | 本例值 | 管什么 |
|---|---|---|
| `model_id` | 原始 Qwen 本地目录 | 结构、权重与 tokenizer 来源 |
| `target_vendor` | `mediatek` | 编译后端及 vendor 布局 |
| `target_soc` | `MT6897` | 明确目标芯片，覆盖厂商默认值 |
| `prefill_lengths` | `[128]` | 每次 prefill 图的输入块大小 |
| `cache_length` | `2048` | 每层物理 past KV 的时间轴容量 |
| batch size | `1` | 单会话教学配置 |
| `split_cache` | `True` | 主图输出新 KV，aux/运行器负责更新历史缓存 |
| dynamic shape | `False` | 本版 split-cache 导出明确拒绝动态形状 |
| `k_ts_idx` / `v_ts_idx` | `2` / `3` | MediaTek 的 K/V 时间轴位置 |
| `weight_quantization_recipe` | `dynamic_wi8_afp32` | 第一阶段权重量化；激活仍浮点 |
| `max_calibration_decode_steps` | `32` | 每条校准轨迹最多生成步数，允许提前 EOS |
| 产品示例生成预算 | `128` | Android/业务层生成上限，独立于校准步数 |

保持单一 cache 档位。多 prefill 块长和多 cache 容量是两件事；当前校准 loader 对带 `_cache_2048` 后缀的多容量签名存在解析限制，本例不走那条分支。

### 4.2 真实 API 接线

以下代码展示第一阶段；后续章节在同一 `cfg` 与路径下接校准、量化、编译。

```python
from litert_torch.generative.export_hf.experimental.npu_export.config_manager import (
    build_pipeline_config,
)
from litert_torch.generative.export_hf.experimental.npu_export.stages import (
    npu_export, npu_calibrate, npu_quantize, npu_compile,
)

cfg = build_pipeline_config(
    model_id="/work/models/Qwen2.5-0.5B-Instruct",
    target_vendor="mediatek",
    target_soc="MT6897",
    prefill_lengths=[128],
    cache_length=2048,
    max_calibration_decode_steps=32,
    calibration_dataset_format="jsonl",
    calibration_eval_task_names=["prompts"],
    use_float_input_output_normalizer=False,
    skip_mlir_passes=True,
)
cfg.save_json("/work/qwen05/pipeline.json")

exported = npu_export(
    cfg,
    output_dir="/work/qwen05/export",
    keep_temporary_files=True,
)
# exported == "/work/qwen05/export/model.litertlm"
```

`npu_export` 实际打开 split-cache，`LT:generative/export_hf/experimental/npu_export/stages.py:68–73`：

```python
overrides = {
    "output_dir": output_dir,
    "split_cache": True,  # Mandatory for NPU backends
    "keep_temporary_files": keep_temporary_files,
    **kwargs,
}
```

主图导出选择独立 attention 实现，`LT:generative/export_hf/core/export_lib.py:614–620`：

```python
if export_config.split_cache:
  assert (
      not has_dynamic_shape
  ), 'Dynamic shape is not supported for split cache.'
  model.set_attn_implementation('lrt_split_cache_attention')
  # In case of the attn_implementation is not set.
  cast(Any, model).config._attn_implementation = 'lrt_split_cache_attention'
```

因此这个方案改动的是**部署前向接口和 attention/cache 表达**，模型权重仍来自原始 Qwen。不能把普通 `model.generate()` 整段 Python 循环直接理解成导出的单张 NPU 图。

第一阶段产物中的 `prefill_128` 和 `decode` 是同一模型的不同签名/子图入口；aux 承接 mask、位置相关数据和 cache 更新等辅助工作。`.litertlm` 是打包容器，扩展名本身不能证明里面已经包含 MT6897 编译结果。

## 5. 第三、四步：真实校准，再把主干静态量化为 A16W8

### 5.1 动态与静态：这版是分阶段完成的

| 阶段 | 输入 | 做什么 | 结果 |
|---|---|---|---|
| 初始导出/权重量化 | 原始浮点HF模型 | 对适用权重做对称、per-channel INT8量化 | W8权重、FP32激活接口为主的中间图 |
| 校准 | 上一步bundle＋真实提示串 | 在PC interpreter上走prefill/decode/aux轨迹，收集中间tensor范围 | 模型级min/max JSON和校准状态 |
| 静态量化 | 中间bundle＋统计 | 保留已有W8，量化适用激活/bias，统一KV范围 | 主干A16W8，保留规定的浮点例外 |
| AOT | 静态量化bundle | 用MT6897后端编译主图 | 目标芯片产物 |

**A16＝INT16激活，W8＝INT8权重。** 此处16位激活采用静态scale；它与FP16推理的数值合同不同。

`dynamic_wi8_afp32`也需要讲准确：图接口中的activation保留FP32，recipe的compute precision为INTEGER，runtime的hybrid kernel可以在算子内部临时量化激活。不能从“AFP32”推导每次乘法都是FP32，也不能把它与显式反量化权重后做浮点运算的weight-only路径混为一谈。

### 5.2 Stage 3真的没有回读原始浮点权重

`LT:generative/export_hf/experimental/calib/quantize.py:242–251` 相关节选：

```python
unpacked = litertlm_utils.unpack_litertlm(input_litertlm, unpack_dir)
model_path = model_path or unpacked.get('tf_lite_prefill_decode')
aux_model_path = aux_model_path or unpacked.get('tf_lite_aux')
embedder_path = embedder_model_path or unpacked.get('tf_lite_embedder')
```

输入来自第一阶段包。量化器对FC/Conv已有量化权重跳过再次处理，`AQ:algorithms/uniform_quantize/common_quantize.py:549–559`：

```python
ignored_inputs = [bias_index]  # Bias tensor is quantized separately.
should_ignore_weight = False
if graph_info:
  w_tensor = graph_info.subgraph_tensors[op_info.op.inputs[weight_index]]
  should_ignore_weight = check_if_quantized(w_tensor)
if should_ignore_weight or _are_weights_too_small(
    op_info, graph_info, weight_index
):
  ignored_inputs.append(weight_index)
```

后续bias计算沿用该权重的scale/zero-point/axis。因此第一阶段W8的误差不会被第二轮校准“修回来”。若改成从纯浮点图一次性静态量化，要显式关闭第一阶段权重量化并重新校准，不能复用另一张图的统计。本专题保留现有默认链，便于与源码逐步对应。

### 5.3 校准收集什么，32步控制什么

本例使用CPU profiler校准，默认取观测min/max包络；无需标签或反向传播。代表性数据应覆盖中文任务、数字、代码/英文混输、短长prompt和多轮模板。数据量先以覆盖主要任务和长度区间为原则；单条示范JSONL只证明格式。

校准调用接第4章的 `cfg` 与 `exported`：

```python
npu_calibrate(
    cfg,
    input_litertlm=exported,
    calibration_dataset_dir="/work/qwen05/calibration_text",
    calibration_result_save_dir="/work/qwen05/calibration_stats",
    enable_formatting=False,  # 第3章已经由Qwen tokenizer生成完整模板
)
```

`LT:generative/export_hf/experimental/calib/calibrate.py:314–322` 中的实际调用节选：

```python
prompt = quant_utils.get_example_prompt(
    example, enable_formatting, executor.tokenizer
)
executor.sample_text(prompt, max_sample_step=max_calibration_decode_steps)
```

每条prompt最多走32次decode调用，EOS或容量边界可能提前停止。它与校准样本数量、prefill128、cache2048和产品输出128分别独立。采样器走自回归生成，未用标准答案做teacher forcing。

观测统计更新为“旧最小值与新最小值取更小，最大值取更大”。`AQ:utils/qsv_utils.py:105–122` 中的两条更新式：

```python
updated_qsv["min"] = np.minimum(qsv["min"], new_qsv["min"])
updated_qsv["max"] = np.maximum(qsv["max"], new_qsv["max"])
```

默认profiler路径只支持min/max更新，配置中的 `ema_smoothing_factor=0.1` 此时不参与统计。该链没有自动做AWQ、GPTQ、QAT、KL阈值搜索或百分位裁剪；看到“量化”两个字，别把所有算法都装进去。

主图prefill/decode及实际调用的辅助签名都会产生统计。**aux的JSON也必须保留**；当前代码只有在aux模型与其统计同时存在时才生成新aux。缺统计时可能留下 `aux_output_path=None`，这对依赖split-cache辅助图的部署有实际影响。

### 5.4 具体怎么从实数变整数

本例静态recipe的主要粒度：

- 激活：有符号INT16、对称、per-tensor，一般一个tensor一个scale。
- 权重：有符号INT8、对称、per-channel。FC权重形状为`[out_features,in_features]`时按axis0，每个输出通道一套scale。
- bias：与输入scale及对应权重通道scale配合。

普通对称量化可写为：

```text
b = max(|min|, |max|)
s = b / qmax                 # 实现对极小b还设正下界
z = 0
q = clip(round(x/s), qmin, qmax)
x_hat = s × q
```

本量化器对称常量的narrow-range规则为W8使用`[-127,127]`，16位对称量化通常使用`[-32767,32767]`。存储类型INT16仍能表示-32768；具体算子运行时舍入/饱和要遵守kernel合同。

**权重例子：** 一行权重`[-1.0,-0.3,0.2,0.8]`，`s=1/127≈0.007874016`：

| 原值 | INT8 | 反量化值 |
|---:|---:|---:|
| -1.0 | -127 | -1.000000 |
| -0.3 | -38 | -0.299213 |
| 0.2 | 25 | 0.196850 |
| 0.8 | 102 | 0.803150 |

另一输出通道的最大绝对值若为0.1，就用`0.1/127`，无需被第一行的大幅值拖粗步长。

**激活例子：** 校准得到`min=-2,max=3`，对称范围取`[-3,3]`，`s=3/32767≈0.0000915555`：

- `x=0.75 → q=8192 → x_hat≈0.7500229`。
- `x=4 → q=32767 → x_hat=3`，出现饱和，1.0的差值已丢失。

16位能细分已选区间，不能自动挽救区间外的值。把范围放宽会减少裁剪，同时增大舍入步长；本例 `calibration_range_scale=1.0` 保持原统计，不预设“越大越好”。

**Bias例子：** `s_input=0.01`、某通道`s_weight=0.02`，则`s_bias=0.0002`，`bias=0.1`对应整数500。此AEQ实现先按INT32量化bias，A16路径再转存INT64；这说明模型bias表示，不能据此断言MT6897硬件使用64位累加。

Softmax等算子还有固定输出量化合同。例如该实现的16位Softmax输出使用`scale=1/32768,z=0`，不会对每个输出都套上述absmax公式。转换后的tensor metadata与kernel约束才是最终接口。

### 5.5 KV范围对齐：决定多步生成能不能保持正确

同一个整数100，scale=0.01表示1.0，scale=0.02表示2.0。把前者的整数直接交给后者解释，shape和dtype都正确，数值却错了一倍。

本版逐层、逐K/V执行：

1. 收集主图全部prefill/decode签名中同一层的`kv_cache_k_i`、`kv_slice_k_i`。
2. 沿同尺度约束节点逆向找到实际需要统一统计的tensor。
3. 对**主图**这些tensor取min/max并集，写回各自统计。
4. 将这个主图范围覆盖到aux对应K tensor。
5. 对V重复，再处理下一层。

它没有把24层的K/V全压成一个scale，也没有计算main+aux的全体范围并集。方向是main→aux，基于aux执行状态更新/搬运这一语义。

`LT:generative/export_hf/experimental/calib/quant_utils.py:613–624` 节选：

```python
min_val, max_val = alignment_utils.align_quant_stats(
    calibration_results, signature_data_main
)
if aux_alignment_utils and aux_calibration_results is not None:
  # …（省略aux对应tensor的收集）
  aux_alignment_utils.update_quant_stats(
      aux_calibration_results, signature_data_aux, min_val, max_val
  )
```

某层K的三个范围若为`[-1,1.5]`、`[-2,1]`、`[-1.8,1.2]`，并集为`[-2,1.5]`；对称A16使用`scale=2/32767`，对应aux K也使用相同范围。另一层、V各自计算。

若改变aux语义，让它对状态做额外数值变换，这种“用主图范围覆盖aux”的假设也必须重审。量化尺度是接口的一部分，不能只写在优化笔记里。

### 5.6 真正生效的开关与浮点例外

虽然vendor配置有 `allow_float_operations=True`，四阶段wrapper按目标函数签名绑定参数；Stage3实际接收的是 `use_float_input_output_normalizer`。`LT:generative/export_hf/experimental/calib/quantize.py:316–325` 节选：

```python
q_main = quantizer.Quantizer(model_path)
q_main = quant_utils.add_main_model_quant_recipe(
    q_main,
    allow_float_operations=use_float_input_output_normalizer,
    a16w8=True,
)
```

本例显式选择False，跟随pipeline API默认，使主路径与normalizer相关边界进入A16方案。**False仍保留RMSNorm内部等浮点例外**；命名匹配embedding的主图输入也有免量化处理，Stage3复用原embedder。shape/index张量自然也不遵循“所有东西都16位”。

True会把若干composite、特定post-QKV ADD和skip-scale MUL保留浮点；它属于量化recipe选择，不提供“任何不支持算子自动回CPU”的保证。直接运行底层quantize CLI与通过API的默认值还有差异，教程显式写参数，避免默认值替读者做决定。

`skip_mlir_passes=True`跳过该阶段可选的Q/DQ融合和quantized BMM融合后处理，保留本版默认；它不表示前面的MLIR转换消失，也不表示后续硬件编译器停止优化。

### 5.7 静态量化调用与产物

```python
npu_quantize(
    cfg,
    input_litertlm=exported,
    output_litertlm="/work/qwen05/quantized.litertlm",
    calibration_dir="/work/qwen05/calibration_stats",
    use_float_input_output_normalizer=False,
    skip_mlir_passes=True,
)
```

这里不传Stage3 `quantization_recipe`，让其使用默认NPU静态规则。Stage1接受recipe名称，Stage3显式recipe的装载合同不同，不能把第一阶段字符串照抄过去。

当前这一步重新生成主图/aux并打包，embedder、tokenizer等沿用输入包。操作时让每组导出图、校准JSON和量化产物对应在独立目录；修改模型、shape、模板或量化前输入图后，不能继续拿旧统计当同一份证据。

## 6. Prefill/decode：沿 130 个输入 token 走一遍

### 6.1 先说原理，再看这版特殊接口

Prefill 批量处理已知 prompt，建立各层历史表示。Decode 每次消费一个已知 token，读取历史 K/V，计算下一个 token 的概率。两者共享权重；区别在输入长度、状态和执行安排。

attention 可写成 `softmax(QKᵀ/√64 + mask)V`。缓存保存历史 K/V，省去重新计算历史 token 的投影及前面各层；每个新 token 仍要与允许读取的历史做 attention。KV cache 没有把整段 attention 成本变成常数。

**本包 split-cache prefill 不输出 logits。** `LT:generative/export_hf/core/split_cache/exportable_module.py:249–252`：

```python
inputs |= self.attention_kwargs()
output = self.model(**inputs)
output_cache = output.past_key_values
return self.post_process_kv_cache(output_cache)
```

decode 同文件 `294–299` 才保留 logits：

```python
inputs |= self.attention_kwargs()
output = self.model(**inputs)
output_cache = output.past_key_values
ret = self.post_process_kv_cache(output_cache)
ret['logits'] = output.logits
return ret
```

所以首个新 token 的概率仍来自最后一个 prompt 位置，但要经 decode 签名取得。随包 PC 校准采样器会 prefill 完全部 prompt，再回退一位重算；后文引用的 Android C++ runtime 则把最后一个 prompt token 留给 decode。两种调度都围绕同一输出合同，执行细节不能混写。

### 6.2 主图实际吃 embedding，外部生成位置与 mask

`LT:generative/export_hf/core/exportable_module_config.py:397–402`：

```python
if self.split_cache:
  self.externalize_embedder = True
  self.externalize_rope = True
  if self.cache_implementation is None:
    self.cache_implementation = "LiteRTLMSplitCache"
  self.moe_exports_implementation = "litert_moe_sequential"
```

本模型是 dense；末行是统一配置赋值，不会把 Qwen 变成 MoE。

以下为源码生成的逻辑 shape，batch=1、T=2048；dtype 和最终平铺名字应从产物 SignatureDef 读取：

| 模块/签名 | 主要输入 | 主要输出 |
|---|---|---|
| `prefill_embedder_128` | token IDs `[1,128]` | embedding `[1,128,896]` |
| `prefill_rope_128` | 绝对位置 `[128]` | cos/sin 各 `[1,128,1,64]` |
| `prefill_mask_128` | token、游标、valid mask | mask `[1,1,128,2176]` |
| `prefill_128` | embedding、RoPE、mask、各层历史 KV | 各层128位置的新 K/V slice |
| `prefill_cache_update_128` | 历史 KV、新 slice、位置 | 更新后的历史 KV |
| `decode_embedder` / `decode_rope` / `decode_mask` | 单 token、绝对位置 | embedding、RoPE、`[1,1,1,2049]` mask |
| `decode` | 上述结果、各层历史 KV | `[1,1,151936]` logits、单位置 K/V slice |
| `decode_cache_update` | 历史 KV、单位置 slice、位置 | 更新后的历史 KV |

外置 RoPE 使用绝对位置生成 cos/sin。看到主图内部的局部 `arange(P)`，不能据此认定每块都把绝对位置重置成 0。

### 6.3 Split cache 的计算顺序

MediaTek 布局中，每层：

```text
旧 K：[1,2,2048,64]       新 K：[1,2,P,64]
旧 V：[1,2,64,2048]       新 V：[1,2,64,P]
```

新 Q 分别与旧 K、新 K 相乘，拼分数后做共同 softmax。`LT:generative/export_hf/core/split_cache/attention.py:78–80,94–97`，两处相关节选：

```python
logits0 = bmm_fn(query, key_past)
logits1 = bmm_fn(query, key)
logits = torch.cat([logits0, logits1], dim=-1)
```

```python
padded_logits = logits + mask
probs = F.softmax(padded_logits, dim=-1).type_as(key)
probs0, probs1 = probs[..., :-t], probs[..., -t:]
```

随后两段概率分别乘对应 V，结果相加。**不能给两段分别做 softmax 再相加**，那会改变概率归一化。

因为旧缓存有 T 个物理槽、新块有 P 个位置，mask 宽度为 `T+P`。这解释了 prefill128 的 2176、decode 的 2049。有效历史由 mask 限制，物理 buffer 长度不随会话每步重新分配。

旧历史仅开放 `j < time_step`；本块内部用因果三角与 padding 有效标记。该源码把屏蔽项设成有限负值 `-100.0`，不要把它背成字面上的 `-inf`。

### 6.4 KV 更新和 padding：读懂逻辑位置

主图返回新 slice，aux 负责写回完整 cache。`LT:generative/export_hf/core/cache.py:135–144`：

```python
positions = cache_position[0]  # The position of the first input token.
k_slice_indices = _get_slice_indices(positions.clone(), cache_dim, k_ts_idx)
v_slice_indices = _get_slice_indices(positions.clone(), cache_dim, v_ts_idx)
k = tfl_dus.dynamic_update_slice(
    key_state, k_slice, [x for x in k_slice_indices]
)
v = tfl_dus.dynamic_update_slice(
    value_state, v_slice, [x for x in v_slice_indices]
)
return k, v
```

这里按第一个位置连续更新；它没有实现任意位置的 scatter。K 与 V 的时间轴不同，不能用同一个 stride 盲拷贝。

本 Qwen 的 full-attention 路径会把 padding slice 一并写入预留槽。逻辑游标只按真实 token 前进，未来槽被 mask 挡住，后续真实 token 到来时覆盖。**缓存里存在一个数值，不等于这个位置已经成为可读历史。**

### 6.5 完整阶段表：N=130、P=128、T=2048、G=128

这里假设完整模板经 tokenizer 后有130个 token；`x0…x129` 是 prompt，`y0…` 是新输出。此表按**本地 PC 校准采样器**展开：

| 阶段 | 输入与位置 | 允许读取的旧历史 | 写回与输出 |
|---|---|---|---|
| 初始化 | 游标0 | 无 | 分配/初始化24层历史 KV |
| 第1次 prefill | x0…x127，位置0…127 | 无 | 写0…127；游标128 |
| 第2次 prefill | x128、x129＋126个pad，位置128…255 | 0…127 | 整块写128…255；真实长度130；游标回退至129 |
| 第1次 decode | 重算x129，位置129 | 0…128 | 覆盖槽129；采样y0，预测位置130 |
| 第2次 decode | 输入y0，位置130 | 0…129 | 覆盖原pad槽130；采样y1 |
| 第128次 decode | 输入y126，位置256 | 0…255 | 写256；采样y127，预测位置257 |

没有 EOS 提前结束时，2次 prefill＋128次 decode，得到128个新 token。结束时最后输出 `y127` 尚未作为输入处理，KV 只覆盖有效输入位置0…256。继续生成时要保留这个 pending token，单独保存 KV 不够。

为何重算 x129 不会看见两份自己？游标129时，旧历史只允许 `<129`；旧槽129被遮住，当前 x129 从新 slice 分支参加 attention，之后覆盖同一槽。

游标回退的原文在 `LT:generative/export_hf/experimental/calib/sampling_executor.py:537–543`：

```python
processed_token_length = min(remaining_input_size, input_size)
time_step += processed_token_length
if time_step == num_input_tokens:
  generate = True
  time_step -= 1
elif time_step < num_input_tokens:
  generate = False
```

### 6.6 能处理非128长度吗？能，但要有成套调度

本地采样器对130 token调用两次128签名；20 token只调用一次，补108个pad，并传有效标记。`prefill_lengths=[128]` 表示图的块长，不代表 prompt 上限128。

T=2048可被128整除，从空会话开始且 `1≤N<2048` 时，末块补齐仍不越界。追加已有会话、调整T/P时，还必须检查 `块起点+P≤T`，不能只看真实 token 数。当前 PC 采样器会拒绝 `N≥T`。

可以另导出 `[32,128]` 减少短尾浪费，但 embedding、RoPE、mask、update 必须切换到匹配签名。固定形状通常更利于 NPU 编译/内存规划，代价是 padding 和多签名管理。

### 6.7 最小生成状态机：面试可以手写这段

这是逻辑伪代码，省略量化 Q/DQ 与具体 API 包装；顺序遵守本地 split-cache 合同：

```text
ids = tokenize(chat_template(messages))
N = len(ids)
assert 1 <= N < T and 1 <= G and N + G <= T
cache = init_cache()
s = 0
while s < N:
    n = min(P, N-s)
    block, valid = right_pad(ids[s:s+n], P)
    pos = range(s, s+P)
    assert s+P <= T
    emb = embed_prefill(block)
    rope = rope_prefill(pos)
    mask = mask_prefill(block, s, valid)
    slices = prefill(emb, rope, mask, cache)
    cache = update_prefill(cache, slices, pos, valid)
    s += n

s = N-1
pending = ids[-1]
for generated_count in range(G):
    emb = embed_decode(pending)
    rope = rope_decode(s)
    mask = mask_decode(pending, s)
    logits, slices = decode(emb, rope, mask, cache)
    cache = update_decode(cache, slices, s)
    token = sample(logits)       # baseline可用greedy/argmax
    emit(token)
    s += 1
    pending = token
    if token in stop_ids:
        break
```

采用保守预算 `N+G≤T`，避免把“最后一次可以预测一个尚未写入 cache 的 token”当成普遍的额外上下文容量。停止后继续会话，需要保存位置、pending token、模型/量化版本以及必要采样状态。

## 7. KV cache 的容量、dtype 和性能账

### 7.1 2048 到底占多少内存

密集未压缩 KV 的逻辑容量：

```text
M = 2 × batch × layers × positions × KV_heads × head_dim × element_bytes
  = 2 × 1 × 24 × T × 2 × 64 × element_bytes
```

| cache位置数T | FP32 KV | 16位 KV主体（FP16或INT16） | 8位 KV主体 |
|---:|---:|---:|---:|
| 512 | 12 MiB | 6 MiB | 3 MiB |
| 1024 | 24 MiB | 12 MiB | 6 MiB |
| 2048 | 48 MiB | 24 MiB | 12 MiB |
| 4096 | 96 MiB | 48 MiB | 24 MiB |

16位元素下，每个新 token 的24层 K/V合计 **12 KiB**；128 token的新slice合计 **1.5 MiB**。T=2048时，每层 K/V各512 KiB，两者合计1 MiB，24层共24 MiB。

这些只计主体；量化参数、对齐、输入输出副本与工作区另算。FP16和INT16虽然都占2字节，数值表示完全不同，不能互相解释字节。

### 7.2 初始导出、PC校准、Android部署可能使用不同存储

初始导出默认FP32 cache。PC采样器即使面对量化图，也以FP32保存外部状态，在调用边界转换。`LT:generative/export_hf/experimental/calib/sampling_executor.py:297–305`：

```python
def init_cache(self) -> dict[str, np.ndarray]:
  """Init cache to zeros."""
  input_details = self.decode_runner.get_input_details()
  kv_cache = {
      k: np.zeros(input_details[k]['shape'], dtype=np.float32)
      for k in input_details
      if 'cache' in k
  }
  return kv_cache
```

同文件 `864–866`：

```python
in_kwargs = try_get_quantized_input(signature_input, signature_runner)
out_kwargs = signature_runner(**in_kwargs)
return try_get_dequantized_output(out_kwargs, signature_runner)
```

所以PC校准器的48 MiB外部缓存、反复Q/DQ与Python耗时，不能拿来冒充Android NPU的实际缓存占用/速度。Android应按最终张量dtype与scale/zero-point分配、绑定和更新状态。

### 7.3 cache调大为什么可能更慢

- 状态容量线性增加；2048→4096使本模型16位KV从24→48 MiB。
- 静态 mask 和 attention 的物理形状也变大。虽然有效历史可能只有130，后端仍可能对固定2048槽做稠密计算；mask保证语义，不自动保证跳过计算。
- 融合、内存规划和主机/设备复制影响实际峰值。不能从源码返回新cache就认定原地更新，也不能默认零拷贝。
- 小模型decode每步只有一个token，权重读取、KV访问、调度/同步容易成为瓶颈；prefill能在一块已知token上并行，矩阵计算利用率通常更高。

作为可复算的形状例子，prefill128的FP32 mask为 `128×2176×4=1.0625 MiB`。若单层14头score全部物化，FP32中间量为14.875 MiB；实际是否分配这整个张量由融合与内存计划决定。

**推荐先用2048把完整生成做对。** 需要4096时同步更改静态图、mask、校准覆盖与AOT产物，不能仅把手机CLI中的上下文数字调大。

## 8. 第五步：为 MT6897 做 AOT，再交给 Android 消费

### 8.1 目标选择有两层：SoC与SDK

本地 LiteRT 2.3.0的 `ai_edge_litert/aot/vendors/mediatek/target.py:35–40,48–54` 节选：

```python
MT6895 = "MT6895"
MT6897 = "MT6897"
MT6983 = "MT6983"
MT6985 = "MT6985"
MT6989 = "MT6989"
MT6991 = "MT6991"
```

```python
match soc_model:
  case SocModel.MT6993:
    return "v9"
  case _:
    return "v8"
```

MT6897进入v8系列，SDK wrapper再解析到 `v8_0_10/host/lib`。枚举说明有这个编译目标；模型算子/shape/dtype与手机ROM兼容性仍由后续编译和真实Run决定。同属v8的MT6991产物不能直接冒充MT6897产物。

### 8.2 Qwen应关掉默认Gemma专项优化

本地 package compiler 的 MediaTek默认值带有Gemma优化。`LT:generative/export_hf/experimental/litert_lm_npu_compiler/litert_lm_npu_compiler_configs.py:56–64` 节选：

```python
'mediatek': {
    'prefill_decode': {
        'compile': True,
        'flags': [
            '--mediatek_enable_gemma_compiler_optimizations=true',
            '--mediatek_performance_mode_type=turbo_boost',
            '--mediatek_enable_l1_cache_optimizations=true',
            '--mediatek_optimization_hint=low_latency',
```

本例显式提供一份清晰配置：只编译主图，关闭Gemma专项优化，aux以及若存在的text_encoder section保留原图。此处的text_encoder可能承载embedding等模型部件，不能从名字推断Qwen变成了encoder-decoder模型。

接前面的 `cfg` 和静态量化产物：

```python
npu_compile(
    cfg,
    input_litertlm="/work/qwen05/quantized.litertlm",
    output_litertlm="/work/qwen05/qwen05_mt6897.litertlm",
    compile_configs={
        "prefill_decode": {
            "compile": True,
            "flags": ["--mediatek_enable_gemma_compiler_optimizations=false"],
        },
        "aux": {"compile": False},
        "text_encoder": {"compile": False},
    },
    disable_aux_compilation=True,
    disable_text_encoder_compilation=True,
    intermediate_dir="/work/qwen05/compile_intermediates",
    keep_temporary_files=True,
)
```

`compile_configs` 在这版整体替换默认配置，没有自动深合并。MediaTek在当前wrapper中不能打开aux编译；text_encoder若被编译还可能被追加Gemma decode bundle，因此这里明确禁用这两类编译。

AOT的实际工作是：解包`.litertlm`→按section取TFLite子模型→ApplyPlugin/MediaTek compiler进行分区与目标编译→替换为带Dispatch/编译字节码的图→重新打包。它不会训练权重，也不会自动补齐缺失的tokenizer或生成循环。

按上述示例执行时，各阶段产物用途如下：

| 路径 | 用途 |
|---|---|
| `/work/qwen05/pipeline.json` | 固定本轮模型、target、shape与校准参数 |
| `/work/qwen05/export/model.litertlm` | 第一阶段W8中间包，供校准和静态量化读取 |
| `/work/qwen05/calibration_stats/` | 主图和aux等模型的统计JSON，必须对应当前导出图 |
| `/work/qwen05/quantized.litertlm` | 静态量化主图/aux与其他组件组成的包 |
| `/work/qwen05/compile_intermediates/` | AOT解包清单和各子模型编译中间物 |
| `/work/qwen05/qwen05_mt6897.litertlm` | 最后交给Android运行器的目标包 |

### 8.3 三种文件别混淆

- **`.tflite`**：一个或多个签名的计算图与常量，可以含编译分区。
- **`.litertlm`**：LLM完整管线容器，包含子模型、tokenizer/模板和元数据。
- **`.dla`**：厂商层的编译网络表示。

本例交付AOT后的完整`.litertlm`给LiteRT-LM。只抽一个`.dla`给原工程的一次性MLP探针，会丢掉生成所需的状态与调度合同。原始 `aot_on_apu.cpp` 和 `litert_npu_check.cpp` 也不能直接充当Qwen聊天程序。

## 9. 第六步：Android上怎样真正运行NPU生成

### 9.1 选已有生成运行器

推荐官方 **`litert_lm_advanced_main`** 做首轮设备诊断：它已有tokenizer、生成循环、上下文/输出上限和NPU日志接口。手机上要运行Android arm64/bionic的程序与库，PC的Linux x86_64 host SDK留在PC。

本地包中的 `litert_lm_builder` 负责打包，未提供完整Android runtime。本次参考的官方LiteRT-LM源码固定为 `458390c0456e119db35d43d31862dbbf241e2c1a`，其WORKSPACE固定LiteRT提交 `03923b053070e809c10e2b1f855017d78842f94f`。[固定版本WORKSPACE](https://github.com/google-ai-edge/LiteRT-LM/blob/458390c0456e119db35d43d31862dbbf241e2c1a/WORKSPACE)

这是一份明确的消费端源码参考，尚未与本地导出产物做设备联调。部署时把runner、LiteRT core、MediaTek Dispatch固定为配套构建，核对section、签名、dtype和metadata，不能只凭各包名字中都有LiteRT就混装。

### 9.2 构建与运行模板

下面在Linux上的固定版本LiteRT-LM源码根目录执行；需要已准备相应Android构建环境和MT6897 AOT产物。此处只给操作示例，本次未执行构建或adb操作。

```bash
bazel build --config=android_arm64 //runtime/engine:litert_lm_advanced_main
bazel build --config=android_arm64 \
  @litert//litert/vendors/mediatek/dispatch:dispatch_api_so

DEVICE_FOLDER=/data/local/tmp/qwen05_mt6897
adb shell mkdir -p "$DEVICE_FOLDER"
adb push /work/qwen05/qwen05_mt6897.litertlm "$DEVICE_FOLDER/model.litertlm"
adb push bazel-bin/runtime/engine/litert_lm_advanced_main "$DEVICE_FOLDER/"
adb push bazel-bin/external/litert/litert/vendors/mediatek/*/*.so "$DEVICE_FOLDER/"
adb push prebuilt/android_arm64/*.so "$DEVICE_FOLDER/"
adb shell chmod 755 "$DEVICE_FOLDER/litert_lm_advanced_main"

adb shell "LD_LIBRARY_PATH=$DEVICE_FOLDER \
  $DEVICE_FOLDER/litert_lm_advanced_main \
  --backend=npu \
  --model_path=$DEVICE_FOLDER/model.litertlm \
  --litert_dispatch_lib_dir=$DEVICE_FOLDER \
  --max_num_tokens=2048 \
  --max_output_tokens=128 \
  --enable_npu_debug_logging=true \
  --input_prompt='请用三句话解释KV缓存为什么能加速大模型推理。'"
```

伴随`.so`的实际集合以该构建和官方部署说明为准；静态/动态链接方式会影响是否另带core library。需要设备ROM提供的厂商runtime/驱动，不能把另一款手机的vendor库整包复制过来“补兼容”。AOT-only运行通常不需要手机重新执行PC侧编译。

MediaTek库部署方式见[官方NPU指南](https://developers.google.com/edge/litert/next/litert_lm_npu#mediatek-neuropilot)。该文档展示的现成模型为Gemma3-1B的MT6989/MT6991变体，未提供本例Qwen/MT6897的现成成功记录。advanced入口对输出上限的消费另从[固定CLI源码](https://github.com/google-ai-edge/LiteRT-LM/blob/458390c0456e119db35d43d31862dbbf241e2c1a/runtime/engine/litert_lm_advanced_main.cc#L285-L286)确认。

### 9.3 CPU和NPU各干什么

推荐首版：CPU承担文本模板、分词、会话调度、embedding/辅助工作与采样；NPU执行已编译Transformer主图。KV由运行器的buffer和更新路径管理，不必假设每一步经Python或回传整块给应用。

在上述C++版本中，prompt最后一个token作为pending留给decode；其余token按prefill路径处理。prefill和decode都调用真实的 `CompiledModel::Run`，随后更新缓存。它与第6章PC采样器的“先全prefill、再回退一位”是独立实现。[NPU executor固定源码](https://github.com/google-ai-edge/LiteRT-LM/blob/458390c0456e119db35d43d31862dbbf241e2c1a/runtime/executor/llm_litert_npu_compiled_model_executor.cc#L1687-L1727)

运行器的部分wrapper请求NPU与CPU两种加速器，所以`aux compile=False`只代表没有提前编译它；最终辅助图后端仍要看运行日志。[后端选择源码](https://github.com/google-ai-edge/LiteRT-LM/blob/458390c0456e119db35d43d31862dbbf241e2c1a/runtime/executor/npu/llm_litert_npu_compiled_model_executor_utils.cc#L428-L429)

另一个命名坑：`use_hw_cache_update_for_npu`中的`hw`指**handwritten，手写实现**，不能翻译成硬件NPU更新。本例不主动调整它，遵循选定runtime与模型包的默认合同。[参数定义](https://github.com/google-ai-edge/LiteRT-LM/blob/458390c0456e119db35d43d31862dbbf241e2c1a/runtime/engine/shared_flags.cc#L255-L256)

### 9.4 什么证据能说明“真的用NPU跑了”

需要连起来看到：MT6897编译目标正确；prefill/decode关键计算进入目标分区；设备加载MediaTek Dispatch与网络成功；实际Run完成；输出结果与相同量化图参考一致。设备允许时再结合厂商profiling/APU提交完成记录确认。

仅有`.litertlm`文件、NPU设备枚举、CompiledModel创建成功或`--backend=npu`参数，都不够说明主干实际执行位置和收益。

数值排错分三层：

1. **浮点HF vs 导出浮点图**：相同token/position/mask下查结构、布局与cache语义。
2. **导出参考 vs 静态量化图**：查量化引入的logits/任务质量变化与长decode累积误差。
3. **同一量化图参考 vs MT6897产物**：查后端实现、分区、同步和状态更新。

先比较相同前缀下的logits或greedy生成，有助于隔离差异；采样随机性本身会放大文本轨迹差异。性能记录TTFT、prefill吞吐、每token decode延迟、峰值内存和持续运行热状态，前后处理与同步都计入端到端。

## 10. 面试问答：从一分钟概述到源码追问

### A. 技术栈与转换流程

**Q1：这套方案用什么技术把Qwen转成MT6897模型？**  
使用LiteRT Torch的HF生成模型前端，用torch.export捕获ATen/FX图，经MLIR与LiteRT Converter生成TFLite子图；AI Edge Quantizer做PTQ，MediaTek AOT/NeuroPilot生成MT6897目标产物，LiteRT-LM负责Android上的生成过程。

**Q2：为什么不能直接运行convert_dit.py换个模型名？**  
它的加载器、输入和校准绑定DiT。Qwen需要token/embedding、位置、mask、KV、prefill/decode和生成runtime，必须走LLM专项入口。

**Q3：为什么PC安装CPU版torch，最后能在NPU跑？**  
PC执行图捕获、校准和编译；部署产物由手机的LiteRT/MediaTek运行库执行。编译器运行在哪个处理器上，与生成代码面向哪个处理器是两件事。

**Q4：`.safetensors`、`.tflite`、`.litertlm`、`.dla`有什么区别？**  
依次是权重存储、计算图、LLM管线容器、厂商编译网络。本例需要带完整元数据与辅助模型的`.litertlm`交给生成运行器。

**Q5：为什么不直接ONNX→NPU？**  
本项目已有LiteRT Torch和MediaTek AOT链，Qwen前端也现成；当前脚本没有ONNX importer。换一条格式链会增加算子、量化与状态适配工作，首轮没有收益。

### B. 量化

**Q6：A16W8里的A16是FP16吗？**  
这里表示静态整数16位激活，W8表示8位权重。INT16与FP16同占2字节，但需要不同的数值解释；量化tensor还需要scale和zero-point。

**Q7：PTQ和QAT的差别？本例是哪一种？**  
PTQ利用已有权重和校准统计确定量化参数；QAT在训练中模拟量化误差并更新权重。本例采用PTQ，不进行反向传播训练。

**Q8：第一阶段已经W8，第三阶段还在做什么？**  
在已有权重量化图上根据校准结果静态化激活、处理相应bias和图边界，并对齐跨图KV范围。当前实现不重新读取原始HF浮点权重。

**Q9：为什么不能只用随机token校准？**  
真实模板、语言、长度、任务和decode状态决定激活分布。随机数据可能给出与实际任务不同的范围，导致真实输入饱和或有效精度浪费。

**Q10：为什么校准prefill还不够？**  
decode的输入长度、历史长度和KV状态不断变化；aux也参与状态更新。prefill、decode及辅助图共同决定最终跨步数值合同。

**Q11：权重量化和KV量化能混为一谈吗？**  
权重是固定参数，KV是请求运行时产生的状态。W8不代表KV也是8位，模型文件变小也不代表上下文内存按同样比例减少。

**Q12：为什么必须对齐prefill/decode/aux的KV量化尺度？**  
同一个整数在不同scale/zero-point下代表不同实数。没有统一语义或显式requantize，字节复制正确也会把下一步状态变错。

### C. Prefill、decode与缓存

**Q13：prefill128和cache2048分别限制什么？**  
128是每块prefill图的输入长度；2048是历史cache容量。本例130token分两块处理，prompt与输出预算共同受会话容量约束。

**Q14：首token一定直接从prefill拿吗？**  
通用原理来自最后prompt位置的logits。这版split-cache prefill不输出logits，需用decode处理最后prompt token；PC采样器和Android runtime安排这一步的方式有差异。

**Q15：重算最后prompt token，会重复读到它吗？**  
旧历史mask只允许位置小于当前游标；当前token走新slice分支，旧槽同位置被屏蔽，随后覆盖写回。

**Q16：cache2048，decode mask为什么宽2049？**  
split-cache把2048个旧物理槽与1个当前新slice分开参与attention，mask同时覆盖两者；其中未使用的旧槽被屏蔽。

**Q17：GQA为什么节省KV？这款模型节省多少？**  
14个Q头共享2个KV头，每7个Q头一组。相同其他条件下，缓存主体为14-KV-head MHA的1/7。

**Q18：请口算这款模型2048长度的16位KV。**  
`2×24×2048×2×64×2=25,165,824字节=24 MiB`。这里没有包括权重、临时slice和工作区。

**Q19：padding token对应的KV是否完全不写？**  
本例full-attention更新会写整块，包含padding。真实长度和mask决定哪些槽能读，后续真实token覆盖未来槽。

**Q20：缓存保存什么？为什么不缓存Q？**  
后续token的attention需要历史K/V；每个新token使用自己的Q做查询，历史Q通常不再被后续位置使用，所以标准自回归KV缓存保存K/V。

**Q21：输出完最后一个token，它是否已经进入KV？**  
通常还没有。一次decode消费当前token并预测下一个；最后采样出的token需要下一次前向才形成它的KV。续跑时保留pending token和位置。

**Q22：只把cache_length改成4096就行吗？**  
它改变静态KV、mask等形状，需要对应导出、校准和编译产物；运行时设置也必须匹配。原模型32768位置上限不会自动变成本次2048产物的能力。

### D. 设备部署与排错

**Q23：MT6897出现在目标枚举里，能保证Qwen跑通吗？**  
能选择编译目标；还要满足算子/shape/dtype、SDK与ROM、runtime模型格式和内存条件。枚举没有提供该模型的完整执行证据。

**Q24：为什么关Gemma优化？**  
随包MediaTek默认配置启用Gemma专项选项，而本例是Qwen。先取消架构专属假设，再在正确基线上评估通用优化。

**Q25：NPU运行时CPU为什么仍忙？**  
分词、采样、会话、embedding/aux与同步仍需要主机参与。小模型decode的主干计算短，这些开销可能很显眼，必须看端到端数据。

**Q26：量化后第一步正常，几十步后胡说，查什么？**  
先查位置/有效长度、KV布局与跨图量化尺度、pending token、更新边界和长decode校准覆盖；再查局部数值敏感层。只比较一次prefill不够。

**Q27：NPU速度比CPU慢，优先怀疑什么？**  
查看关键矩阵是否实际卸载、分区是否碎片化、copy/sync与aux时间、固定cache计算浪费以及热降频。芯片TOPS不能替代这条任务的延迟分解。

**Q28：这版已经能承诺哪些东西？**  
源码中有Qwen前端、split-cache、量化、MT6897编译目标与官方NPU生成运行器参考。Qwen0.5B/MT6897这一组的编译成功、数值正确和性能，需要实际产物与对应ROM的运行记录。

### 一分钟陈述

> 我选Qwen2.5-0.5B-Instruct原始浮点权重，使用LiteRT Torch的HF前端和torch.export导出，经过MLIR/LiteRT转换生成prefill、decode和aux子图。配置batch1、prefill128、cache2048，先做W8权重量化，再用真实模板文本和最多32步decode校准，将主干激活静态量化为INT16，并统一跨图KV范围。随后关闭默认Gemma专项优化，以MT6897目标做MediaTek AOT，打包给Android LiteRT-LM运行。该模型24层、2个KV头、head_dim64，2048长度的16位KV主体是24 MiB。这版prefill没有logits，首个生成token要经decode取得；主图只返回新KV slice，历史状态由aux或运行器更新。最终我看实际分区和Run、正确性、TTFT与decode延迟，不能拿一个转换成功文件代替整条链路成功。
