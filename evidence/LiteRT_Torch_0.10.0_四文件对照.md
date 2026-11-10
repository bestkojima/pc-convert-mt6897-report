# LiteRT Torch 0.10.0：DiT 相关源码的本地／上游对照

## 结论：四个指定文件与官方 v0.10.0 完全一致

本次直接读取本地 wheel 中的四个 Python 成员，再读取 Google AI Edge 官方仓库 `v0.10.0` 同路径源码：**四个文件的字节内容全部相同**，统一 diff 全为空。抽查覆盖转换入口、MLIR→FlatBuffer 桥接、扩展模型分发、FLUX.2-klein 分片导出。

报告应采用以下表述：

> 抽查的四个关键源码文件与官方 LiteRT Torch v0.10.0 完全相同；这些文件中未发现本地修改转换核心、改变接口或新增 FLUX.2-klein 实现的证据。官方同 tag 已包含 FLUX.2-klein 适配。这个结论覆盖所检查的文件，整个 wheel、底层编译器、MediaTek 插件和实际运行环境的来源仍未由本次检查确认。

因此，不宜写“这个离线包把官方没有的 DiT 功能补进 LiteRT 核心”，也不宜写“整套离线包是官方原样”。当前证据足以回答本轮疑问，无须为这段技术报告升级成完整供应链审计。

## 1. 比较对象和结果

- 本地归档：`D:\misc\pc_convert\pc_convert\offline\pip_pkgs\litert_torch-0.10.0-py3-none-any.whl`。
- 比较方法：`zipfile.ZipFile.read()` 读取指定成员；HTTPS 读取官方 tag 同路径；对两组原始 bytes 直接执行相等比较，并生成文本 unified diff。没有安装或执行归档内代码，没有计算哈希。
- 抓取完成：2026-11-10 15:11:15 UTC，即北京时间 23:11:15。
- 行号以直接读取的 UTF-8 源码 `splitlines()` 从 1 开始计数。网页阅读器可能合并显示行，不能拿它的显示行数代替文件物理行号。

以下成员均以 `litert_torch/` 为前缀。

| wheel 成员 | 本地／上游字节数 | 本地／上游行数 | 直接比较 |
|---|---:|---:|---|
| `_convert/core.py` | 11,676／11,676 | 310／310 | 完全相同 |
| `_convert/litert_converter.py` | 8,998／8,998 | 258／258 | 完全相同 |
| `generative/export_hf/model_ext/exportables.py` | 9,158／9,158 | 224／224 | 完全相同 |
| `generative/export_hf/model_ext/flux2_klein/flux2_klein.py` | 11,929／11,929 | 315／315 | 完全相同 |

文件长度仅用于读者核对；结论来自实际内容比较。

上游对应文件：[转换入口](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/_convert/core.py)、[转换桥接](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/_convert/litert_converter.py)、[模型分发](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/exportables.py)、[FLUX.2-klein 导出](https://github.com/google-ai-edge/litert-torch/blob/90cbbaa61ab3f029f1273b108909a5b4d9cc2c12/litert_torch/generative/export_hf/model_ext/flux2_klein/flux2_klein.py)。结论对应本次抓取时该 tag 返回的内容；该 tag 的 commit 为 `90cbbaa61ab3f029f1273b108909a5b4d9cc2c12`。

## 2. 转换核心：入口和调用链相同

以下短摘录均取自本地 wheel，已在上述文件级对比中确认与上游一致；完整源码快照保留原 Apache-2.0 版权／许可头。

### 2.1 `core.py` 仍走标准导出、FX Pass 和转换桥接

成员 `litert_torch/_convert/core.py:L269–279`：逐签名导出 PyTorch 程序，再执行通用转换 passes。

```python
  exported_programs = []
  for sig in signatures:
    with progress.task(f"Torch Export: {sig.name}"):
      exported_program = export_torch_signature(
          sig, strict_export=strict_export
      )
    exported_programs.append(exported_program)

  # Apply default fx passes
  with progress.task("Run FX Passes"):
    exported_programs = list(map(_run_convert_passes, exported_programs))
```

同成员 `L281–290`：调用通用 FlatBuffer 转换，再包装成 `LiteRTModel`。

```python
  exporter = litert_converter.exported_programs_to_flatbuffer(
      exported_programs,
      signatures,
      enable_x64=enable_x64,
      quant_config=quant_config,
      lightweight_conversion=lightweight_conversion,
      runtime_constant_folding=runtime_constant_folding,
  )

  return model.LiteRTModel(exporter)
```

`use_v2` 分支也包含在整文件相等结果中。本次没有发现这个入口文件为 DiT 改用本地专属转换流水线的改动。

### 2.2 `litert_converter.py` 负责桥接，实际编译 pass 来自依赖

成员 `litert_torch/_convert/litert_converter.py:L167–172`：PyTorch 导出程序降到 MLIR。

```python
    with progress.task(f"Lower to MLIR: {sig.name}"):
      lowered = backend.export.exported_program_to_mlir(
          exported_program,
          ir_context=ir_context,
          lowering_context_plugins=[cross_program_inline_consts_ctx],
      )
```

同成员 `L224–229`：调用 LiteRT 转换 pass。

```python
  # Run LiteRT converter passes.
  with ir_context, progress.task("Run LiteRT Converter Passes"):
    pass_manager = passmanager.PassManager()
    converter_api_ext.run_convert_to_tfl_passes(
        merged_module, pass_manager, config
    )
```

这个 Python 桥接文件相同，可以确认该层没有差异。实际被调用的扩展来自另一个依赖；同成员 `L31`：

```python
from litert_converter.mlir._mlir_libs import converter_api_ext
```

因此，Python 桥接相同无法证明 `litert_converter` 的本机扩展二进制相同，更不能据此证明 MediaTek 编译器相同。本轮没有检查这些依赖的二进制内容。

### 2.3 AOT 也另有调用边界

成员 `litert_torch/_convert/core.py:L306–310`：AOT 调用下沉到 `ai_edge_litert`。

```python
  litert_model = litert_types.Model.create_from_bytes(cpu_model.model_content())
  return aot_compile_lib.aot_compile(
      litert_model,
      config=compilation_configs,
  )
```

MT6897 目标编译与设备运行能力需沿该依赖、vendor plugin、SDK 和设备 runtime 继续分析。四个 Torch 源文件一致不能替代那些层的兼容性证据。

## 3. FLUX.2-klein：官方同 tag 已有模型适配

### 3.1 官方模型分发同时识别内部名称和 Pipeline 名称

成员 `litert_torch/generative/export_hf/model_ext/exportables.py:L217–224`：

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

这一项直接推翻“看见本地有 `flux2_klein` 就认定是作者自行增加的架构支持”的推断：本次读取的官方 v0.10.0 分发文件已有完全相同的分支。

### 3.2 官方适配层有分片设计，转换时仍调用公共 API

成员 `litert_torch/generative/export_hf/model_ext/flux2_klein/flux2_klein.py:L55–61`：

```python
class Flux2Klein(bonsai_flux2.BonsaiFlux2):
  """FLUX.2-klein multi-shard text-to-image model."""

  _TEXT_EMBED_FILENAME = "qwen_embed_fp16.bin"
  _NUM_TE_SHARDS = 3
  _NUM_DIT_DOUBLE_SHARDS = 2
  _NUM_DIT_SINGLE_SHARDS = 4
```

该设计将文本编码器分为 3 片；DiT 再加 initial、final，组成 8 片。这里属于**模型适配与导出编排层**，它存在于官方同 tag。

同成员 `L189–193`：分片依旧交给 `litert_torch.convert()`，随后按设置调用 DiT 量化函数。

```python
      fp32_path = os.path.join(output_dir, f"{stem}_fp32.tflite")
      litert_torch.convert(mod.eval(), inputs).export(fp32_path)
      if quantize and self._should_quantize(export_config):
        int4_path = os.path.join(output_dir, f"{stem}_int4b32.tflite")
        bonsai_flux2.quantize_dit(fp32_path, int4_path)
```

可以确认这个适配文件及其公共 API 调用与上游相同；其引用的 `modeling.py`、`bonsai_flux2.py` 等未纳入本轮四文件对比。该文件具有导出实现，也不自动构成 MT6897 的编译成功或 NPU 运行证据。

## 4. 三个层次，面试时别混在一句话里

1. **模型改写／适配**：模块包装、forward 改写、固定输入形状、拆分子图、外置 embedding、修改量化方案。即使调用官方原样转换库，项目也完全可能在这一层有自己的工程代码。
2. **转换库改动**：修改 Torch 导出、FX Pass、算子 lowering、MLIR pass 配置或 FlatBuffer 导出代码。本轮抽查的入口和桥接文件与上游相同，没有在这些文件中发现此类改动。
3. **底层编译器／设备后端改动**：`litert_converter` 二进制、`ai_edge_litert`、MediaTek 插件／SDK、Android runtime。本轮未比较这些对象。

因此，“项目为模型做了适配”“LiteRT Torch 某些文件保持官方内容”“整个工具链具有官方来源”是三项不同事实，应各用自己的证据回答。版本号是定位比较对象的线索，直接内容比较才回答本次指定文件有没有不同。

## 5. 比较记录与本地原始快照

- 比较记录：`D:\misc\pc_convert\analysis\dit-provenance\comparison.json`。
- 本地四文件快照目录：`D:\misc\pc_convert\analysis\dit-provenance\local`，保持成员相对路径。
- 上游四文件快照目录：`D:\misc\pc_convert\analysis\dit-provenance\upstream-v0.10.0`，保持成员相对路径。
- 四份 `.diff` 位于 `D:\misc\pc_convert\analysis\dit-provenance` 根目录；它们为空，对应完全相同的比较结果。

本次完成项是源码来源的有限对照；没有运行模型、编译器或设备端程序。

公开仓库中的直接比较记录见 [comparison.json](litert_torch_0_10_0_comparison.json)。上面的完整源码快照与空 diff 保留于本地分析目录，不随报告上传。
