# pc_convert → MT6897 NPU 技术报告

## 下载 NPU Convert 工具包

[Release 压缩包](https://github.com/bestkojima/pc-convert-mt6897-report/releases/tag/v2026.11.10) · [解压与安装说明](RELEASE_README.md)

附件包含转换脚本、Python 离线依赖和报告，适用 Linux x86_64 / CPython 3.12。NeuroPilot SDK 受独立再分发许可限制，由使用者按许可单独获取并补入；公开 ZIP 文件名中的 `sdk-external` 表示这一条件。

## 先读最新对照：DiT 官方支持与本地断口

[官方 LiteRT 与 pc_convert：具体适配、未接通环节、MT6897 生图路线与证据清单](DiT_官方LiteRT与pc_convert适配对比及MT6897落地报告.md)

2026-11-10 核对结果：

- 官方 LiteRT Torch v0.10.0 已有特定 FLUX.2-klein/DiT 适配；本包抽查的四个相关文件与官方相同。
- 本地主要封装导出、图规整、量化、MediaTek AOT 和诊断；真实轨迹校准、RoPE 消费、多输入运行器与完整生图循环尚未接通。
- MT6897 公开 target/支持表与 issue 中的支持答复存在未澄清冲突；需要先在目标 ROM/runtime 验证初始化和实际 NPU Run。
- 本次完成源码与资料核对，没有执行模型转换或 MT6897 生图。原 README 的 MT6991 小模型实验单独标记为作者历史记录。

证据附件：[四文件来源对照](evidence/LiteRT_Torch_0.10.0_四文件对照.md) · [直接比较记录](evidence/litert_torch_0_10_0_comparison.json) · [官方适配源码索引](evidence/官方DiT适配源码索引.md)

## 转换原理、Qwen 与语音扩展

[阅读全文：转换链、设备运行、Cache 设置与面试题](PC_CONVERT_MT6897技术报告.md)

[Qwen2.5-0.5B 实例：转换技术栈、逐步部署、量化、prefill/decode、KV 与面试问答](Qwen2.5-0.5B_MT6897逐步转换与面试问答.md)

[扩展专题：其他 LLM、ASR、TTS 能否转换，怎样选模型和拆分 NPU 子图](LLM_ASR_TTS扩展可行性分析.md)

面向需要讲清端侧模型部署原理、代码执行路径和实际限制的工程师。报告包含：

- PyTorch/权重到 TFLite、校准量化、MediaTek AOT 的逐阶段分析。
- MT6991 到 MT6897 的 target 选择、SDK 映射、运行库与设备兼容条件。
- LiteRT 与 Neuron 两条设备路径，以及真实 NPU 执行的证据层级。
- DiT 图像 token、LLM KV cache、编译缓存、工作区和 APU L1 的区别。
- Cache 配置 API、KV 内存计算、prefill/decode 与跨图量化参数对齐。
- 当前代码的具体缺陷、排错表、30 道面试问答和 4 道推导题。

版本依据为随包 LiteRT 2.3.0、LiteRT Torch 0.10.0、Quantizer 0.10.1。前三份报告初稿日期为 2026-11-06；DiT 对照及官方支持状态核对更新于 2026-11-10，来源固定到具体 commit。

Git 文档区发布分析报告、必要源码节选和比较记录；转换工具与 Python 依赖通过 Release ZIP 发布。模型权重和受限 NeuroPilot SDK 归档不上传。报告中的源码节选用于解释，版权与许可归原项目。

本次完成静态分析与资料研究，没有执行模型转换或 MT6897 实机推理。操作示例的适用条件和未完成的设备依赖在正文中列明。
