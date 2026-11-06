# pc_convert → MT6897 NPU 技术报告

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

版本依据为随包 LiteRT 2.3.0、LiteRT Torch 0.10.0、Quantizer 0.10.1；官方在线资料核对日期为 2026-11-06。

本仓库仅发布分析报告。原始工具包、模型权重和 MediaTek 二进制 SDK 不随报告上传。报告中的源码节选用于解释，版权与许可归原项目。

本次完成静态分析与资料研究，没有执行模型转换或 MT6897 实机推理。操作示例的适用条件和未完成的设备依赖在正文中列明。
