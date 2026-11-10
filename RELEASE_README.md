# NPU Convert 工具包：下载、补齐 SDK、开始使用

[下载 v2026.11.10 Release](https://github.com/bestkojima/pc-convert-mt6897-report/releases/tag/v2026.11.10)

下载附件 **`npu-convert-2026.11.10-sdk-external.zip`**。它包含 `pc_convert` 转换/量化/AOT 脚本、Python 离线依赖，以及 DiT、Qwen、Cache、ASR/TTS 报告。

**NeuroPilot SDK 需要单独获取。** 原包 SDK 的许可未授予独立再分发权，因此公开 ZIP 排除了 `pc_convert/offline/sdk/neuropilot-sdk-v8.tar.gz`。Python SDK wrapper 保留，它不含该 SDK 二进制。

## 1. 解压后的目录

```text
npu-convert-2026.11.10/
├─ RELEASE_README.md           # 先读这里
├─ package_manifest.json       # 文件清单、大小与排除项
├─ pc_convert/
│  ├─ convert_dit.py
│  ├─ install_offline.sh
│  ├─ README.md                # 原作者说明，含历史实验记录
│  └─ offline/
│     ├─ pip_pkgs/             # 原样保留的 wheel / wrapper 与内部许可证
│     ├─ SHA256SUMS            # 原包安装器使用的清单
│     └─ sdk/README.md         # SDK 补齐说明
└─ reports/                   # 本仓库报告与 evidence 附件
```

## 2. 按许可获取 SDK，放到安装器期待的位置

在 [MediaTek NeuroPilot](https://neuropilot.mediatek.com/) 获取适用于你的授权和用途的 SDK，阅读并接受相关许可。也可参考原包 `pc_convert/downloads.txt` 中的 SDK 下载来源；该地址来自官方 Python wrapper 的 `setup.py`，下载入口不替代许可条件。

本快照原安装器期待的归档路径是：

```text
pc_convert/offline/sdk/neuropilot-sdk-v8.tar.gz
```

原包这份归档大小为 **66,646,832 字节**，内部包含多个版本目录，安装器使用 `v8_0_10/host/lib` 等目录；不要仅凭文件名改名混用其他 SDK。原 `SHA256SUMS` 中仍保留该 SDK 的记录，供使用者补齐后由原安装器核对。

**未补齐 SDK 时，`install_offline.sh` 会明确报“缺 SDK tar.gz”。** 公开包不是开箱即用的完整离线 SDK 镜像。

## 3. 在匹配的 Linux 环境安装

本快照的 Python 依赖面向 **Linux x86_64、CPython 3.12**，包含 manylinux_2_28 wheel；使用匹配 glibc 的 Linux 发行版，例如 Ubuntu 22.04/24.04 或对应 WSL2 环境。Windows 原生 Python 无法安装这些 Linux wheel。

先补齐 SDK，再进入解压目录执行：

```bash
cd npu-convert-2026.11.10/pc_convert
bash install_offline.sh
source venv/bin/activate
python3 convert_dit.py --help
```

本次发布保留原脚本，没有修复报告中已列出的 AOT 返回码、量化回退和设备脚本问题。原安装脚本还包含一次 `pip` 在线升级尝试；需要严格断网部署时，先按报告说明处理。转换与设备运行步骤见报告，不应把安装成功当成模型或 NPU 验收。

## 4. 阅读顺序

1. `reports/README.md`：报告总入口。
2. `reports/DiT_官方LiteRT与pc_convert适配对比及MT6897落地报告.md`：官方适配、本地断口、MT6897 兼容风险与完整生图路线。
3. `reports/Qwen2.5-0.5B_MT6897逐步转换与面试问答.md`：量化、prefill/decode、KV cache。
4. `reports/PC_CONVERT_MT6897技术报告.md`：主流程与代码逐段分析。
5. `reports/LLM_ASR_TTS扩展可行性分析.md`：模型任务扩展。

## 5. 版本与发布状态

- 快照日期：2026-11-10。
- 随包版本：LiteRT 2.3.0、LiteRT Torch 0.10.0、AI Edge Quantizer 0.10.1、CPU PyTorch 2.14.1。
- 本次完成压缩、文件清单整理与公开发布，没有运行安装器、模型转换或 MT6897 实机推理。
- Python 依赖按原归档分发，保留其内部 LICENSE/NOTICE；各组件遵循各自许可。本发布没有为第三方代码重新授予统一许可证。
- 不包含模型权重、凭据、Git 元数据、本地分析缓存或受限 NeuroPilot SDK 归档。
