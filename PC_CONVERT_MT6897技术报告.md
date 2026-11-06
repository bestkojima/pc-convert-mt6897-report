# 从 pc_convert 到 MT6897 NPU：转换链、运行机制与 Cache 设计

**面试与工程阅读版 · 2026-11-06**

分析对象：`D:\misc\pc_convert\pc_convert`。源码与离线依赖共同作为版本依据：LiteRT **2.3.0**、LiteRT Torch **0.10.0**、AI Edge Quantizer **0.10.1**、PyTorch **2.14.1+cpu**、MediaTek SDK wrapper **2.3.0**。本文完成源码分析与官方资料核对；没有安装工具链、转换实际模型或连接 MT6897 手机。

## 先把结论讲清楚

1. **这套程序是以 DiT 为中心的转换与诊断工具包。** 它提供模型加载、固定输入导出、TFLite 图规整、量化、MediaTek AOT 编译、部署/诊断辅助。完整文生图 App 还需要文本编码、采样调度、VAE、资源管理等。普通 LLM 的 tokenizer、prefill/decode、KV 更新也没有在主脚本里实现。
2. **MT6897 的正确编译入口是 `--soc MT6897`，随后重新 AOT。** 随包 LiteRT 中存在 `SocModel.MT6897`，推荐 SDK 大版本为 `v8`。把 MT6991 产物改名、把 manifest 的 `soc` 改掉，都不能完成芯片适配。
3. **推荐路线：x86_64 Linux 上编译，Android arm64 上用 LiteRT CompiledModel + MediaTek Dispatch 加载完整 AOT `.tflite`。** 保留编译后的图、分区信息和嵌入字节码；不要把现有“抽一段 `.dla` 跑小 MLP”的探针升级解释成通用推理引擎。
4. **`cache size` 必须先指明对象。** 当前 `convert_dit.py` 没有 KV cache 配置。`--seq-len` 控制图像 patch token 数；LLM 的 cache 容量应在另一条生成模型导出/运行链配置。
5. **现有程序的返回码与成功日志偏宽松。** AOT 异常可能被吞掉后仍退出 0；静态量化失败会转动态量化；部署脚本还有 `adb` 同名函数递归。学它的分层思路很有价值，直接拿“一键成功”作为交付标准会栽跟头。

### 怎么读这份报告

- **理解转换原理**：第 1～4 章。
- **准备 MT6897 部署**：第 5～7 章。
- **回答 cache、内存和性能追问**：第 8～10 章。
- **准备面试陈述和排错题**：第 11～12 章。
- **换成其他 LLM、ASR 或 TTS**：[扩展可行性分析](LLM_ASR_TTS扩展可行性分析.md)，包含随包模型适配、语音子图、明确断口与推荐起点。

正文中的源码摘录均注明文件与行号，紧邻分析；省略号表示非连续节选。`WHEEL:` 表示离线 wheel 内部路径，行号属于该成员文件。报告中的操作示例是部署方案，不能当成本次实机执行记录。

---

## 1. 一张图看懂整个系统

```text
训练/下载得到的模型
  ├─ 网络定义、forward、config
  ├─ 参数/常量：safetensors / state_dict
  └─ 输入预处理、shape、layout、dtype、输出语义
           │ PC：构建同结构 nn.Module，严格匹配权重
           ▼
       PyTorch eval 模型
           │ litert_torch.convert（固定样例输入）
           ▼
       浮点 .tflite 图
           │ builtin opcode 回填 / RESHAPE 规整
           ▼
       规整后的浮点图
           │ 代表性样本校准 → scale/zero point → 权重/激活量化
           ▼
       量化 .tflite
           │ target=MediaTek_MT6897；插件划分可支持子图
           │ NeuroPilot host compiler → 目标机器码与内存计划
           ▼
       AOT .tflite
       ├─ 未卸载的 LiteRT 算子（若存在）
       ├─ CUSTOM DISPATCH_OP / 分区元信息
       └─ 嵌入的 MediaTek 编译字节码
           │ 分发：模型 + Android arm64 对应版本运行库
           ▼
       Android App / native runner
           │ LiteRT Environment → Model → CompiledModel
           │ 创建/绑定 I/O buffer → Run → 等待完成 → 取回结果
           ▼
       MediaTek Dispatch → Neuron adapter/runtime → 驱动 → APU/MDLA
```

**PC 编译器的 CPU 架构与目标 NPU 架构是两回事。** PC 上的 x86_64 程序负责生成 MT6897 的代码；手机上的 arm64 程序负责提交和管理执行。就像交叉编译 Android App，编译机不必长着手机那颗芯片。

### 1.1 六种文件各自装的是什么

| 文件/对象 | 内容与职责 | 常见误解 |
|---|---|---|
| `.safetensors` / state_dict | 参数名称、形状、数值 | 单靠权重无法重建任意 `forward` |
| 浮点 `.tflite` | 图、算子、张量、常量、签名 | 扩展名不表明使用哪个硬件 |
| 量化 `.tflite` | 更低精度的权重/激活及量化参数 | INT8 出现了也不保证全部算子进 NPU |
| AOT `.tflite` | LiteRT 包装图 + 编译分区 + 字节码 | 不能用普通 CPU Interpreter 代替 Dispatch 执行厂商自定义分区 |
| `.dla` | Neuron 编译网络的二进制产物 | 通常需匹配 SoC、编译器与 runtime，不能跨芯片随意复用 |
| `.npug` | README 描述的第三方私有图容器 | 本项目没有通用 NPUG 导出器，也没有任意权重到该 App 的转换器 |

`compile_plan.json` 是元数据和操作提示；它既不编译网络，也不执行 NPU。`rope_*.bin` 是位置编码数据，主脚本生成它之后没有自动插入模型图。

### 1.2 AOT、JIT 与图导出的区别

- **导出**：把模型程序变成部署 IR/图，解决框架表示问题。
- **量化**：改变数值表示与计算精度，解决内存、带宽、算子适配问题。
- **AOT**：部署前针对目标 SoC 编译，降低端上首次启动的编译开销。
- **JIT / on-device compilation**：设备初始化时编译，可按设备环境生成产物，首次启动耗时和峰值内存更高。
- **执行**：加载编译产物、提供输入、提交设备任务、等待结果。上面三步成功都不能替代这一步。

大型、目标固定的模型优先 AOT。这与官方的 NPU 部署建议一致。[LiteRT NPU 文档](https://ai.google.dev/edge/litert/next/npu)

---

## 2. 程序地图：每个文件负责到哪一层

| 文件（位于 `D:\misc\pc_convert\pc_convert`） | 角色 | 阅读时盯住的接口 |
|---|---|---|
| `convert_dit.py` | 转换主流程 | source、输入 shape、recipe、soc、输出目录 |
| `install_offline.sh` | 本地依赖安装 | CPython 3.12、Linux x86_64、wheel 标签、SDK 目录 |
| `setup_npu_toolchain.sh` | 联网安装 | 版本约束、CPU torch、SDK 下载、插件路径 |
| `install_on_device.sh` | adb 推送和兼容提示 | ADB 选择、设备属性、runtimeMajor、目标目录 |
| `litert_stage.sh` | 组装 LiteRT 诊断环境 | 来源库、ABI、loader、动态依赖 |
| `litert_npu_check.cpp` | CompiledModel 创建与状态探针 | accelerator mask、Dispatch 路径、fully accelerated |
| `verify_on_apu.sh` | Neuron 小模型验证的编排 | 提取字节码、构建 runner、基准输入/输出 |
| `aot_on_apu.cpp` | 直接 Neuron restore/compute | `.dla`、I/O binding、同步执行、资源释放 |
| `apu_baseline.py` | 小模型参考输出与数值比较 | 支持的算子和量化语义 |
| `acc_check.py` | 浮点/量化 TFLite CPU 对照 | 输入量化、输出反量化、误差统计 |
| `fetch_deps.sh` | 从原环境收集依赖 | 强环境耦合，不能当通用 Android SDK 安装器 |
| `README.md` | 历史实验叙述与用法 | 区分旧结论、后续更正、原始证据是否随包 |

### 2.1 环境选择：用本包就固定到 CPython 3.12

`install_offline.sh:25–33` 的实际选择逻辑：

```bash
PY=""
for c in python3.12 python3 python; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)
    [ "$v" = "3.12" ] && { PY="$c"; break; }
  fi
done
[ -n "$PY" ] || die "找不到 CPython 3.12（本包 wheel 全是 cp312 标签）。
   装：sudo apt install python3.12 ；或重做一份对应你 Python 版本的离线包。"
```

包中同时含纯 Python、abi3 与 cp312 wheel；约束来自其中 cp312 的二进制依赖。项目某些文件写“Python 3.10～3.14”，不能用这句话放宽当前离线包的安装范围。

**推荐工作站：x86_64 Ubuntu 24.04 / CPython 3.12；Windows 用 WSL2 的 Linux 环境。** Windows Python 无法直接加载这些 Linux `.so` 和 manylinux wheel。`torch+cpu` 足以完成这条导出链，无需为了转换额外装一整套 CUDA。

依赖关系以本地 wheel 的 `METADATA` 为准：

```text
litert-torch 0.10.0:
  torch >=2.4.0,<2.15.0
  ai-edge-litert[model-utils] >=2.3.0,<2.4.0
  ai-edge-quantizer ==0.10.*
  litert-converter ==0.5.*
  torchao >=0.17.0

ai-edge-quantizer 0.10.1:
  ai-edge-litert ==2.3.0
```

还有一处文档与实现冲突：`install_offline.sh:85` 会执行下列联网升级尝试，所以“全程绝不联网”的表述不符合代码。离线部署应去掉这一行或改成本地来源；本文没有修改安装脚本。

```bash
pip install -q --upgrade pip 2>/dev/null   # 离线环境这条多半失败，无害
```

---

## 3. 按执行顺序拆解 `convert_dit.py`

### 3.1 参数确定模型合同

`convert_dit.py:1176–1190`：

```python
ap.add_argument("source", help=".safetensors / .pt / .bin 权重，或已导出的 .tflite")
ap.add_argument("--out", default="./npu_out")
ap.add_argument("--soc", default=SOC_DEFAULT)
ap.add_argument("--tag", default="dit", help="产物前缀，学玑绘的 dva1/ditc1c 命名")
ap.add_argument("--recipe", default="w8a16", choices=list(RECIPES),
                help="w8a16=玑绘主力档(静态W8A16) w8a8 w4a16(自造recipe) "
                     "w8dyn/w4dyn(激活fp32，APU多半不认，仅作对照基线)")
ap.add_argument("--height", type=int, default=512, help="像素高（latent=像素/8）")
ap.add_argument("--width", type=int, default=512, help="像素宽")
ap.add_argument("--seq-len", type=int, default=None, help="留空按 (像素/16) 行列乘积推")
ap.add_argument("--latent-ch", type=int, default=4)
ap.add_argument("--hidden-dim", type=int, default=2048)
ap.add_argument("--text-len", type=int, default=77)
ap.add_argument("--text-dim", type=int, default=2048)
ap.add_argument("--head-dim", type=int, default=256, help="逆向得玑绘 head_dim=256")
```

这些默认值构成一套特定 DiT 输入假设。真实模型可能吃 `[B,C,H,W]` latent，也可能吃尚未 hidden projection 的 patch；文本编码长度、维度、时间步范围也可能不同。应先写与原模型等价的 wrapper，再选参数。让参数“看起来像某模型”无法替代 `forward` 对齐。

默认 token 数：

```text
latent_h = H / 8，latent_w = W / 8
2×2 latent patch → grid_h = H / 16，grid_w = W / 16
L = grid_h × grid_w

512×512   → 32×32 → 1024 tokens
768×768   → 48×48 → 2304 tokens
768×1024  → 48×64 → 3072 tokens
896×1152  → 56×72 → 4032 tokens
```

这里的“token”是图像 patch。改变它意味着改变 attention 的工作量和编译形状。

### 3.2 `preflight`：把依赖错误拦在前面

它检查 OS/架构、LiteRT、PyTorch、SDK、编译插件和尺寸整除关系。导出与量化可以独立于 AOT；`--skip-aot` 只得到量化图，仍需后续设备编译/运行实现。

一个容易误读的分支是 `convert_dit.py:1250–1257`：

```python
if aot_only:
    log("")
    if not (args.skip_aot or args.force):
        log("AOT 段跑不了，但导出与量化仍可继续。加 --skip-aot 显式确认，否则这里直接停：")
        for i, p in enumerate(aot_only, 1):
            log(f"  [{i}] {p}")
        return 2
    log("AOT 段相关组件不可用，你已确认跳过 —— 只产量化 .tflite，继续：")
```

`--force` 只放过前置障碍，并没有设置 `args.skip_aot=True`；后面仍可能调用 AOT。不要把它理解为自动修环境或保证跳过。

### 3.3 RoPE 表：生成常量，尚未接进图

`convert_dit.py:217–229`：

```python
half = head_dim // 2
inv = 1.0 / (10000.0 ** (np.arange(0, half, 2, dtype=np.float64) / half))
ys, xs = np.mgrid[0:grid_h, 0:grid_w]
pos_y = ys.reshape(-1).astype(np.float64)
pos_x = xs.reshape(-1).astype(np.float64)
cos = np.concatenate([np.cos(np.outer(pos_y, inv)),
                      np.cos(np.outer(pos_x, inv))], axis=1)[:, :half]
sin = np.concatenate([np.sin(np.outer(pos_y, inv)),
                      np.sin(np.outer(pos_x, inv))], axis=1)[:, :half]
tbl = np.concatenate([cos, sin], axis=1).astype(np.float16)
assert tbl.shape == (L, head_dim), tbl.shape
name = f"rope_{grid_h}x{grid_w}.bin"
(out_dir / name).write_bytes(tbl.tobytes())
```

机制：对固定二维位置提前计算 cos/sin，导出 FP16 表，避免推理时重复构造位置函数。大小为 `L × head_dim × 2 bytes`：默认 1024×256×2 = **512 KiB**；4032×256×2 = **1.96875 MiB**。

三个面试要点：

1. RoPE 的频率基数、二维轴分配、维度排列、旋转配对方式必须与训练时一致。
2. **文件大小一致只说明元素数量一致**，不能证明位置编码数值和布局正确。
3. 代码只把表加入产物列表；`make_inputs()` 和 `to_tflite()` 没有读取它。需让 wrapper 注册常量 buffer 并真正使用，或由运行器按正确接口输入。仅生成 `.bin` 不会自动把图中的 sin/cos 消掉。

注释中“APU 无三角函数指令”不能扩展成所有 MediaTek 芯片与所有编译器的能力结论。合理的工程表述是：**为这个固定形状模型预计算位置编码，可降低运行时算子适配与计算成本。**

### 3.4 加载：权重文件之外，还必须有模型结构

`convert_dit.py:258–267`：

```python
spec = importlib.util.spec_from_file_location("udef", args.repo_or_script)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
if not hasattr(mod, "build"):
    raise SystemExit(f"{args.repo_or_script} 需提供 build() -> nn.Module")
model = mod.build()

sd = _read_state_dict(src)
_inject(model, sd, args.allow_partial)
model.eval()
```

本包没有目标业务模型的 `model_def.py` 或实际权重。这里的 `build()` 应构造与 checkpoint 一致的网络。另有 `--model-class` 和 `--from-hub` 分支；后者先取 Diffusers pipeline 的 `.transformer`，之后仍会读 `source` 并注入权重，不是只传 Hub ID 就自动完成任意模型适配。

权重匹配检查见 `convert_dit.py:291–305` 节选：

```python
named = dict(model.named_parameters()) | dict(model.named_buffers())
hit = {k: v for k, v in sd.items() if k in named}
extra, miss = set(sd) - set(hit), set(named) - set(hit)
# ...
if miss:
    msg = f"  模型缺权重 {len(miss)}（前 5）：{sorted(miss)[:5]}"
    if allow_partial:
        log(msg + "  ← --allow-partial 已放行；这些层保持随机初始化，"
                  "只够打通链路，出图一定是错的")
    else:
        raise SystemExit(msg + "\n  用 --allow-partial 强行继续，或修 model class 的名称映射")
model.load_state_dict(hit, strict=False)
```

`strict=False` 仍会在匹配键 shape 不兼容时失败；它不会自动转置权重或修层结构。缺失参数可导致随机初始化，缺失 buffer 也可能只是默认常量，两者要分别确认。正式转换不启用 `--allow-partial`。

还应确认：模型在 CPU、参数为预期浮点 dtype、dropout 已关闭、输出包装和训练时一致。`eval()` 只改变训练/推理行为，不负责把 GPU 权重搬回 CPU，也不强制转换 FP32。动态导入和 `exec()` 会执行模型定义文件，因此只加载可信模型代码。

### 3.5 固定输入与导出

`convert_dit.py:318–324` 与 `342–346`：

```python
L = args.seq_len
lh, lw = args.height // 8, args.width // 8
hs = torch.randn(1, L, args.hidden_dim, dtype=torch.float32)
tb = torch.randn(1, args.text_len, args.text_dim, dtype=torch.float32)
t = torch.full((1,), 500.0, dtype=torch.float32)
return ({"hidden_states": hs, "encoder_hidden_states": tb, "timestep": t},
        (args.latent_ch, lh, lw, L))
```

```python
sample_args = tuple(inputs.values())
log(f"  litert_torch {getattr(litert_torch, '__version__', '?')}")
try:
    conv_result = litert_torch.convert(model, sample_args, strict_export=False,
                                       output_file_path=str(out_path))
```

传给转换器的是**位置参数 tuple**，所以 wrapper 的参数顺序必须为 hidden states、text states、timestep。字典里的名字不会让这个调用自动按关键字绑定。导出后签名名可能成为 `args_0` 等，后续校准需要按实际签名映射。

`strict_export=False` 不会赋予 NPU 动态形状能力，也不能让不支持的 Python 控制流或算子凭空可导出。典型转换过程会捕获 PyTorch 图、分解高层算子、降到 LiteRT 支持的表达；具体 lower 路径属于转换器版本行为。

代码兼容旧导出分支不自动落盘的情况，`convert_dit.py:351–355`：

```python
if not out_path.is_file():
    content = getattr(conv_result, "model_content", None)
    if callable(content):
        out_path.write_bytes(content())
        log(f"  （convert() 未落盘，已从返回的 LiteRTModel 写出）")
```

应为每次转换使用独立输出目录。当前实现若同名旧文件已经存在，会跳过这一段回写，存在把旧模型误当新导出结果的风险。

### 3.6 图规整：修的是表示兼容性

`convert_dit.py:429–433`：

```python
n_backfill = 0
for oc in m.operatorCodes:
    if not getattr(oc, "builtinCode", 0) and getattr(oc, "deprecatedBuiltinCode", 0):
        oc.builtinCode = oc.deprecatedBuiltinCode
        n_backfill += 1
```

旧 TFLite 可能把算子编号保存在兼容字段。若下游只读 `builtinCode`，默认值 0 会被误认为 ADD。回填让量化器正确识别真实算子。

另一项规整把单输入 RESHAPE 的静态形状 option 物化成第二个 INT32 常量输入。它保持 reshape 语义，同时适配要求显式 shape operand 的后端。代码只对第 0 个 subgraph 做该 RESHAPE 修补；不能宣称已经规整了多签名 LLM 的全部子图。

“等价检查通过”的含义也要缩小。`convert_dit.py:484–503` 节选：

```python
try:
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    return True
# ... 每个模型设置样例输入并 invoke ...
outs.append(np.asarray(it.get_tensor(
    it.get_output_details()[0]["index"]), dtype=np.float64))
return outs[0].shape == outs[1].shape and np.array_equal(outs[0], outs[1])
```

当前只比较一次输入下的第一个输出；导入失败直接返回 True。对多输出、有状态或整数输入模型，这不足以建立完整等价性。报告把它称为**有限样例的规整自检**。

---

## 4. 量化：真正困难的是数值语义

### 4.1 W8A16 究竟表示什么

- `W8A16`：目标是 8-bit 权重、16-bit 激活。
- `W8A8`：8-bit 权重、8-bit 激活；通常更省激活带宽，但精度压力更大。
- `W4A16`：4-bit 权重、16-bit 激活；小文件不等于后端一定有高效 kernel。
- `W8dyn/W4dyn`：脚本使用权重量化、浮点激活的动态配方。应作为 CPU 对照候选，不能作为本项目 NPU 静态量化失败后的等价替代。

**推荐顺序：先 W8A16 建立保真基线，再评估 W8A8 的收益，最后尝试 W4A16。** 一上来追最低位宽，误差与算子兼容问题会搅成一锅粥。

实际图可能混合精度，bias/累加器也不遵循激活位宽。典型整数卷积/全连接会用更宽 bias 与 accumulator 来容纳点积累加。

### 4.2 量化与反量化公式

```text
q = clip(round(x / scale) + zero_point, q_min, q_max)
x_hat = scale × (q - zero_point)

INT8 范围：[-128, 127]
对称量化常用 zero_point=0
per-tensor：整张量一个 scale
per-channel：按输出通道分别设置 scale
```

直接 `astype(int8)` 只会截断/转换数值，不会完成上述映射。送入模型的量化 I/O 应使用它自身的 scale、zero point；读输出时对应反量化。

对点积，近似有 `scale_acc = scale_x × scale_w[channel]`；bias 与 accumulator 必须处在兼容尺度上。之后 requantize 到输出 scale。看见 INT64 bias 很正常，不能擅自压回 INT8。

### 4.3 校准数据要覆盖完整输入分布

DiT 校准至少应包含：不同 prompt 的真实文本条件、不同噪声水平与时间步、对应 latent/hidden states、真实分辨率与形状。如果模型使用 CFG，正/负条件路径和实际运行方式也要覆盖。训练权重、scheduler、timestep 约定须保持一致。

当前 `--calib` 的名字容易让人高估它。`convert_dit.py:799–808`：

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

它只读取真实 hidden states，文本仍然随机，时间步固定 500，并硬编码三个输入名。**完整真实轨迹校准尚未实现。** 建议用每条样本保存全部输入的 NPZ/字典，并按导出后的 signature 名称逐一映射；这属于后续代码改造，不能在当前脚本上只换个文件后缀就声称支持。

随机校准分支根据 shape `[1]` / `[1,1]` 猜时间步，并给 0～1000 均匀值。这个启发式可能误判其他标量控制量；有些扩散模型的时间参数本来就归一化到 0～1。模型自身合同优先于这个猜法。

### 4.4 静态失败后降级，会改变交付合同

`convert_dit.py:896–910` 节选：

```python
res = qt.calibrate(calib_data,
                   mode=calibrator.CalibrationMode.CALIBRATION_PROFILER_BASED)
qm = qt.quantize(calibration_result=res)
# ... except Exception as e ...
qt = quantizer.Quantizer(str(model_tflite))
qt.load_quantization_recipe(rc.dynamic_wi8_afp32())
qm = qt.quantize()
out = out_dir / f"dit_{args.recipe}_fallback_w8dyn.tflite"
```

原要求是静态 W8A16，降级文件变为动态 W8。虽然文件名有 `fallback_w8dyn`，后续部分检查和 plan 仍使用 `args.recipe`，导致“请求配置”和“实际产物”不一致。

正式转换应在静态校准失败处终止并修正样本/签名；若明确选择动态基线，应单独记录 effective recipe，并停止把它当 NPU 候选成功交付。

### 4.5 怎样证明量化有效

程序检查量化参数张量数量、整型张量数量、部分 bias dtype、scale 有限且大于零。这些是结构检查。还需理解它们的范围：

- `_quant_stats()` 把 shape/index 等整型张量也计入，日志“整型权重”不够准确。
- 有 quantization 字段不代表该张量有完整有效的 scale；增加计数不代表所有目标权重已经量化。
- 量化后 CPU 装载失败可能是特定 kernel 的 dtype 组合不支持，也可能是真正的图错误；不能看见 INT16 就一概忽略错误。
- CPU 冒烟只设置第一个输入；多输入 DiT 的完整数值语义没有得到覆盖。

正确的比较链是：

```text
PyTorch FP32 ─比较─ 导出 FP32 LiteRT
                      └─比较─ 量化语义参考
                                   └─比较─ 同量化模型 NPU 输出
```

前两段定位导出误差与量化误差，最后一段定位后端执行误差。DiT 还要看整条去噪轨迹与最终图像；LLM 则看 logits、困惑度/任务准确率与生成结果。余弦相似度一个数撑不起完整质量结论。

---

## 5. 把目标从 MT6991 改到 MT6897

### 5.1 编译 target 的真实映射

随包 `ai_edge_litert-2.3.0-…whl` 内，`WHEEL:ai_edge_litert/aot/vendors/mediatek/target.py:34–40,48–54`：

```python
MT6893 = "MT6893"
MT6895 = "MT6895"
MT6897 = "MT6897"
MT6983 = "MT6983"
MT6985 = "MT6985"
MT6989 = "MT6989"
MT6991 = "MT6991"
```

```python
def map_soc_to_np_version(soc_model: SocModel) -> str:
  """Returns the NP version for the given SOC model."""
  match soc_model:
    case SocModel.MT6993:
      return "v9"
    case _:
      return "v8"
```

`MT6897 → v8 → v8_0_10/host/lib` 是本包的编译 SDK 选择链。`v8` 是 SDK 大版本选择，不表示两颗芯片生成的代码一样，也不保证任意手机 ROM 的 runtime 接受该字节码。

这里还存在 Android target 默认值，`target.py:75–77`：

```python
soc_model: SocModel
soc_manufacturer: SocManufacturer = SocManufacturer.MEDIATEK
android_os_version: AndroidOsVersion = AndroidOsVersion.ANDROID_15
```

它是这个 Python target 的默认字段，不能直接推断“所有 Android 15 设备都支持”或“低于 Android 15 必然不能运行”。设备支持应核对具体运行库、系统兼容性及官方发行说明。

### 5.2 `--soc` 确实传到了厂商编译器

`convert_dit.py:1075–1085`：

```python
soc = getattr(T.SocModel, args.soc, None)
if soc is None:
    raise SystemExit(f"{args.soc} 不在 MTK 支持枚举里。可选："
                     f"{[s.value for s in T.SocModel if s.value.startswith('MT')]}")
tgt = T.Target(soc)
log(f"  target={tgt!r}  np={tgt.recommended_np_version}  {tgt.flatten()}")

try:
    res = aot_compile(str(tflite), output_dir=str(out_dir),
                      config=[aot_types.CompilationConfig(target=tgt)],
                      keep_going=False)
```

编译 backend 会取 SDK 搜索路径，并把 SoC 转为小写传入插件。`WHEEL:ai_edge_litert/aot/vendors/mediatek/mediatek_backend.py:160–175` 节选：

```python
sdk_libs_path = str(
    ai_edge_litert_sdk_mediatek.path_to_sdk_libs(recommended_np_version)
)
# ...
extra_kwargs = {"libs": lib_dir, "sdk_libs_path": sdk_libs_path}
if recommended_np_version == "v9":
  extra_kwargs["mediatek_sdk_version_type"] = "version9"
# ...
return component(
    input_model,
    output_model,
    backend.soc_manufacturer,
    backend.soc_model.lower(),
    **extra_kwargs,
)
```

所以 `--soc MT6897` 会选择 `mt6897` 编译目标。改 target 后应该使用新输出目录重新编译，不能复用 MT6991 的编译结果。

### 5.3 图分区为什么决定加速效果

插件检查算子、dtype、shape、常量性等约束，把可支持子图编译成 NPU 分区，再把它们替换成 Dispatch 节点。剩余计算可能留给其他后端。

例如：

```text
原图：MatMul → Add → LayerNorm → MatMul → 自定义算子 → MatMul
分区：└──────── NPU 分区 0 ────────────┘     CPU      NPU 分区 1
```

NPU 外围的内存复制、同步和布局转换可能抵消收益。因此要看**关键耗时算子覆盖率与分区边界**；“90% 的算子被卸载”若剩下的是最贵的 attention，整体仍可能慢。

AOT 通常会完成常量处理、算子 lowering、融合/调度与存储规划；具体融合策略在厂商编译器内部。本项目能观察分区报告与产物，无法仅靠这些 Python 包装代码证明某个 DiT 已使用 FlashAttention 或某项专有融合。

### 5.4 AOT 文件和退出码要一起读

`convert_dit.py:1086–1093`：

```python
except FileNotFoundError as e:
    log(f"  ✗ AOT 组件缺失：{e}")
    log("    这就是 preflight 预告的坑。见 preflight 里的 bazel 补法，或改用 --skip-aot")
    return None
except Exception as e:
    log(f"  ✗ aot_compile 失败：{type(e).__name__}: {e}")
    log(f"     {str(e)[:400]}")
    return None
```

主流程 `convert_dit.py:1317–1325,1347`：

```python
if args.skip_aot:
    log("[5/6] AOT 分区诊断 —— 已跳过（--skip-aot）")
else:
    log("[5/6] AOT 委派 + 分区诊断（能否上 NPU 的硬判据）")
    aot_partition(quant, args, out_dir, artifacts)

hr()
log("[6/6] 端上编译清单")
log(f"  ✓ {emit_plan(out_dir, args, artifacts)}")
# ...
return 0
```

调用方没消费 `None`，因此 AOT 失败仍可能生成 plan 并返回 0。`aot_verify.txt` 的 FAIL 也只落盘，不阻止后续。生产程序应把“仅导出成功”“仅量化成功”“AOT 成功”“真机执行成功”建成明确状态。

产物内的 `DISPATCH_OP`、bytecode offset/size、编译器标记可以帮助确认“确实生成了厂商编译内容”。字符串和扫描二进制布局属于版本相关的诊断方法，不能替代结构化解析、兼容检查与实际执行。

### 5.5 MT6897 支持状态：用对证据层级

截至本次分析，官方 MediaTek 文档与 LiteRT 2.3.0 target 都列有 MT6897；公开问题中又存在特定 LiteRT-LM/运行库组合无法初始化或执行的记录。**结论是：MT6897 有编译目标与官方列出的集成入口；具体模型、运行库和手机 ROM 的闭环尚需设备证据。**

公开 issue 的模型与版本条件见第 13 章来源索引。不要把“枚举中存在”说成“任何 LLM 均已支持”，也不要把某个 LM 配置失败说成“整颗 NPU 对 LiteRT 永久不可用”。

---

## 6. 从零到可部署产物：推荐操作路线

### 6.1 在 WSL2 / Linux 准备固定版本环境

以下操作是方案，没有在本次执行。已具备 x86_64 Linux 与 CPython 3.12 时，在内层源码目录运行：

```bash
cd /mnt/d/misc/pc_convert/pc_convert
bash install_offline.sh
source venv/bin/activate
```

离线安装脚本的 pip 升级联网问题见 §2.1；严格隔离环境应先修该行。此处不推荐重新走无固定版本的联网安装，以免分析依据与运行依赖漂移。

### 6.2 准备模型 wrapper 和真实校准输入

先确定六项合同：网络结构/参数名；输入顺序；每个输入的 shape/dtype/layout；时间步与位置编码；签名/输出；前后处理。准备 CPU FP32 基线。

对符合本脚本输入假设的 DiT，可用下面命令形成候选产物：

```bash
python3 convert_dit.py /models/dit.safetensors \
  --repo-or-script /models/model_def.py \
  --soc MT6897 \
  --height 512 --width 512 \
  --hidden-dim 2048 --text-len 77 --text-dim 2048 \
  --recipe w8a16 \
  --out ./out_mt6897_512_w8a16
```

参数中的 2048/77 是脚本示例默认值，必须换成真实模型定义。该命令未传 `--calib`，走随机校准，**用途限链路候选**；质量交付应先实现 §4.3 的完整真实样本加载，再进行校准。现有 `--calib` 仅真实 latent 的局限同样适用。

若已有合适的 FP32 `.tflite`：

```bash
python3 convert_dit.py /models/model_fp32.tflite \
  --soc MT6897 --recipe w8a16 --no-rope-tables \
  --out ./out_mt6897_existing
```

这条支路跳过 PyTorch，但仍强制进入当前量化流程。已有正确量化模型无需再量化，应直接调用 AOT API：

```python
from ai_edge_litert.aot.aot_compile import aot_compile
from ai_edge_litert.aot.core import aot_types
from ai_edge_litert.aot.vendors.mediatek import target as T

result = aot_compile(
    "/models/model_quantized.tflite",
    output_dir="./out_mt6897_aot",
    config=[aot_types.CompilationConfig(target=T.Target(T.SocModel.MT6897))],
    keep_going=False,
)
if not result.models_with_backend or result.failed_backends:
    raise RuntimeError(result.compilation_report())
print(result.compilation_report())
result.export("./out_mt6897_aot", model_name="model_npu")
```

该 API 用法来自随包 2.3.0；它解决“请求正确后端并导出结果”，还需确认报告内实际分区覆盖。导出文件名带 backend target 后缀，部署时使用真实生成文件，不要猜成固定 `dit_npu.tflite`。

### 6.3 产物清单怎样解释

```text
out_mt6897_512_w8a16/
  dit_fp32.tflite                 # 权重入口导出时才有
  *_norm.tflite                   # 发生规整修改时才有
  dit_w8a16.tflite                # 静态量化成功时
  dit_npu*.tflite                 # AOT 导出，必须确认目标 MT6897
  aot_report.txt                 # 后端/分区结果
  aot_verify.txt                 # 本脚本的结构/字节码诊断
  compile_plan.json              # 部署说明，存在硬编码问题
  rope_*.bin                     # 未关闭且生成成功时；尚需消费端接入
```

这些文件“都在”也不足以判成功；旧输出目录可能混入陈旧结果。独立目录及明确实际产物选择，比文件名里加一个 `npu` 有用得多。

### 6.4 哪些 MT6991 痕迹必须去掉

`convert_dit.py:1156–1159`：

```python
"vendor_lib": "/vendor/lib64/libneuron_runtime.8.so -> mt6991/",
"acceleration_gate": "LiteRtCompiledModelIsFullyAccelerated() 必须为 true，"
                     "否则静默回 CPU",
"disk_note": ".tflite 与 .dla 双份驻留（玑绘实测 .npug 443MB + .dla 448MB）"},
```

即使传 `--soc MT6897`，这段文字仍写 MT6991 路径，并过度解释状态标志。应从实际设备能力生成运行说明；不要照抄 vendor 私有目录，也不要只靠 `runtimeMajor==8` 放行。

还要修正三类接口：部署脚本的设备识别/ADB；运行器的真实多输入输出合同；缓存/模型清单中的 target、shape、effective recipe、工具链和 runtime 兼容信息。本文负责指出修改位置与影响，保留原程序用于对照。

---

## 7. Android 上怎样真正用 NPU 运行

### 7.1 先配运行库，再谈 `Run`

推荐使用同一发行版的官方组件。LiteRT 2.3.0 把厂商 NPU 运行库转为独立 Maven 分发；旧 runtime ZIP 方案面向更早版本。[LiteRT 2.3.0 发布说明](https://github.com/google-ai-edge/LiteRT/releases/tag/v2.3.0)

Android Gradle 的依赖示意：

```kotlin
dependencies {
    implementation("com.google.ai.edge.litert:litert:2.3.0")
    implementation("com.google.ai.edge.litert:litert-npu-runtime-mediatek:2.3.0")
}
```

这两个坐标的配套使用可见官方样例的问题修复记录，但该记录针对 MT6991，不代表 MT6897 已实测成功。[官方 sample issue #102](https://github.com/google-ai-edge/litert-samples/issues/102)

设备 ABI 选择 `arm64-v8a`。主机的 x86_64 `libneuron_adapter.so` 不能推到 Android 替代设备库；ARM Linux glibc 程序也不能自动替代 Android bionic 程序。系统存在私有 vendor 库，不代表普通 App 的 linker namespace 允许任意 `dlopen`。

NPU sample 的 Android API 要求、LiteRT 通用 minSdk、NDK 的编译 API 下限是三个不同约束，按所选 2.3.0 集成示例和目标系统配置，不能只抄一个数字。[官方 NPU 集成指南](https://developers.google.com/edge/litert/next/npu)

### 7.2 推理对象与 buffer 的生命周期

下面是**生命周期伪代码**，用于面试解释；正式工程应使用 2.3.0 配套 C/C++ 头文件或 Kotlin API，不按本包手写 ABI：

```text
应用初始化：
  environment = 创建 LiteRT 环境并注册/定位 MediaTek Dispatch
  model = 读取完整 MT6897 AOT tflite
  options = 选择 NPU；明确是否允许 fallback
  compiled = 创建 CompiledModel(environment, model, options)
  查询签名、所有 I/O tensor 的 shape / dtype / buffer requirements
  inputs  = 为每个输入创建匹配且满足对齐/内存类型要求的 buffer
  outputs = 为每个输出创建匹配的 buffer

每次推理：
  写入所有输入，按 tensor 的量化参数处理数值
  结束 CPU 写访问 / 完成必要的 cache coherency 同步
  提交 Run，等待同步返回或异步 completion/fence
  获得输出读权限，读取/反量化输出
  执行采样、scheduler 或其他后处理

应用退出或切换模型：
  等待在途任务结束
  释放 I/O buffers → compiled/model → environment
```

图的生命周期应长于一次推理。一次请求创建一次 CompiledModel，会重复付出解析、恢复、内存规划/分配等成本。反过来，把所有分辨率模型永久驻留，可能把手机内存吃光；只驻留当前工作集。

**DMA 与零拷贝**：runtime 可能利用共享内存、AHardwareBuffer 或 DMA buffer 降低复制。选了共享 buffer 仍需遵守 CPU/NPU 读写同步与 fence 规则；缺一次同步，读到旧数据的概率就会上升。`DMA_BUF_IOCTL_SYNC` 处理可见性，不配置 LLM 的 token cache。

### 7.3 当前部署脚本首先会卡在 ADB 包装

`install_on_device.sh:12–23` 节选：

```bash
SRC="${1:-./npu_out}"
SER="${2:-}"
if [ -n "$SER" ]; then ADB=(adb -s "$SER"); else ADB=(adb); fi
# ...
adb(){ "${ADB[@]}" "$@"; }
```

函数体解析命令名 `adb` 时又找到自己，形成递归。后续实际推送前必须修复。建议保留调用接口，将函数体改为：

```bash
adb(){ command "${ADB[@]}" "$@"; }
```

这段是建议修改，原文件未改。即使修了，它也只是把顶层文件推到下载目录，既不安装 runtime，也不启动推理。普通 App 要用 app-private storage 或正规的文件选择/导入流程。

### 7.4 `litert_npu_check.cpp` 到底做了什么

`litert_npu_check.cpp:91–104` 节选：

```cpp
LiteRtModel *model = NULL;
r = modelFromFile(env, modelPath, &model);
// ...
LiteRtCompiledModel *cm = NULL;
r = createCompiled(env, model, opts, &cm);
// ...
bool fully = false;
r = isFully(cm, &fully);
```

这个程序完成模型创建和状态查询，**没有输入输出 buffer 与推理 Run**。它还手工定义若干 ABI 结构、选项编号、函数指针，版本变化有调用错位风险。正式运行器使用官方头文件，不靠反汇编猜结构体布局。

`IsFullyAccelerated` 是图/委派状态的信号，不能独自证明实际设备执行，更不能证明结果正确。随包 `WHEEL:ai_edge_litert/environment.py:217–224` 的注释明确记录过一类误导路径：

```python
# Auto-discover the compiler-plugin directory too, so JIT compilation
# for raw .tflite models works out of the box. Without this, dispatch
# loads and silently returns a non-partitioned model that the TFLite
# fallback interpreter runs on CPU while the benchmark still reports
# "Fully accelerated: True". Pair the compiler plugin with whichever
# vendor's dispatch library was selected, so mixed wheels (e.g.
# google_tensor + intel_openvino) do not cross-wire an Intel compile
# into a google_tensor dispatch path.
```

这个源码警告说明“需要组合证据”；它不意味着该 API 在所有版本都没有价值。正确用法是同时记录请求后端、实际分区、运行库加载、真实执行及数值。

### 7.5 直接 Neuron 探针在哪一步真正执行

`aot_on_apu.cpp:114–122,165–170` 节选：

```cpp
NeuronModel *model = NULL; NeuronCompilation *comp = NULL;
int r = fp_res(&model, &comp, bc, (size_t)bcn);
if (r || !comp) return 4;
NeuronExecution *ex = NULL;
r = fp_exc(comp, &ex);
if (r || !ex) return 5;
```

```cpp
r = fp_sin(ex, 0, NULL, Xin, (size_t)nin * inEb);
if (r) { say("  setInput -> %d (第 %d 个)", r, s); return 7; }
r = fp_sout(ex, 0, NULL, Ymdla, (size_t)nout * outEb);
if (r) { say("  setOutput -> %d", r); return 7; }
r = fp_cmp(ex);
if (r) { say("  compute -> %d (第 %d 个)", r, s); return 7; }
```

对应的 API 是：

```text
NeuronModel_restoreFromCompiledNetwork
  → NeuronExecution_create
  → NeuronExecution_setInput / setOutput
  → NeuronExecution_compute
```

`restore` 接受的是已经编译好的网络。当前 `verify_on_apu.sh` 从 AOT 容器提取 `.dla` 后调用它；这条脚本没有把通用 TFLite 翻译成 Neuron operands/operations，也没有现场调用 `NeuronCompilation_finish`。

探针的限制很具体：只绑定 index 0 的单输入与单输出；只按 signed int8/int16 解释数据；固定 32 个样本；循环内分配内存却没有逐次释放。因此它适合原作者小 MLP 的诊断，无法原样承担三输入 DiT 或多层 KV 的 LLM。

直接 Neuron **现场编译**若另行实现，才是下面这条职责链：

```text
解析模型 → 创建 NeuronModel
  → addOperand / setOperandValue / setOperandSymmPerChannelQuantParams
  → addOperation / 指定模型输入输出 / Model_finish
  → Compilation_createForDevices / 配置优化项 / Compilation_finish
  → 导出编译网络或创建 Execution → compute
```

列出符号名远远少于实现这套翻译器的工作量。当前需求优先用 LiteRT compiler/dispatch 已有能力，省下自己维护全套算子翻译的债务。

### 7.6 为什么不推荐原样运行 `litert_stage.sh`

`litert_stage.sh:18–20,32–36` 节选：

```bash
RT=${RT:-/data/local/tmp/rt}
NDK=${NDK:-/tmp/work/android-ndk-r27c/toolchains/llvm/prebuilt/linux-x86_64}
ADB="/root/dsh-bin/adb-shell --su"
# ...
SRC=/root/jihui/dec/assets/litert
for f in libLiteRt.so libLiteRtDispatch_MediaTek.so libLiteRtCompilerPlugin_MediaTek.so libc++_shared.so; do
  [ -s "$RT/$f" ] && { echo "    已有 $f"; continue; }
  [ -f "$SRC/$f" ] && cp "$SRC/$f" "$RT/$f" && echo "    ✓ $f" || echo "    ✗ 缺 $SRC/$f"
done
```

它绑定特定容器、root 包装命令和第三方 APK 解包路径，还会写 `/vendor` 位置。MT6897 普通手机没有这些前提。不要为跑这个探针去修改系统 vendor 分区；正式 App 走官方运行库分发与系统驱动接口。

依赖复制还有确定的假成功漏洞，`litert_stage.sh:22–27` 节选：

```bash
python3 -c "open('/sdcard/Download/_p.bin','wb').write(b'\x7fELF'+b'\0'*200)"
$ADB "toybox cp $dev /sdcard/Download/_p.bin" >/dev/null 2>&1
head -c 4 /sdcard/Download/_p.bin 2>/dev/null | grep -q ELF || return 1
cp /sdcard/Download/_p.bin "$dst"
```

占位文件已经有 ELF 魔数，复制失败仍可能被误收为真库。修复重点是复制退出状态与有效目标 ELF 的确认，不是继续给缺失的库改名。

### 7.7 如何建立“确实使用 NPU”的证据

| 证据层 | 能回答的问题 | 单独不能证明的事 |
|---|---|---|
| `MT6897` target + AOT 分区报告 | 编译器选择了哪些子图 | 手机真的执行过 |
| Dispatch、Neuron 版本与恢复日志 | 运行时找到了对应执行后端 | 输入已计算且输出正确 |
| 真实 Run 成功；调试权限允许时关联设备任务提交/完成 | 当前请求进入了设备执行路径 | 量化无损、端到端更快 |
| 同一量化图、同一输入的输出比较 | 后端结果是否符合数值合同 | 原始模型质量没有下降 |
| 真实任务与时延/内存测量 | 可用性及加速收益 | 其他模型/ROM 组合也成立 |

量化参考与 NPU 对照最好共用**同一份整数输入**。Python `np.round` 与 C++ `lroundf` 对半整数的规则可能不同；若两端各自量化，会把输入差异混进后端误差。

CPU 占用降低、出现 `/dev/apusys`、看到一次 dma-buf sync、或破坏一个字节后报错，均可作辅助迹象；需与当前推理区间和成功完成的调用关联。公开产品不要求 root，调试取证方式应服从设备权限。

---

## 8. Cache size：先分对象，再给数字

### 8.1 七种“cache”经常被混成一件事

| 名称 | 缓存对象 | 单位 | 设置入口/归属 |
|---|---|---|---|
| **LLM KV cache** | 历史 token 每层 K/V | token 容量，换算字节 | LLM 导出配置 + 会话 runtime |
| 编译产物缓存 | AOT 文件、DLA/字节码 | 磁盘字节 | App 分发/磁盘缓存策略 |
| Attention mask cache | 因果/有效位置遮罩 | 张量尺寸 | 模型导出器与 runtime |
| RoPE 表 | 位置对应 sin/cos | 位置数 × 维度 × dtype | 位置编码实现 |
| NPU 工作区 | 激活、scratch、arena、传输 buffer | 字节 | 图编译器与 runtime |
| APU L1 / SRAM | 芯片局部工作数据 | KB | 硬件容量及编译使用策略 |
| Dynamo 编译缓存 | 捕获/编译的图变体 | 变体数量 | PyTorch 导出器配置 |

例如 `WHEEL:litert_torch/generative/utilities/converter.py:350–352` 节选：

```python
# pylint: disable=protected-access
torch._dynamo.config.cache_size_limit = 64
```

64 是图编译缓存的限制；把它改成 2048，LLM 上下文不会因此长出 1984 个 token。

### 8.2 DiT 为什么不能直接套自回归 KV cache

自回归 LLM 的旧 token 在因果图中不依赖未来 token，因此旧位置的 K/V 可复用。DiT 每次去噪后 latent 都变了，图像 self-attention 的 K/V 通常要重新计算。

可复用对象包括固定文本 embedding；某些模型中与 timestep/latent 无关的 cross-attention 投影也可复用。必须先证明依赖不变。当前主脚本没有跨去噪步的 KV 管理。

DiT 的主要容量旋钮是分辨率、patch 数、hidden dim、batch、精度、分区和激活生命周期。`--seq-len` 需要与图像网格一致，不能任意加大当预留内存。

### 8.3 LLM 的 prefill / decode 是怎样用 KV 的

```text
tokenize + chat template → prompt tokens
        │
        ▼
prefill：处理提示词（一次或分 chunk）
        每层生成历史 K/V，写入 cache；最后位置 logits 用于采样
        │
        ▼
decode：把新 token 送入模型，逐层计算当前 Q/K/V
        当前 Q 对“历史 K/V + 当前 token K/V”做 attention
        按当前 position 持久化新 K/V，更新有效长度/mask
        最后一层输出 logits → 采样下一 token
        到 EOS / 预算 / 容量边界停止
```

通常只保存 K/V：下一次 attention 用新位置的 Q，不需要历史 Q。KV cache 避免重复计算历史 token 的投影及各层表示，decode 仍需访问历史 K/V，因此长上下文会增加读取与 attention 开销。

标准因果 attention 允许关注当前位置，只屏蔽未来。当前 K/V 必须参与这一步 attention；split-cache 方案可以把持久 buffer 的写回交给辅助图，不能因此漏掉当前 token 的 self-attention。

mask 必须屏蔽未填充位置，position 与 RoPE 必须对应正确的绝对/相对位置。新会话重置或隔离状态；改 system prompt、编辑历史或换权重后，要重算受影响的前缀。

### 8.4 容量怎么选：先算 token 预算

不使用窗口淘汰时，保守预算为：

```text
cache_length >= 完整输入 token 数 + max_new_tokens

完整输入包括：system prompt + chat template + 特殊 token
            + 全部保留历史 + 本轮输入 + 多模态展开 token
```

示例：总输入 1500 token，输出预算 400，选择 2048 档尚余 148。输入 3000、输出 512，则至少需要支持 3512 的档位，例如已导出且匹配的 4096 档。这里的 token 必须用对应 tokenizer 数，不能用汉字数估。

**首个 MT6897 LLM 工程实验，我选静态 cache 2048、prefill chunk 128、batch 1。** 它便于先验证多步状态和设备链，同时控制图与工作区规模。业务确实需要长文，再上 4096 档并重新核算内存与延迟。2048 是实验起点，不是 MT6897 的硬件上限。

超长输入应明确拒绝、裁剪/总结历史，或选择已有长档位。滑窗、ring buffer、KV 压缩需要模型与 runtime 支持；不能静默覆盖老 K/V 后继续按原位置使用。

### 8.5 KV 内存公式：面试必须能现场算

标准、各层等维的 dense attention：

```text
M_KV = 2 × B × N_layer × T × H_kv × D_head × bytes_per_element

2：K 和 V 两份
B：batch / 有效并发序列数
N_layer：层数
T：缓存 token 容量
H_kv：K/V 头数（注意与 Q 头数区分）
D_head：每头维度
bytes_per_element：FP16/INT16=2，FP32=4，INT8=1
```

- MHA：`H_kv = H_q`。
- GQA：一组 Q 头共享 K/V，`H_kv < H_q`。
- MQA：`H_kv = 1`。

若各层不等维、采用局部滑窗或其他注意力架构，应逐层求和：

```text
M = B × Σ_layer [T_layer × (H_k×D_k×bytes_k + H_v×D_v×bytes_v)]
```

量化 KV 还要加 scale/zero-point、分组、对齐等开销；运行时的输入输出 cache 若不能原位复用，还可能出现重复 buffer。

#### 真实配置例：本包 Qwen2.5-1.5B

`WHEEL:litert_torch/generative/examples/qwen/qwen.py:34–37,67–76`：

```python
attn_config = cfg.AttentionConfig(
    num_heads=16,
    head_dim=128,
    num_query_groups=2,
```

```python
def get_1_5b_model_config() -> cfg.ModelConfig:
  """Returns the model config for a Qwen 2.5 1B model."""
  config = get_3b_model_config()
  # Qwen has only one block config.
  block_config = config.block_config(0)
  block_config.attn_config.num_heads = 12
  block_config.ff_config.intermediate_size = 8960
  config.num_layers = 28
  config.embedding_dim = 1536
  return config
```

函数名为 1.5B，docstring 的“1B”与之不一致；计算用实际字段：28 层、12 个 Q 头、**2 个 KV 头**、head_dim=128。示例假定 B=1、KV FP16：

```text
每 token = 2 × 28 × 2 × 128 × 2
         = 28,672 bytes = 28 KiB
```

| Cache token 数 | FP16 KV 主体 | FP32 KV 主体 | 假设同维改成 12 KV 头的 MHA，FP16 |
|---:|---:|---:|---:|
| 512 | 14 MiB | 28 MiB | 84 MiB |
| 1024 | 28 MiB | 56 MiB | 168 MiB |
| 2048 | 56 MiB | 112 MiB | 336 MiB |
| 4096 | 112 MiB | 224 MiB | 672 MiB |
| 8192 | 224 MiB | 448 MiB | 1344 MiB |

表中是按配置计算的逻辑张量大小，未测 MT6897 物理占用，也没有包含权重、activation、mask、scratch 和 driver buffers。**权重 W4 与 KV 4-bit 是两项独立配置。** 最终以产物中 K/V 张量的 dtype、布局和量化元数据计算。

### 8.6 静态 cache 会进入图的形状

本包 HF 导出器，`WHEEL:litert_torch/generative/export_hf/core/cache.py:405–411`：

```python
if k_ts_idx == 2:
  k_cache_shape = (
      1,
      batch_size * num_kv_heads,
      cache_length,
      embed_size_per_head,
  )
```

`WHEEL:litert_torch/generative/export_hf/core/exportable_module.py:502–505,610–614`：

```python
"mask": torch.ones(
    (1, 1, prefill_length, cache_length),
    dtype=torch.bool if use_bool_mask else torch.float32,
    device=device,
```

```python
"input_pos": torch.ones((1), dtype=torch.int32, device=device),
"mask": torch.ones(
    (1, 1, 1, cache_length),
    dtype=torch.bool if use_bool_mask else torch.float32,
    device=device,
```

cache 2048 改 4096，关联 K/V、mask、prefill/decode 签名、buffer 分配与编译计划。静态方案通常需要重新导出、校准/量化和编译；若预先导出多档位，选择已存在的匹配档即可。

`max_new_tokens` 控制停止预算，无法扩容模型已经固定的 tensor shape。静态图即使实际历史很短，也可能分配完整 T，并计算被 mask 的填充区域；是否跳过这些工作要看实际 kernel，不能只看有效 token 数。[Hugging Face 官方 cache 策略说明](https://github.com/huggingface/transformers/blob/main/docs/source/en/kv_cache.md)

### 8.7 片上 L1 cache：与 KV 容量分别设置

LiteRT 2.3.0 MediaTek 编译代码 `compile_model.cc:144–160` 节选：

```cpp
neuron_adapter_api.api().get_l1_memory_size_kb(&apu_mem_size)
```

```cpp
neuron_adapter_api.api().compilation_set_l1_memory_size_kb(
    compilation->get(), apu_mem_size)
```

启用相关优化后，它查询 SDK 报告的 L1 大小，并把非零结果用于 compilation。单位是 KB，作用是片上存储使用策略；不会增加芯片物理 SRAM。没有足够的一手依据给 MT6897 编造一个固定“最佳 L1 数值”。[固定发行提交的实现](https://github.com/google-ai-edge/LiteRT/blob/4352e26/litert/vendors/mediatek/compiler/compile_model.cc#L144-L160)

`cache_length=2048` 表示 KV token 容量；`--mediatek_enable_l1_cache_optimizations=true` 是编译优化开关。两项不能互换。本项目 DiT CLI 没有暴露任意设置 APU L1 KB 的参数。

### 8.8 编译缓存应该怎样管理

建议产物身份至少包含模型版本、target SoC、shape、有效量化配置、编译器版本与 runtime 兼容标识。配置变化就选择新产物或重新编译，不能继续命中旧文件。

磁盘缓存以字节/配额管理，并保留在用文件；App 可按真实使用频率淘汰不用的分辨率档。会话内复用 compiled model 与 buffers，主要节省初始化开销。KV cache 生命周期归具体会话，不能因两个对话 tensor shape 相同就交叉复用，否则既错答案也泄漏上下文。

---

## 9. 如果目标是 LLM：用独立生成模型管线

### 9.1 本包真的有 cache 配置，但在依赖里

三个入口的命名与默认值不同：

| 入口 | 参数 | 本包默认 | 用途 |
|---|---|---:|---|
| `generative/utilities/converter.py` | `kv_cache_max_len` | 1280 | 手工重写模型的旧式导出 |
| `generative/export_hf/core/exportable_module_config.py` | `cache_length` / `cache_lengths` | 4096 / None | HF 导出单档/多档 |
| `generative/export_hf/experimental/npu_export` | `cache_length` | 1280 | 实验性 NPU 四阶段流程 |

旧接口 `WHEEL:litert_torch/generative/utilities/converter.py:104–108`：

```python
flags.DEFINE_integer(
    'kv_cache_max_len',
    1280,
    'The maximum size of KV cache buffer, including both prefill and decode.',
)
```

因此不能笼统说“LiteRT 默认 cache=4096”，也不能给 `convert_dit.py` 塞一个它没定义的 `--cache-size`。

### 9.2 NPU 四阶段的设置入口

`WHEEL:litert_torch/generative/export_hf/experimental/npu_export/config_manager.py:144–151,191–198`：

```python
def build_pipeline_config(
    model_id: str,
    target_vendor: str,
    target_soc: str,
    prefill_lengths: list[int] | None = None,
    cache_length: int | None = None,
    max_calibration_decode_steps: int | None = None,
    **user_overrides: Any,
```

```python
merged["backend"] = clean_vendor
merged["soc_model"] = clean_soc

# 4. User Overrides
if prefill_lengths is not None:
  merged["prefill_lengths"] = prefill_lengths
if cache_length is not None:
  merged["cache_length"] = cache_length
```

源码级接线示例（模型目录、数据格式及依赖须符合该实验接口；尚无本任务的 MT6897 执行结果）：

```python
from litert_torch.generative.export_hf.experimental.npu_export.config_manager import (
    build_pipeline_config,
)
from litert_torch.generative.export_hf.experimental.npu_export.stages import (
    npu_export, npu_calibrate, npu_quantize, npu_compile,
)

cfg = build_pipeline_config(
    model_id="./models/qwen2.5-1.5b",
    target_vendor="mediatek",
    target_soc="MT6897",
    prefill_lengths=[128],
    cache_length=2048,
    max_calibration_decode_steps=32,
    calibration_dataset_format="jsonl",
    calibration_eval_task_names=["prompts"],
)
cfg.save_json("./llm_out/pipeline.json")

exported = npu_export(cfg, output_dir="./llm_out/export")
npu_calibrate(
    cfg, input_litertlm=exported,
    calibration_dataset_dir="./calibration_text",
    calibration_result_save_dir="./llm_out/calibration",
    enable_formatting=False,  # text 已用同一模型 tokenizer 格式化，见下文
)
npu_quantize(
    cfg, input_litertlm=exported,
    output_litertlm="./llm_out/quantized.litertlm",
    calibration_dir="./llm_out/calibration",
)
npu_compile(
    cfg, input_litertlm="./llm_out/quantized.litertlm",
    output_litertlm="./llm_out/mt6897.litertlm",
    compile_configs={
        "prefill_decode": {
            "compile": True,
            "flags": ["--mediatek_enable_gemma_compiler_optimizations=false"],
        },
        "aux": {"compile": False},
    },
    keep_temporary_files=True,
)
```

`prefill_lengths=[128]` 指 prefill 输入 chunk/signature 长度；运行器支持 chunked prefill 时可以分批吃长 prompt。`cache_length=2048` 管整段会话容量。`max_calibration_decode_steps=32` 控制校准轨迹长度，既不是产品输出上限，也不是 KV 总容量。

`npu_export()` 返回导出的 `model.litertlm` 路径字符串；`cfg.save_json()` 会建立父目录。`compile_configs` 在这个版本**整体替换默认编译配置**，没有自动深合并。上例只编译 prefill/decode，辅助图留 host，并关闭 Gemma 专项优化；它是 Qwen 首轮适配的保守起点，不能保证芯片支持。

#### 校准文件必须按真实读取合同准备

准备 `./calibration_text/prompts.jsonl`，每行是 `{"text": "完整提示串"}`。这里的完整提示串先由源模型配套 tokenizer 的 chat template 生成；真实多轮也应一起格式化：

```python
import json
from pathlib import Path
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("./models/qwen2.5-1.5b")
messages = [
    {"role": "system", "content": "你是一位数学教师。"},
    {"role": "user", "content": "80 元的商品打八折，是多少钱？"},
]
text = tok.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)
Path("./calibration_text").mkdir(parents=True, exist_ok=True)
with open("./calibration_text/prompts.jsonl", "w", encoding="utf-8") as f:
    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
```

这只有一条格式示例；真实校准应覆盖任务、长度和 decode 轨迹，不能拿一个八折题代表全部使用分布。上方 `enable_formatting=False` 防止二次套模板。

为什么一定显式写 `calibration_eval_task_names=["prompts"]`？`WHEEL:litert_torch/generative/export_hf/experimental/calib/quant_utils.py:62–65`：

```python
if dataset_dir is not None:
  return {
      task_name: f'{base_dir}/{task_name}.{ext}' for task_name in task_names
  }
```

它拼接指定文件名，不扫描目录；保留默认 `ALL` 会尝试找 `ALL.jsonl`。开源读取路径接受 JSON 数组或 JSONL，不接受本项目 DiT 使用的 NPY，也不接受裸 TXT。

该版本的 `messages` 分支还会丢掉非 user 角色，`quant_utils.py:145–152`：

```python
if 'messages' in example:
  user_messages = [
      msg for msg in example['messages'] if msg.get('role') == 'user'
  ]
  if tokenizer is not None:
    return tokenizer.format_chat_prompt(
        user_messages, enable_formatting=enable_formatting
    )
```

因此本报告选“源 tokenizer 预格式化 → JSONL text → 禁止二次格式化”的路径，保留 system/assistant 历史。模型包缺模板时，内部格式器可能回退到 Gemma 标记；对 Qwen 来说，这会改变校准的真实输入分布。

### 9.3 默认 target、layout 与编译选项必须看

`WHEEL:litert_torch/generative/export_hf/experimental/npu_export/configs/vendor_configs.py:28–35`：

```python
"mediatek": {
    "backend": "mediatek",
    "default_soc": "mt6993",
    "k_ts_idx": 2,
    "v_ts_idx": 3,
    "a16w8": True,
    "allow_float_operations": True,
},
```

必须显式覆盖 MT6897。配置构建函数主要验证厂商与赋值 target，没有充分验证 MT6897 运行支持。K 的时间轴在 2，V 的时间轴在 3；相同元素数不意味着能用相同 strides 或回写方法。

编译器默认项，`WHEEL:litert_torch/generative/export_hf/experimental/litert_lm_npu_compiler/litert_lm_npu_compiler_configs.py:56–64`：

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

通用配置启用了 Gemma 优化与 turbo，库没有给 MT6897 单独提供一套覆盖参数。对 Qwen 等其他架构应审查这些项，通过 `npu_compile(..., compile_configs=...)` 提供项目确认的配置。turbo 模式也不能保证持续性能，热降频会改变结果。

第一阶段默认 `dynamic_wi8_afp32` 是中间导出形式；第三阶段会用真实校准重新生成静态 A16W8。不能看到中间配置的动态 recipe，就认定最终产物仍是浮点激活；同样不能只看 `a16w8=True` 就认定所有分区满足 MT6897 kernel 要求。

### 9.4 跨 prefill/decode 的 cache 量化参数必须一致

`WHEEL:litert_torch/generative/export_hf/experimental/npu_export/stages.py:68–72`：

```python
overrides = {
    "output_dir": output_dir,
    "split_cache": True,  # Mandatory for NPU backends
    "keep_temporary_files": keep_temporary_files,
    **kwargs,
```

这条支路默认拆分 cache 更新职责。MediaTek 的辅助图在该实验编译器中保留主机执行，prefill/decode 才是重点 NPU 编译对象；整个生成引擎仍是 CPU/NPU 合作。

`WHEEL:litert_torch/generative/export_hf/experimental/calib/quantize.py:304–314`：

```python
print('--- Aligning KV cache parameters across models...')
quant_utils.align_kv_cache_params(
    calibration_results=calibration_result,
    model_path=model_path,
    kv_cache_k_patterns=kv_cache_k_name_pattern
    or _DEFAULT_KV_CACHE_K_PATTERN,
    kv_cache_v_patterns=kv_cache_v_name_pattern
    or _DEFAULT_KV_CACHE_V_PATTERN,
    aux_calibration_results=aux_calibration_result,
    aux_model_path=aux_model_path,
)
```

同一个整数 100，在 `scale=0.01` 和 `scale=0.1` 下代表完全不同的实数。prefill 输出、新 K/V slice、aux 更新和 decode 输入必须共享一致的量化语义，或显式 requantize。只把字节复制对了，生成质量仍可能逐 token 漂移。这是很值得在面试里主动讲的一层。

### 9.5 `.litertlm` 到聊天还缺哪些消费端职责

模型包需要与 LiteRT-LM runtime 对接，管理 tokenizer/chat template、BOS/EOS、签名选择、会话 cache、位置/mask、采样参数、停止条件、取消与资源回收。现有 `aot_on_apu.cpp` 没有这些对象。

当前官方 MediaTek LLM 示例表列出的预编译 Gemma3-1B 目标为 MT6989/MT6991，context 1280；它不能用来保证 MT6897 或任意 Qwen 的兼容性。[官方 MediaTek LLM 指南](https://developers.google.com/edge/litert/next/litert_lm_npu)

推荐推进顺序是：先在 MT6897 上建立固定 shape 小图的正式 Runtime 执行链，再做一个短上下文、batch 1 的完整 LLM prefill/decode，最后扩上下文和模型规模。第一轮最值得解决的未知量是**目标手机能否按所选版本完成 NPU Run**，先别拿长上下文把内存问题叠上来。

---

## 10. 内存、速度与精度：怎么做出有依据的取舍

### 10.1 总内存要按同时存活的对象计算

```text
推理峰值内存 ≈ 权重驻留
              + KV / 持久状态
              + 同时存活的激活与 attention 中间量
              + scratch / compiler-planned workspace
              + 输入输出与跨设备 staging buffers
              + runtime / driver / App 开销
              + 不能共享的副本
```

AOT 文件大小衡量磁盘产物，不能代替运行时峰值。量化 3B 参数到 W4 的裸权重下限约为 `3×10^9×0.5 bytes ≈ 1.40 GiB`，还没包括 scales、未量化层、对齐和重复存储。物理内存容量也不等于 App 可用预算。

NPU 使用系统内存、片上分块存储和中间缓冲的具体方式由 backend 决定。不要只查看进程 RSS 后就认定“全部 NPU 内存都统计了”；dma-buf/驱动分配可能需要单独观察，且 shared buffer 的多处计数要避免重复相加。

### 10.2 DiT 分辨率为什么涨得这么凶

在本脚本的 token 化规则下，H、W 各翻倍：

```text
L = (H/16)×(W/16)       → token 数变为 4 倍
线性层工作量约 O(LD²)  → 约 4 倍
朴素 attention O(L²D)  → 约 16 倍
```

举例，以 D=2048、8 个 attention heads 为计算假设：

| 张量 | L=1024 | L=4032 |
|---|---:|---:|
| 单份 `[1,L,2048]` FP32 hidden states | 8 MiB | 31.5 MiB |
| 若显式保存 `[1,8,L,L]` FP16 attention 分数 | 16 MiB | 248.0625 MiB |
| 本脚本 `[L,256]` FP16 RoPE 表 | 0.5 MiB | 1.96875 MiB |

这些是形状公式推导，不是 MT6897 测量。融合/分块 attention 可以避免完整 materialize `L×L` 分数，所以不能把第二行强行计入所有实现。应从实际图与编译 workspace 看真实峰值。

优先优化顺序：**先减少不必要的分区与跨设备搬运，再优化 attention/激活驻留，最后调精度和分辨率档位。** 采样步数下降能减少重复 DiT 执行，但可能改变画质；模型是否蒸馏到少步数，需要该模型自己的训练/质量依据。

### 10.3 LLM 的速度要拆 prefill 和 decode

| 指标 | 衡量内容 | 常见陷阱 |
|---|---|---|
| 首次初始化时间 | 加载、恢复或 JIT 编译、分配资源 | 混入 token/s 后看起来极慢 |
| TTFT | 请求到首 token | 包含模板/tokenize、排队、prefill、首次采样 |
| Prefill throughput | 提示词处理速度 | chunk 长度不同，比较不公平 |
| Decode latency / token | 每个后续 token 的计算与采样 | 上下文长度不断变化，不能只报一个短序列峰值 |
| End-to-end latency | 完整请求耗时 | 省略 CPU 辅助图或采样会虚高 |
| 持续性能 | 热稳定后的吞吐、功耗、温度 | turbo 短跑成绩无法代表十分钟会话 |

对比 CPU/GPU/NPU 时固定模型、精度或分别说明精度、prompt token 数、输出 token 数、batch、采样策略、cache 档位与温度状态。首次冷启动和后续热执行分开记录。

增大 cache 主要换取上下文容量。decode 读历史 K/V，静态图还可能处理填充位置，容量增大常带来更多内存与时延。分页 KV、滑窗、前缀共享是具体实现能力；本项目没有自动获得这些优化。

### 10.4 怎样定位慢在 CPU、传输还是 NPU

先把一次请求切成时间段：

```text
读取模型 / 初始化
→ 输入预处理 / 量化
→ H2D 或共享 buffer 同步
→ NPU 执行（分区分别记录）
→ D2H 或输出同步
→ CPU 残留算子 / 采样 / VAE / 解码
```

随后对照：

- 分区很多且每段短：优先看调度与同步，合并/改写不支持算子。
- 设备执行长、激活巨大：看 shape、attention 实现、布局和工作集。
- 初始化长但热执行快：AOT、持久对象和编译缓存优先。
- 越跑越慢：看温度、频率、内存压力和泄漏。
- NPU 内核很快但整图慢：看 CPU 辅助路径及复制，不要只优化最显眼的矩阵乘。

README 曾用“权重文件大小/步耗时”估带宽，并把 TFLOPS 与 INT8 TOPS 相除推利用率。这样的计算缺少真实运算精度、FLOP 计数和全部内存访问量，不能据此排除带宽或算力瓶颈。一次推理还会读写激活、KV、scratch，权重也未必只读一遍。

### 10.5 保真评估要匹配任务

- **图导出**：同输入比较 PyTorch 与 FP32 LiteRT 的全部关键输出，必要时定位到层。
- **量化**：用代表性数据评估误差和任务质量；随机正态只用于小算子诊断。
- **后端**：同量化图、同整数输入比较 CPU/可靠参考与 NPU，排除前后处理差异。
- **DiT**：固定种子、prompt、scheduler、步数，观察去噪轨迹与最终图像质量。
- **LLM**：检查 prefill 与长 decode 的 logits/任务表现、EOS、长上下文状态、重复文本和 cache 越界。

相对误差在参考值接近 0 时不稳定；余弦忽略整体幅值差；整数输出的一步误差不能直接推广到所有深层网络。要用多指标解释变化，再给业务接受阈值。

---

## 11. 代码审查结论与排错路径

### 11.1 先修什么：按对结果的影响排序

以下是静态分析发现的行为，未修改原源码。

| 优先级 | 问题 | 直接影响 | 建议处理 |
|---|---|---|---|
| P0 | AOT 失败返回 None，主流程仍退出 0 | CI/使用者把未编译产物当 NPU 成果 | 明确阶段状态，AOT 必需时失败即非零退出 |
| P0 | 静态量化失败自动转动态 | 实际精度与请求/plan 不一致 | 正式路径停止；对照路径单独记录 effective recipe |
| P0 | `adb` 包装函数递归 | 设备推送无法开始 | 函数体用 `command` 或改 wrapper 名 |
| P0 | ELF 占位文件被当真库 | 依赖收集假成功，随后 loader 报错 | 检查复制返回码及 ELF 有效性，不预写“成功魔数” |
| P0 | 把小 MLP 模拟器用于未知复杂图 | 可能生成错误数值基准 | 不支持算子即拒绝，换匹配的参考实现 |
| P1 | `--calib` 只真实 latent，文本随机且 t=500 | 校准分布失真，画质下降 | 样本保存全部真实输入与实际签名 |
| P1 | 编译/导出输出目录混用旧文件 | 旧模型被误认成新结果 | 每次独立目录；显式落盘和实际文件选择 |
| P1 | 第 0 个 subgraph / 第 1 个输出的假设 | 多签名 LLM 与多输出模型覆盖不足 | 逐签名、逐 I/O 建立合同与检查 |
| P1 | plan 写死 MT6991，runtime major 只报警 | 换芯片的元数据与兼容提示误导 | 记录真实 target/依赖/设备匹配结果 |
| P1 | Neuron runner 固定单进单出和 32 样本 | 多输入错误绑定、样本长度不一致 | 按模型元信息构建 I/O；读入长度与 dtype 明确校验 |
| P1 | runner 循环分配而未释放 | 长会话内存增长 | 预分配复用，规范对象释放顺序 |
| P1 | 手工 LiteRT ABI、私有路径与库别名 | 换版本/ROM 易初始化失败 | 官方头文件、配套 runtime、正式 App 集成 |
| P2 | 离线安装偷偷尝试 pip 升级 | 离线合同不实、等待网络超时 | 删除升级或指定本地来源 |
| P2 | 少量样例相等被描述成完整等价 | 形成过强质量结论 | 说明比较的签名、输入与输出范围 |

P0 表示会阻止主路径或误导成功判定；P1 表示会破坏真实模型/长期运行；P2 表示可复现性与表述质量问题。

### 11.2 模拟基准为什么可能“看似对照，实际瞎算”

`apu_baseline.py:129–147` 节选：

```python
for op in sg.operators:
    code = int(m.operatorCodes[op.opcodeIndex].builtinCode)
    # ...
    if code == int(S.BuiltinOperator.FULLY_CONNECTED):
        wq, bq = tens(ins[1]), tens(ins[2])
        wsc, bsc = _scales(sg.tensors[ins[1]]), _scales(sg.tensors[ins[2]])
        acc = act @ wq.T.astype(np.float64) * cs
        acc = acc * (np.array(wsc)[None, :] if len(wsc) > 1
                     else (wsc[0] if wsc else 1.0))
        # ...
        act = np.clip(np.round(acc / osc[0]) + oz, -32768, 32767)
        cs = osc[0]
    elif code == int(S.BuiltinOperator.RELU):
        act = np.maximum(act, float(oz))
return act, cs
```

未知算子没有报错分支；数据流使用单一 `act` 串联，无法表达一般 DAG；还缺非零输入/权重 zero point、部分 fused activation 等语义。CPU Interpreter 失败后自动回退到这里，可能把“不支持的参考计算”伪装成已求出的标准答案。

这是很好的面试题：**验证器也有适用合同。** 参考实现覆盖不全，比较数字再漂亮也不能建立正确性。解决办法是限制输入模型集合并显式失败，或使用完整支持该量化语义的后端。

### 11.3 常见报错从哪一层开始查

| 现象 | 首查 | 最短处理方向 |
|---|---|---|
| wheel 不支持平台 | Python tag、OS、CPU 架构 | 本包固定 Linux x86_64 + CPython 3.12 |
| `'module' object is not callable` | `aot_compile` import | 使用 `from ai_edge_litert.aot.aot_compile import aot_compile` |
| `apply_plugin_main` 找不到 | wheel 内容/安装架构/路径 | 确认相对包根的 tools 与 vendors 目录 |
| 权重 missing / size mismatch | 类结构、参数名、config | 修模型定义和映射，勿靠 allow-partial 交付 |
| 导出输入不匹配 | forward 位置参数与 shape | 按真实模型写 wrapper |
| 量化前后张量统计不变 | 算子编码、权重 dtype、recipe 覆盖 | 区分旧 opcode 与非 FP32 权重，别只看返回码 |
| signature/calibration 失败 | SignatureDef、输入名、全部样本 shape | 对照模型的实际签名，修样本加载 |
| scale 非有限或非正 | 样本、输入名、未覆盖路径 | 找到具体张量与输入分布；别静默换 recipe |
| RESHAPE operand 数量错误 | 静态 shape 是否在 options | 物化常量 shape 输入并保留语义 |
| 量化 CPU kernel 不支持 | 实际 tensor/bias dtype | 使用适配参考后端，保留原始错误以区分真图问题 |
| AOT 只覆盖少量算子 | 不支持 op、dtype、shape、动态图 | 查最贵的残留算子，改写或拆业务子模块 |
| Android `dlopen` 失败 | ABI、依赖、namespace、许可/来源 | 用配套 arm64 官方运行库，勿靠换库名碰运气 |
| bytecode/runtime incompatible | SoC、编译器与实际加载 runtime | 用正确 target 与兼容发行组合重新编译 |
| “fully accelerated”但性能像 CPU | 真实 Run、fallback、Dispatch | 分离创建与执行证据，记录实际后端 |
| 第一个 token 正常，后续漂移 | KV layout、尺度、位置、mask、同步 | 对照 prefill→decode 的 cache 合同 |
| 长文本 OOM 或速度骤降 | 静态 T、mask、KV 副本、工作区 | 缩短到已支持档位并计算/测量真实峰值 |

### 11.4 哪些 README 结论可以保留

README 较后段记录了原作者在 MT6991 环境的小模型 AOT、APU compute 和数值对照；它更新了前文“尚未执行成功”的旧状态。该历史记录可以作为探索路径参考，不能改写为本次 MT6897 实测。

README 还把某次 `Compilation_finish=0` 描述关联到 `verify_on_apu.sh`，但当前脚本走“提取已编译字节码 → restore → compute”。这说明说明文档与代码历史存在错位。报告采用当前代码确定执行步骤，历史实验单独标记。

此外：

- 私有 NPUG 格式、第三方 App 的 token 约定和性能，只适用于所分析的那个系统。
- 某台设备 NNAPI vendor extension 受限，不能泛化到全部 MediaTek。Android 官方将 NNAPI 在 Android 15 标为弃用，新接入优先 LiteRT 的厂商直接集成。[Android NNAPI 文档](https://developer.android.com/ndk/guides/neuralnetworks)
- `s4` 尾缀含义仍未确认，不能解释为“4 个采样步”。
- 模型 header、文件字节数、编译器字符串只覆盖结构层证据，不能推出任务质量和硬件效率。

---

## 12. 面试题：从讲流程到接住追问

### 12.1 90 秒讲清这个项目

> 我把部署拆成模型语义、编译适配和设备执行三层。模型侧先用网络定义加载权重，固定真实输入 shape，导出 FP32 TFLite；然后用代表性样本校准，把权重和激活量化。编译侧选择 MT6897 target，用 MediaTek 插件划分可支持子图，再由 NeuroPilot 编译成嵌在 TFLite 中的厂商字节码。手机侧使用匹配版本的 LiteRT 与 MediaTek runtime，加载完整 AOT 模型，按签名创建所有输入输出 buffer，真正调用 Run。
>
> 我分别确认编译覆盖、实际执行、数值质量和端到端性能。当前代码默认 MT6991，需要显式换 MT6897 并重新编译；有 target 枚举不代表所有 ROM 和模型已经跑通。
>
> Cache 我会先分清对象：这套 DiT 主脚本的 seq-len 是图像 patch 数；LLM 的 KV cache 在独立生成管线设。它的容量覆盖 prompt 和输出，内存用 2×batch×层数×长度×KV 头数×head_dim×元素字节数计算，静态长度改变通常要重新导出编译。

以上可用于讲架构。面试介绍个人经历时，把“本次源码分析”“已有历史记录”“自己实机测过”分别说清楚。

### 12.2 基础题

**1. 为什么 `.safetensors` 不能直接烧进 NPU？**  
它提供参数，不携带任意网络的完整执行语义。还要网络结构、权重映射、输入输出、算子实现、量化与目标编译。

**2. 量化和编译的区别？**  
量化改变数值表示；编译把图映射成目标可执行分区与代码。一个 INT8 图也可能因算子/shape 不支持而无法上 NPU。

**3. 为什么在 PC 用 CPU-only torch 就够？**  
该流程主要做模型读取、图导出和离线编译，目标 NPU 的字节码由 host compiler 生成，无需 PC CUDA。大模型校准若另有 GPU 加速需求，应独立配置。

**4. AOT 后手机还要做什么？**  
仍需加载/恢复网络、创建执行资源、分配绑定 buffer、调度运行与同步；主要省掉部署时的图编译阶段。

**5. MT6897 与 MT6991 都是 v8，能否互用产物？**  
按各自 target 重新编译。SDK 大版本相同不表示架构、设备能力、优化计划和字节码合同相同。

**6. `.tflite` 与 `.dla` 哪个应该交给 App？**  
本项目推荐完整 AOT `.tflite` 交给 LiteRT。它保留多分区、签名、残留算子和调度关系；单独 `.dla` 适用于实现了相应 Neuron 合同的消费端。

### 12.3 量化与图编译题

**7. 为什么静态量化要校准？**  
它要从真实激活分布估计范围与 scale。输入分布错了，图可以编译成功，数值仍会严重失真。

**8. Per-channel 的价值？**  
不同输出通道的权重幅度差异大，逐通道 scale 能减少共享一个大范围造成的精度浪费；后端需支持对应量化维度与参数。

**9. W8A16 为什么出现 INT64 bias？**  
点积的乘积、累加与 bias 需要更宽表示，bias 的量化尺度还需与输入×权重 scale 对齐。W/A 位宽不约束所有中间量。

**10. CPU 参考装不下 W8A16，能判断模型损坏吗？**  
先读具体 kernel 错误与 dtype。若是参考后端缺少该组合，换支持它的参考；若是非法 shape/参数，修图。不能用“INT16 都这样”把真错误盖过去。

**11. 为什么固定 shape 对 NPU 有利？**  
编译器可提前做内存规划、布局和调度，减少运行时不确定性。代价是分辨率、cache 长度等变化常需新图/新编译档。

**12. NPU 算子覆盖 95%，为什么仍然慢？**  
算子数不代表耗时占比；残留可能是最贵算子。多个分区边界还引入复制、同步与布局转换。

**13. “转换成功”最少应包含什么？**  
先声明阶段。若交付可运行 NPU 模型，至少有正确目标的 AOT 产物、可执行运行链、匹配 I/O 和数值证据。文件存在与退出 0 只说明程序走到了某个出口。

### 12.4 Cache 重点题

**14. 为什么缓存 K/V，不缓存 Q？**  
下一步的新 query 要与历史 K/V 做 attention；历史 Q 的输出已算完，通常不会被下一 token 使用。

**15. GQA 怎样节省内存？**  
多个 Q 头共享少量 KV 头。公式用 `H_kv`，不能误用 `H_q`；本报告的 Qwen 配置示例 KV 只有 2 头。

**16. Prefill 128、cache 2048、max_new_tokens 400 有何区别？**  
128 是一次 prefill chunk 的输入长度；2048 是静态缓存容量；400 是生成停止预算。它们控制不同维度。

**17. 权重 4-bit，KV 也自动 4-bit 吗？**  
两者分别配置。查看最终 cache tensor 的 dtype 和量化参数，不从权重文件名推断。

**18. Cache 改成 4096，只改 App 参数可以吗？**  
对本报告的静态方案，要使用已经导出的 4096 档，或重新导出/量化/编译并匹配 buffer。改停止长度无法改变 tensor shape。

**19. 更大 cache 为什么可能更慢？**  
需要更大的驻留内存和历史访问；静态图还可能为被 mask 的位置做计算。容量与速度没有单调正收益。

**20. Cache 量化参数为什么要跨图一致？**  
prefill 输出会成为 decode 输入。同一整数若尺度不同，表示的真实 K/V 变了，错误随生成累积。

**21. 多轮对话何时能复用前缀？**  
token 前缀、权重、tokenizer/template、位置语义和会话状态一致时追加；改历史后从变化处重算。不同用户会话隔离。

**22. DiT 可以沿去噪步骤复用 self-attention KV 吗？**  
普通 DiT latent 每步变化，K/V 也变化，不能直接复用。只对已证明不变的条件分支或专门设计的近似缓存方法另行评估。

**23. L1 cache optimization 是设置 2048 个 token 吗？**  
它处理硬件局部存储的编译策略，查询/传递的容量单位为 KB；KV token 容量另有配置。

### 12.5 运行与排错题

**24. `IsFullyAccelerated=true` 能证明跑了 NPU 吗？**  
需要再看实际后端与 Run。当前 checker 没有执行输入，所以它只能提供创建/状态信息。

**25. 如何排除 CPU fallback？**  
诊断时明确允许的后端，检查分区、Dispatch/Neuron 日志与实际 invoke，必要时在授权环境关联驱动任务，最后比较数值和性能。

**26. 为什么普通 adb push 之后仍跑不起来？**  
文件推送没有安装依赖、建立 App 存储权限、准备 ABI/驱动接口，更没有实现 I/O 和 Run。

**27. Bytecode 版本检查通过却挂起，该查哪里？**  
分别检查 LiteRT↔Dispatch ABI、Neuron runtime/驱动、buffer 类型/生命周期、shape、事件同步和错误返回；单个版本字段只覆盖一层兼容性。

**28. 如何比较 CPU 与 NPU 的误差？**  
用同量化图、同一份整数输入先比较后端；再与 FP32 比量化损失，最后评估实际任务。把三类误差拆开。

**29. 第一步成功，连续运行出错是什么典型原因？**  
优先看状态和资源：KV 位置/mask、跨图尺度、buffer 复用同步、输入未全设置、对象提前释放、循环内存泄漏。

**30. 如果只能先改一个地方，你改什么？**  
我先修“失败仍报成功”的阶段状态：AOT 必需时失败就退出，静态量化失败不得冒充目标产物。可信的成功判定建立后，再处理模型覆盖与性能。

### 12.6 四道现场推导题

**题 A：28 层、2 KV 头、head_dim=128、FP16、batch=1，4096 token 的 KV 多大？**  
`2×28×2×128×2×4096 = 117,440,512 bytes = 112 MiB`。额外 workspace 和量化元数据另算。

**题 B：512×512 DiT 换到 1024×1024，朴素 attention 分数规模几倍？**  
token 从 1024 到 4096，为 4 倍；`L²` 为 16 倍。融合/分块会影响实际驻留内存。

**题 C：cache 2048，模板和历史后完整输入 1800，要求生成 400，怎样处理？**  
总预算 2200 超出 2048。限制输出到安全预算、明确裁剪历史，或选择已编译的更长档；不能直接写越界。

**题 D：FP32 输出正常、量化 CPU 正常、NPU 第二步开始乱码，第一怀疑什么？**  
先查 KV 的 shape/layout、输出→输入尺度、position/mask 和 buffer 同步；若无状态模型则看多次 I/O 复用与完整输入设置。不要先重新下载权重。

---

## 13. 证据来源与适用范围

### 本地版本证据

- 主程序：`D:\misc\pc_convert\pc_convert\convert_dit.py`。
- 设备实现：同目录的 Shell、C++ 与数值比较脚本，详见第 2 章。
- LiteRT 2.3.0 wheel：`D:\misc\pc_convert\pc_convert\offline\pip_pkgs\ai_edge_litert-2.3.0-cp312-cp312-manylinux_2_27_x86_64.whl`。
- LiteRT Torch 0.10.0 wheel：`D:\misc\pc_convert\pc_convert\offline\pip_pkgs\litert_torch-0.10.0-py3-none-any.whl`。
- SDK wrapper：`D:\misc\pc_convert\pc_convert\offline\pip_pkgs\ai_edge_litert_sdk_mediatek-2.3.0.tar.gz`。

本报告读取压缩包内源码与元数据，没有把包安装进当前环境。Google/LiteRT 源码片段保留于解释所需范围，相关文件采用 Apache-2.0 许可；MediaTek 二进制 SDK 按其自带许可另行管理，不能因包装器开源就推断 SDK 可任意再分发。

### 官方文档与固定版本源码

1. [LiteRT MediaTek 集成与 SoC 列表](https://developers.google.com/edge/litert/next/mediatek)：公开列表包含 MT6897。
2. [MediaTek Dimensity 8300](https://i.mediatek.com/dimensity-8300/)：APU 780、第七代 AI 处理器的产品说明。
3. [LiteRT v2.3.0 target.py](https://github.com/google-ai-edge/LiteRT/blob/v2.3.0/litert/python/aot/vendors/mediatek/target.py)：SoC 枚举与 NeuroPilot 大版本映射。
4. [LiteRT v2.3.0 mediatek_backend.py](https://github.com/google-ai-edge/LiteRT/blob/v2.3.0/litert/python/aot/vendors/mediatek/mediatek_backend.py)：插件、SDK 路径与 target 传递。
5. [MediaTek 编译实现](https://github.com/google-ai-edge/LiteRT/blob/4352e26/litert/vendors/mediatek/compiler/compile_model.cc)：编译与 L1 优化相关处理。
6. [MediaTek dispatch v2.3.0](https://github.com/google-ai-edge/LiteRT/blob/v2.3.0/litert/vendors/mediatek/dispatch/dispatch_api.cc)：设备执行接口与 runtime 兼容处理。
7. [LiteRT NPU 工作流](https://developers.google.com/edge/litert/next/npu)：AOT/JIT、Android 集成、运行库分发。
8. [LiteRT 2.3.0 release](https://github.com/google-ai-edge/LiteRT/releases/tag/v2.3.0)：Maven 依赖拆分等发行变更。
9. [MediaTek NPU 与 LiteRT 技术介绍](https://developers.googleblog.com/mediatek-npu-and-litert-powering-the-next-generation-of-on-device-ai/)：官方集成机制与编译策略。
10. [MediaTek LLM 指南](https://developers.google.com/edge/litert/next/litert_lm_npu)：具体 LLM/SoC 模型包条件。
11. [Transformers KV cache 文档](https://github.com/huggingface/transformers/blob/main/docs/source/en/kv_cache.md)：cache 策略及静态容量开销机制。
12. [Android NNAPI 文档](https://developer.android.com/ndk/guides/neuralnetworks)：API 状态，区别于具体手机的 vendor 限制。

### 兼容性冲突记录：保留具体条件

- [LiteRT issue #6886](https://github.com/google-ai-edge/LiteRT/issues/6886)：LiteRT 2.1.0、LiteRT-LM、Gemma-4-E2B 与 MT6897 设备环境的注册/加载问题。2026-10-15 的 Contributor 回复称 MT6897 当前不支持，与公开支持表存在冲突；该问题不能替代全部模型与 ROM 的结论。
- [LiteRT-LM issue #3413](https://github.com/google-ai-edge/LiteRT-LM/issues/3413)：用户报告 MT6897 在 Neuron 字节码兼容检查通过后挂起，以及不同 LM 版本的 dispatch ABI 问题。它说明兼容检查成功仍可能在后续失败。
- [litert-samples issue #102](https://github.com/google-ai-edge/litert-samples/issues/102)：MT6991 上升级配套 2.3.0 Maven runtime 的记录，用于确认集成方式，不能外推为 MT6897 验证结果。

在线页面会更新，报告日期为 2026-11-06；涉及具体 API 的代码结论优先采用随包版本与固定 tag。

### 本次交付与下一步

**已完成：**当前转换代码、离线依赖 API、设备执行探针、cache 设置与内存推导的技术分析，以及 MT6897 官方资料对照。

**尚缺的执行输入：**具体业务模型/权重与完整真实校准数据、MT6897 手机的 ROM/runtime 组合，以及该组合的真实推理结果。它们影响部署可用性，不影响本报告对当前源码行为的结论。

实际落地时，先修第 11 章的成功判定与部署阻断问题，用官方 Runtime 在 MT6897 完成一次固定 shape 小图的真实 Run，再接完整 DiT 或 LLM 状态循环。这样每往前一步，都清楚自己解决了哪一层问题。
