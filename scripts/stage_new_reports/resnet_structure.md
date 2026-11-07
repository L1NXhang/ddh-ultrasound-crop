# ResNet18 / ResNet50 结构层级对照（PPT 素材 A）

> 全部数字由 `tools/stage_new/measure_structure.py` 在**真实 torchvision 模型**上 hook 实测得到，
> 输入尺寸 = 本项目实际使用的 **384×384**。原始数据：`reports/resnet_structure_raw.json`

---

## 1 核心表：逐层输出分辨率 / 通道数 / 感受野

| 层级 | ResNet18 输出 | ResNet18 通道 | ResNet18 感受野 | ResNet50 输出 | ResNet50 通道 | ResNet50 感受野 | 参数量 18 / 50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| conv1 (7×7, s2) | 192×192 | 64 | 7 px | 192×192 | 64 | 7 px | 9,408 / 9,408 |
| maxpool (3×3, s2) | 96×96 | 64 | 11 px | 96×96 | 64 | 11 px | 0 / 0 |
| Layer1 (BasicBlock ×2 / Bottleneck ×3) | 96×96 | 64 | 43 px | 96×96 | **256** | 35 px | 147,968 / 215,808 |
| Layer2 (×2 / ×4) | 48×48 | 128 | 131 px | 48×48 | **512** | 139 px | 525,568 / 1,219,584 |
| Layer3 (×2 / ×6) | 24×24 | 256 | 483 px | 24×24 | **1024** | 811 px | 2,099,712 / 7,098,368 |
| Layer4 (×2 / ×3) | 12×12 | 512 | 1891 px | 12×12 | **2048** | 1963 px | 8,393,728 / 14,964,736 |
| **主干合计** | — | — | — | — | — | — | **11,176,512 / 23,508,032** |

> 说明：`11,176,512 / 23,508,032` 是**本项目**中 `fc=nn.Identity()` 后的主干实测参数
> （`cropnet_v3.param_breakdown` 输出，见 `results_all_configs.csv` 的 `params_backbone`）。
> 文献常引用的 `11,184,720 / 23,540,816` 是**含 ImageNet 1000 类 fc 层**的总参数，口径不同，不要混用。

**感受野计算口径**：`r_out = r_in + (k−1)·jump_in`，`jump_out = jump_in · stride`，逐层累加（含 conv1 与 maxpool）。
这是理论感受野，不是有效感受野；不同论文因是否计入 maxpool 会略有差异。

---

## 2 三条信息变化规律（PPT 三大结论）

### 结论 1｜空间分辨率下降，通道数上升 —— 这是 ResNet 的基本设计取舍

输入 384×384 → Layer4 只剩 12×12，**空间分辨率下降 32 倍**；同时通道数 64 → 512（18）或 64 → 2048（50），**通道数上升 8 倍 / 32 倍**。
这意味着：网络把"在哪里"（高分辨率、低语义）逐步转成"是什么"（低分辨率、高语义）。

📊 图：`figures/fig1_resnet_structure_compare.png`（三联图）、`figures/fig2_resolution_channel_tradeoff.png`（尺寸示意）

### 结论 2｜感受野逐层增大 —— 语义增强，但定位变粗

Layer1 的感受野只有 43 px（18）/ 35 px（50），到 Layer3 已达 **483 px / 811 px**，Layer4 超过 1800 px。
本项目输入只有 384×384，**Layer3 的感受野已经覆盖整个输入**，Layer4 的感受野是输入的 5 倍——所以 Layer4 只能给出"整体质量/语义"级别信息，**不适合做精细的边界定位**。这正是原代码只把 `layer3` 送进裁剪头、把 `layer4` 只用于 `quality_head` 与 `embedding` 的原因。

### 结论 3｜ResNet50 用"更宽的通道 + 更深的瓶颈"换语义，但在 384 输入下分辨率完全一致

两者 **conv1→Layer4 的空间分辨率逐层完全相同**（192→96→48→24→12），差别只在：
- 通道宽度 ×4（Layer1 起 64→256 等）
- 更深的 Bottleneck 结构（每层 block 数更多）

**参数 2.1 倍，显存 3.6 倍（实测 622 MB → 2232 MB），推理时间约 1.3 倍（待 E 项实测确认）。**
换句话说：ResNet50 付出的成本主要在"宽"和"深"，而**我们任务需要的空间分辨率（Layer3 = 24×24）两者完全一样**——这是"用 ResNet50 未必划算"的结构层面依据。

---

## 3 本项目中的实际取法（`ddh_v2.py:225-228` 原代码，未改动）

```python
x = b.maxpool(b.relu(b.bn1(b.conv1(x))))   # 384 -> 96
x = b.layer1(x)                            # 96 -> 96  (尺度1)
x = b.layer2(x)                            # 96 -> 48  (尺度2)
c3 = b.layer3(x)                           # 48 -> 24  (尺度3)  ← 唯一进入裁剪头
c4 = b.layer4(c3)                          # 24 -> 12             ← 只做 quality/embedding
aligned = roi_align(self.project(c3), boxes, output_size=(24,16), spatial_scale=1/16)
```

| 尺度 | 分辨率 | 通道(18) | 通道(50) | 本项目用途 |
|---|---:|---:|---:|---|
| Layer1 | 96×96 | 64 | 256 | **基线不用**（融合组才用） |
| Layer2 | 48×48 | 128 | 512 | **基线不用**（融合组才用） |
| Layer3 | 24×24 | 256 | 1024 | **基线使用**（roi_align → 行卷积） |

> 因此任务清单 C 的"仅 Layer3"**就是现有代码的现状**，A 组无需改代码；
> "Layer1+2+3 融合"由 `tools/stage_new/cropnet_v3.py` 的 `fusion='l1l2l3'` 实现
> （三层 1×1 卷积统一到 64 通道 → 上采样到 24×24 → 拼接 → 1×1 融合）。

---

## 4 复现

```bash
# 结构实测（只读，不需要 GPU）
.venv/Scripts/python.exe tools/stage_new/measure_structure.py
# 出图 + 表格
python tools/stage_new/plot_structure.py
```

产物：
- `reports/resnet_structure_raw.json`（完整实测数据）
- `reports/resnet_structure_table.csv` / `.json`（表格，可直接进 PPT）
- `figures/fig1_resnet_structure_compare.png/.pdf`
- `figures/fig2_resolution_channel_tradeoff.png/.pdf`
