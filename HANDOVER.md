# 交接说明（HANDOVER）

> 面向接手本项目的同学。最后更新：2026-11-07
> **先读这一页，再看 `README.md` 和 `FINAL_DESIGN.md`。**

---

## 一、这个项目在做什么（一句话）

婴儿髋关节超声（DDH）的**前置环节**：把设备导出的整屏截图，自动裁成**纯成像区**，
为后续 Graf 法测角/分型提供干净、口径一致的输入。**系统不输出任何诊断结论**。

两阶段架构：

| 阶段 | 做什么 | 技术 |
|---|---|---|
| Stage A | 定位**成像窗口**（四边） | 纯规则几何，**无需训练数据**（`tools/detect_window.py`） |
| Stage B | 窗口内回归**上/下裁剪线** | CropNet = ResNet-18@384 + ROIAlign + 一维行卷积 + 累积 softmax |

## 二、当前结论（最重要的一条）

**在 560 张训练图的规模下，换主干/加深层/多层级融合都没有产生统计显著的精度提升；
真正的杠杆是标注口径与数据量。**

- 6 组配置对照（test 88）：`r18_l3` 上边界误差**均值 17.39 px / 中位数 11.2 px**，纵向 IoU 0.9096。
- 标注口径统一带来**中位数 89 → 17.5 px（↓80%）**，远大于任何结构改动。
- **不要**把"ResNet-50 更差"写进任何材料——差异**不显著**（p=0.0805），只能说成本确定更高
  （参数 2.09×、CPU 约 2.5×）。
- 6 组对比全部是**单一种子、88 张比较集**。所有"不显著"都应表述为
  **"当前样本下未检出显著差异"**，而不是"已证明无效"。

## 三、目录导航

```
tools/stage_new/          ← 本轮全部实验代码（重点看这里）
  cropnet_v3.py           6 种配置定义（r18_l3 / r50_l3 / r18_l123 / r50_l123 / r18_l4 / r18_l1234）
  train_eval_v3.py        统一训练/评估入口（同条件对照的唯一实现）
  stats_test.py           配对 Wilcoxon / bootstrap / McNemar
  measure_structure.py    参数量、感受野、各层形状实测
  dp_seam_baseline.py     DP/图搜索基线（对照"学习式方法"是否有必要）
  plot_*.py               出图脚本
  acceptance_check.py / verify_*.py  验收与数据范围核查
scripts/                  ← 复现入口
  reproduce_all.sh        ★ 从头复现全部结果，先看这个
  run_four_models*.sh     四模型对照
  run_layer_ablation.sh   层级消融
  stage_new_reports/      实验报告素材（注意：其中部分数字已过时，见第六节）
docs/开发文档_第2版/       ← 交付文档源码（main.tex，12 页 A4）
docs/*.md                 ← 各类分析与标准文档
```

## 四、怎么跑起来

```bash
# 1) 环境
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows

# 2) 数据（★ 不在仓库里，需另取，见第五节）
#    需要 data/images_*（原图）与 data/reports_v2/split.jsonl（862 行划分）

# 3) 复现主实验
bash scripts/reproduce_all.sh

# 4) 主结果位置
runs/20261103_resnet18_vs_50_corrected/resnet18/r18_l3/metrics.json
```

单配置训练：

```bash
.venv/Scripts/python tools/stage_new/train_eval_v3.py \
  --config r18_l3 --epochs 30 --manifest data/reports_v2/split.jsonl \
  --out runs/<新目录>/resnet18
```

固化超参（已核实，可直接写进材料）：`epochs 30 / patience 6 / batch 8 / size 384 / seed 42 /
AdamW (backbone lr 3e-5, head lr 3e-4, wd 1e-4) / CosineAnnealingLR / ImageNet 预训练`。

## 五、不在仓库里的东西（重要）

`.gitignore` 排除了以下内容，**接手时需要向导师/项目负责人索取**：

| 被排除 | 内容 | 为什么 |
|---|---|---|
| `data/` | 原始超声图像、`split.jsonl`(862)、`labels_v2.jsonl` | 患者影像，合规要求不外传 |
| `runs/`、`runs_v2/` | 全部实验输出、指标、图表数据 | 体积大，可由脚本重建 |
| `*.pt` / `*.pth` / `*.onnx` | 模型权重 | 体积 + 可由训练重建 |

**如果你只拿到仓库：** 代码可运行但**无数据无权重**，无法直接复现数字。

## 六、已知问题 / 接手后该优先做的

1. **`scripts/stage_new_reports/` 里的部分数字已过时**（写于早期），以 `docs/开发文档_第2版/`
   与 `runs/20261103_resnet18_vs_50_corrected/` 为准。
2. **多个脚本硬编码旧 `runs/20261103_resnet18_vs_50` 路径**（应为 `_corrected`）：
   `measure_structure.py`、`plot_structure.py`、`acceptance_check.py`、
   `verify_eval_scope.py`、`verify_labels.py`。运行时会找不到目录或读到旧数。
3. **`runs/20261118_vs_50/` 是无效遗留目录**，不要引用其中任何数字。
4. **感受野曾有 bug（已修）**：`measure_structure.py` 曾把残差支路 `downsample.0`
   （1×1, stride 2）当主干卷积，使 L2–L4 感受野虚高 4 倍（131/483/1891 → 正确 **99/211/435**）。
   **由此"Layer3 已具备全局语义"的说法不成立**（211 < 384），文档已改写。
5. **尚未做的验证**：裁剪对 Graf α/β **测角误差**的影响未测；标注者间一致性（ICC）未算。

## 七、几条不能违反的规矩

- **严禁编造实验数据**。任何数字都要能指回 `runs/` 下的文件。
- **不删除旧文件**。作废的目录用 `INVALID_请勿使用.md` 标记，不要 rm。
- 说是 **"Graf 法"**（Reinhard Graf），不是 "Graph 法"。
- 文献检索的否定性结论写作 **"我们未检索到同类系统"**，不写"世界上不存在"。
- 不显著 ≠ 证明无差异；案例展示必须标注"案例展示 · 筛选口径 · 非总体性能结论"。
