# DDH 超声自动裁剪模型（研究项目）

用超声髋关节截图自动给出**上下裁剪建议**，供医生一键确认，减少手工裁剪的工作量。
**研究代码，非临床诊断工具；所有输出必须经人工确认。**

本项目基于 [原方案包](docs/原包README.md)（`docs/原包README.md`，内容与交付时逐字节相同）搭建，
在真实数据上完成了数据接管、标注流水线、窗口定位与两阶段模型训练。

---

## 当前状态（2026-09-21）

| 项目 | 状态 |
| --- | --- |
| 环境 | Python 3.11 + torch 2.5.1+cu121 / torchvision 0.20.1+cu121，CUDA 可用 |
| 原图 | **1080 张**（5 个来源：余 230 / 吴 200 / 张 230 / 胡 157 / 阳 263），另有新批次 874 张待接入 |
| `window` 标签 | **869 / 1080**（1280×960 主布局，人工确认模板 `[368,136,912,744]`），211 张待人工 |
| `crop` 标签 | **1072 / 1080**（由人工裁切成品图反推，100% 定位成功） |
| `safe_box` / `quality` | 0（需临床标注，无法自动化） |
| 训练 | 见 [训练报告](docs/训练报告.md) |
| 代码测试 | 原包 12 项单元测试全部通过 |

## 目录结构

```
ddh.py                     两阶段模型与全部命令（审计/划分/训练/校准/评估/推理）
tools/                     数据与标注流水线脚本（见下）
tests/                     原包单元测试
docs/项目文档             开工手册、成果清单、建模题目、训练报告、原包设计文档
docs/原包README.md         原交付包的 README（逐字节保留）
data/                      ★ 数据目录（**不进仓库**）
runs/                      ★ 训练产物（**不进仓库**）
```

**数据与模型权重不在本仓库中**（医疗影像含患者编号，且体积大）。
本地运行请把数据放到 `data/`，见 [开工手册](docs/开工手册.md) 第 3 节。

## 快速开始

```bash
python -m venv .venv && .venv\Scripts\activate        # Windows
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python tools/status.py                                # 状态面板：当前进度与下一步
python -m unittest discover -s tests -v               # 12 项测试

python ddh.py audit  --manifest data/train_ready.jsonl --root data --output data/reports/grouped.jsonl --report data/reports/pairs.json
python ddh.py split  --manifest data/reports/grouped.jsonl --output data/reports/split.jsonl --reviewed --seed 42
python ddh.py train  --stage window --manifest data/reports/split.jsonl --root data --output runs/window.pt --size 640 --batch 2 --epochs 40 --patience 8 --device cuda
python ddh.py train  --stage crop   --manifest data/reports/split.jsonl --root data --output runs/crop.pt   --size 384 --batch 8 --epochs 80 --patience 12 --device cuda
python ddh.py evaluate --manifest data/reports/split.jsonl --root data --window-checkpoint runs/window.pt --crop-checkpoint runs/crop.pt --output runs/test.json --device cuda
```

## 方法概述

- **步骤 A（窗口检测）**：Faster R-CNN MobileNetV3-FPN，输出成像窗口矩形；要求分数≥0.8 且唯一，否则拒绝。
- **步骤 B（上下边界）**：ResNet18 + 行向空间头 + 有序间隔参数化，输出窗口内的上下归一化边界；
  辅助任务为安全包络与切面可用性质控。
- **拒绝机制**：无窗口 / 多窗口 / 未知界面样式 / 质控不足 / 边界分歧大 / 特征异常 / 校准不足，一律转人工。
- **窗口标签的加速办法**：同一（设备，视口）下窗口位置固定 → 人工确认**每类布局一个模板**，
  再用"框外条带干净"判据自动套用（869/989 通过），不合者自动列出待人工。

## 工具（tools/）

| 脚本 | 用途 |
| --- | --- |
| `status.py` | 状态面板：一条命令看清进度与下一步 |
| `prepare_dataset.py` | 原始图片复制进项目并生成清单骨架 |
| `prefill_window.py` | 逐图自动检测窗口（建议，不作数） |
| `window_templates.py` | 归并窗口模板、列出需人工处理的图 |
| `match_crops.py` | **人工裁切图反推 crop 标签**（FFT 互相关 + 多尺度 + 唯一性判据） |
| `convert_inbox.py` | 批量转裁切图并出每人/每来源质量报告 |
| `unpack_inbox.py` | 解压裁切包/原图包并按来源归位 |
| `build_manifest.py` | 合并已确认成果 → 生成可训练清单 |
| `bench_speed.py` | 实测训练速度与显存 |

## 边界与声明

- 本项目只解决**训练算法、标注协议与可重复验证**，**不包含临床诊断**；
- 仓库不含训练好的权重，也没有经过真实数据验证的准确率承诺；
- 所有推理输出固定 `review_required=true`，程序**不会**自动批准或覆盖原图；
- 无患者标识，近重复分组**不等于**患者级划分；评估只能称为"经相似图组隔离的探索性评估"；
- 使用与分享数据须遵守提供方（医院/老师）的要求，**请勿把数据提交到本仓库**。

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [docs/开工手册.md](docs/开工手册.md) | 环境、数据、标注、完整命令序列、已知的坑 |
| [docs/成果清单.md](docs/成果清单.md) | 所有文件在哪、每步该做什么、报错对照表 |
| [docs/训练报告.md](docs/训练报告.md) | 数据版本、划分、超参、逐轮日志、指标与局限 |
| [docs/建模题目_超声成像窗口自动定位.md](docs/建模题目_超声成像窗口自动定位.md) | 窗口定位问题的形式化题目（含指标与基线） |
| [docs/技术栈与原理.md](docs/技术栈与原理.md) | 每一步用什么技术、背后的原理、为什么不用别的方案 |
| [docs/裁剪规则与DDH基础.md](docs/裁剪规则与DDH基础.md) | 新人上手：DDH 背景、四个标签怎么标、裁剪口径框架 |
| [docs/原包README.md](docs/原包README.md) | 原交付包说明（逐字节保留） |
| [FINAL_DESIGN.md](FINAL_DESIGN.md) | 原方案设计文档（多设备、无患者信息条件下的算法设计） |
| [TEST_REPORT.md](TEST_REPORT.md) | 原包的代码验证记录 |
