# DDH多设备超声自动裁剪研究包

先读《DDH超声自动裁剪模型最终方案.docx》或 FINAL_DESIGN.md。

本包可以开始标注、训练和验证，但没有训练好的 DDH 权重、真实准确率或临床安全保证。所有结果需人工确认。没有患者信息时，近重复分组不是患者级划分。

## 1 安装

建议 Python 3.11，新建独立虚拟环境。下面示例为 CPU；GPU 请用 PyTorch 官方历史版本安装说明选择与驱动兼容的 2.5.1 轮子：https://pytorch.org/get-started/previous-versions/

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Windows 用 `.venv\Scripts\activate` 激活。ML 代码只依赖 requirements.txt；不需要 Word、网页框架或数据库。第一次正式训练需要联网下载 COCO 与 ImageNet 权重，`--no-pretrained` 只供代码测试，正式训练不要无理由禁用预训练。

## 2 准备数据

将图片放入 data/images，复制 annotations.example.jsonl 为 data/annotations.jsonl，并替换示例为真实标签。示例不是可训练数据。path 相对于 `--root`；window、crop、safe_box 全部是原截图像素，不是归一化坐标。

一行示例：

```json
{"id":"img001","path":"images/img001.png","window":[100,60,600,560],"crop":[140,490],"safe_box":[150,160,550,470],"quality":1,"style_group":"layout_01","source_group":"","leak_group":"","split":""}
```

未知标签写 null 或空字段，不写伪造值。非标准切面 quality=0，crop 可以为 null。safe_box 必须由老师定义并独立标注，不能简单复制 crop 冒充解剖标注。标注时不要旋转图片或改尺寸。

## 3 审核近重复并冻结分组

```bash
python ddh.py audit --manifest data/annotations.jsonl --root data --output data/grouped.jsonl --report data/pairs.json
```

检查 pairs.json 和分组，必要时人工修改 grouped.jsonl 中的 leak_group。确认真实同序列图不会跨组，注意大组的误合并。确认后再执行：

```bash
python ddh.py split --manifest data/grouped.jsonl --output data/split.jsonl --reviewed --seed 42
```

只有确实审核后才传 --reviewed。至少需要 10 个非留出组；这是程序下限，不是足够统计效力。检查划分后各组质量标签的数量，val 必须含正负质控标签；若要分层调整，仅依据元数据在训练前完成并冻结，不用模型成绩挑划分。

可选外观域外测：增加 `--holdout-style layout_03`。这不能证明跨医院泛化，更不能替代患者级外测。

## 4 训练步骤A和步骤B

以下命令假定有可用 CUDA；CPU 调试改为 `--device cpu`。train 不访问 test。

```bash
python ddh.py train --stage window --manifest data/split.jsonl --root data --output runs/window.pt --size 640 --batch 2 --epochs 40 --patience 8 --device cuda
python ddh.py train --stage crop --manifest data/split.jsonl --root data --output runs/crop.pt --size 384 --batch 8 --epochs 80 --patience 12 --device cuda
```

全局池化基线：

```bash
python ddh.py train --stage crop --baseline --manifest data/split.jsonl --root data --output runs/baseline.pt --size 384 --batch 8 --epochs 80 --device cuda
```

保存文件是最佳模型参数与元数据，旁边生成 history.json。未实现中断恢复；不能把保存的权重文件当成含优化器状态的 resume 检查点。训练默认 FP32、单进程读取，不依赖多卡。

## 5 校准与固定策略核查

```bash
python ddh.py calibrate --manifest data/split.jsonl --root data --window-checkpoint runs/window.pt --crop-checkpoint runs/crop.pt --output runs/calibration.json --device cuda --samples 16 --margin 0.02
```

这一步在 val 拟合温度和不确定性阈值，在独立 calib 核查已冻结规则。缺少正负质控标签、安全标注或足够合格组时，文件可能 enabled=false；这是正常的保守结果，不能随意改成 true。若验证集没有任何成功检测且有安全真值的正例，则直接报错。

当 calib 不足时，先补标或继续人工复核。没有患者信息使统计上界只是名义值。即使 enabled=true，预测仍然要求人工确认。校准文件绑定两个检查点的 SHA256，换模型必须重做校准。

## 6 评估

端到端评估：

```bash
python ddh.py evaluate --manifest data/split.jsonl --root data --window-checkpoint runs/window.pt --crop-checkpoint runs/crop.pt --calibration runs/calibration.json --output runs/test.json --device cuda
```

仅评估步骤 B 并使用人工真窗口：

```bash
python ddh.py evaluate --manifest data/split.jsonl --root data --window-checkpoint runs/window.pt --crop-checkpoint runs/crop.pt --oracle-window --output runs/test_oracle.json --device cuda
```

oracle 运行禁止带端到端校准文件，避免错误复用门控评估。为隔离 A 的影响，应比较两次均不加 calibration 的 raw 候选结果；带校准结果另报。没有 calibration 时所有图均要求人工处理，候选边距为零。

所有输出都注明有效分母。safe 是完整包络保留，critical_cut 是已生成候选中的误裁；human_modified 只有真实人工记录才有值。edit_proxy_rate 是容差代理，不能报告成真实人工修改率。

## 7 单图推理与候选保存

```bash
python ddh.py predict --image data/images/img001.png --style layout_01 --window-checkpoint runs/window.pt --crop-checkpoint runs/crop.pt --calibration runs/calibration.json --output runs/img001_prediction.json --preview runs/img001_candidate.png --device cuda
```

--style 必须是已核实的界面样式，未知就传 unknown，不能为了通过筛查乱填。JSON 给出原图 crop_box、分歧、拒绝理由、review_required 和 eligible_for_quick_confirmation。

--preview 只输出“未批准候选图”。无合格窗口时不保存候选。请先人工看原图和候选；本包不实现点击确认界面，也不把预览自动加入训练数据。原图永不覆盖。

## 8 自动测试

```bash
python -m unittest discover -s tests -v
python tests/smoke_train.py --workdir /absolute/path/to/new/smoke_output
```

smoke_output 必须不存在，避免覆盖旧文件。合成烟雾测试使用随机纹理，无预训练，只运行 1 epoch；产生的权重不能用于实际裁剪。测试清单与实际结果见 TEST_REPORT.md。

## 9 实现边界

代码没有声称完成全部科学实验：自动主动学习选择、Graf 分专家、裁剪目标检测对照、bootstrap 统计、非矩形视场分割未实现。完整实验计划见方案文档。多医院真实外测和患者级独立验证，现有信息不足无法完成。

可复现建议：在报告里保存 split.jsonl、训练日志、软件环境、模型 SHA256、校准 JSON、逐图评估 JSON 和专家纠正记录。冻结测试集不要回流训练。
