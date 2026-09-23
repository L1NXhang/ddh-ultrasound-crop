# 三人协作标注 SOP

> 目标：三个人同时标，标完能直接合并训练，且**口径一致**（不会出现第二个"阳那批"）。

---

## 一、第一次开工前的准备（负责人做，30 分钟）

1. **写口径文档**：填好 `docs/裁切口径.md` 的三条（上边界/下边界/quality 口径）；
2. **标 30–50 张金标准**：作为另外两人的对照基准；
3. **拆队列**：
   ```bat
   python tools\split_queue.py --manifest data\todo_relabel.jsonl --parts 3 --names A,B,C
   ```
   得到 `data/todo_relabel_part_A/B/C.jsonl`，一人一份。

## 二、三台电脑怎么同时标（**其他人不用装任何东西**）

**在一台电脑上跑服务，三个人用浏览器连**（推荐）：

```bat
:: 负责人电脑上，为每个人起一个进程（各自的队列、各自的输出、各自的端口）
python tools\annotate_app.py --manifest data\todo_relabel_part_A.jsonl --root data --reference original --crop-top-offset -70 --top-target 120,200 --out data\human_A.jsonl --port 8765 --host 0.0.0.0
python tools\annotate_app.py --manifest data\todo_relabel_part_B.jsonl --root data --reference original --crop-top-offset -70 --top-target 120,200 --out data\human_B.jsonl --port 8766 --host 0.0.0.0
python tools\annotate_app.py --manifest data\todo_relabel_part_C.jsonl --root data --reference original --crop-top-offset -70 --top-target 120,200 --out data\human_C.jsonl --port 8767 --host 0.0.0.0
```

启动后终端会打印两个地址，例如：
```
浏览器打开  http://127.0.0.1:8765      ← 你这台
浏览器打开  http://192.168.1.23:8765    ← 同一个 WiFi 下的其他电脑用这个
```

另外两人只要在浏览器里打开那个 `192.168.x.x` 地址即可——**不需要装 Python、不需要拷数据、不需要显卡**。
（前提：同一个局域网；负责人电脑保持开机。）

## 三、标注动作（每人每张 5–8 秒）

| 快捷键 | 作用 |
| --- | --- |
| **Enter** | 保存并进下一张（最重要） |
| 拖动绿线 / 蓝线 | 调整上下边界（绿线已按新口径预置，只需微调） |
| **S** | 进 safe_box 模式（**裁剪线自动隐藏**，只看解剖，圈住"绝不能丢"的结构） |
| **W** | 重画窗口框（一般不用动，模型已预填） |
| **1 / 0** | quality：1 = 切面可用，0 = 不可用，拿不准就留空 |
| **↑ ↓** | 当前线微调 1px（Shift 10px） |
| **K** | 跳过（模型/数据有问题时用） |

**界面右侧会实时显示「上边距 = N px」**，落在绿色区间内就说明口径没漂——**每标 20 张扫一眼这个数字**。

## 四、三条必须守的规矩

1. **口径以 `docs/裁切口径.md` 为准**，不是"我觉得"；有疑问先问负责人，不要自己定；
2. **不要改别人已经标过的图**（队列是分开的，不会碰到）；
3. **标完不要自己回头改**旧的，需要改就重标那张（工具会覆盖同 id 记录）。

## 五、收工与合并（负责人做）

```bat
:: 看每人标了多少、口径是否一致（上边距中位数之差应 < 15px）
python tools\merge_labels.py --inputs data\human_A.jsonl data\human_B.jsonl data\human_C.jsonl

:: 确认没问题再落盘
python tools\merge_labels.py --inputs data\human_A.jsonl data\human_B.jsonl data\human_C.jsonl --apply
```

输出会显示每人的"上边距中位"和"标准差"。**中位数之差超过 15px 就先别训练**，先对齐口径。

## 六、常见问题

**Q：断网/关机了会丢吗？**
A：不会。每按一次 Enter 就已经写进负责人的本地文件；重开时队列会自动跳过已完成的图。

**Q：两个人标到同一张怎么办？**
A：队列是拆开的，不会。万一发生，合并时后写的覆盖先写的，并在报告里提示。

**Q：浏览器打开是空白？**
A：检查是不是同一个 WiFi（手机热点也行）；确认负责人那边的窗口没关。

**Q：能不能在自己电脑上单独跑？**
A：可以（把项目拷过去、装 Python + Pillow 即可，`--no-model` 模式不需要 torch），
但没必要——浏览器方案零配置。
