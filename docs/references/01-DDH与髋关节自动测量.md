# 01 DDH 与髋关节自动测量

> 检索日期：2026-10-22
> 检索库：PubMed（E-utilities EFetch 取摘要）、OpenAlex、Crossref（核对 DOI/作者/卷期页码）、PMC 全文（开放获取的逐篇精读）
> 收录原则：**只收元数据（DOI + 标题 + 期刊 + 年份 + 作者）已由至少两个独立来源核对过的文献**；只有元数据、没读到摘要的，在「一、文献清单」备注列标「仅元数据」，并在逐篇要点里明确写"摘要未取得"，绝不据标题推测方法细节。
> 与项目的关系：本项目 = Graf 法婴儿髋关节超声**自动裁剪**（规则算法定位成像窗口 window → ResNet18 回归上/下裁剪边界线），痛点是**上边界预测不准**。

---

## 一、文献清单

| 编号 | 标题 | 年份 | 期刊/会议 | 核心方法 | 对我们的价值 | DOI/链接 |
|---|---|---|---|---|---|---|
| 01-01 | Fully Automating Graf's Method for DDH Diagnosis Using Deep Convolutional Neural Networks | 2016 | MICCAI 2016 Workshop（LNCS） | 用 DCNN 全自动复现 Graf 法 | 最早把"Graf 法全流程自动化"拆成子任务的论文，是路线源头（**仅元数据**） | [10.1007/978-3-319-46976-8_14](https://doi.org/10.1007/978-3-319-46976-8_14) |
| 01-02 | Accuracy of New Deep Learning Model-Based Segmentation and Key-Point Multi-Detection Method for Ultrasonographic DDH Screening | 2021 | Diagnostics | Mask R-CNN 分割 4 个解剖结构 → 二阶段训练输出关键点 → check angle ±5° 判平面 | **评价口径 + 可用性门槛**的关键参照；"先分割再取点"与我们的两阶段同构 | [10.3390/diagnostics11071174](https://doi.org/10.3390/diagnostics11071174)（PMC8303134） |
| 01-03 | Joint Landmark and Structure Learning for Automatic Evaluation of DDH | 2022 | IEEE J BHI | Mask R-CNN + landmark 分支多任务，shape similarity loss，landmark-structure 一致性先验 | 用**结构分割反过来约束关键点**——直接可迁移到"上边界被骨影干扰"的场景 | [10.1109/JBHI.2021.3087494](https://doi.org/10.1109/JBHI.2021.3087494)（arXiv:2106.05458） |
| 01-04 | Deep learning-based automated guide for defining a standard imaging plane for DDH screening using US | 2025 | BMC Med Inform Decis Mak | global(检测 ROI 框)+local(框内回归 5 点) 两阶段；EfficientNetB2/ResNet50/YOLOv8 对比 | **与本项目两阶段架构几乎逐项对应**（检测框 → 框内回归），并给出"平面是否标准"的角度化定义 | [10.1186/s12911-025-02926-8](https://doi.org/10.1186/s12911-025-02926-8)（PMC11837398） |
| 01-05 | Diagnosis of DDH by Ultrasound Imaging Using Deep Learning | 2023 | J Pediatr Orthop | CNN 分类/诊断 | Graf 角度分类基线（**仅元数据**） | [10.1097/BPO.0000000000002428](https://doi.org/10.1097/BPO.0000000000002428) |
| 01-06 | The Diagnosis of DDH From Hip Ultrasonography Images With Deep Learning Methods | 2023 | J Pediatr Orthop | 深度学习角度/分类 | 非开放获取，仅作背景（**仅元数据**） | [10.1097/BPO.0000000000002294](https://doi.org/10.1097/BPO.0000000000002294) |
| 01-07 | Rapid Diagnosis and…（**DOI 已撤回使用**，见 01-07 正文说明） | — | Journal of Microscopy and Ultrastructure (JMAU) | — | **未核实成功**：用该 DOI 检索得到的是无关的鼻窦真菌论文，本轮**不采纳** | 见「四、未确认/存疑」A6 |
| 01-08 | Artificial Intelligence Algorithm Supporting the Diagnosis of DDH: Automated Ultrasound Image Segmentation | 2025 | J Clin Med | SegNeXt (MSCAN_L) 分割 8 个结构；点式 landmark 用欧氏距离评价 | **给出了"点误差（像素）+ 基线偏转角（度）"双指标**，是我们评价上边界的最好口径 | [10.3390/jcm14176332](https://doi.org/10.3390/jcm14176332) |
| 01-09 | DDH: ultrasound evaluation | 2025 | Pediatr Radiol | 综述（Graf 法超声读片） | 临床定义与标准切面判据的权威教科书式来源（**摘要未取得**） | [10.1007/s00247-025-06334-y](https://doi.org/10.1007/s00247-025-06334-y) |
| 01-10 | Game-Changing Advances in Infant Hip Dysplasia Evaluation With Ultrasound | 2025 | J Ultrasound Med | 综述：3D US、AI 辅助 2D 扫查、开源软件/数据集 | **与本项目场景最贴近的综述**：明确主张"AI 2D 扫查 + 自动质量控制"路线 | [10.1002/jum.70161](https://doi.org/10.1002/jum.70161) |
| 01-11 | Artificial Intelligence to Detect DDH: A Systematic Review | 2025 | J Paediatr Child Health | 系统评价深度学习诊断 DDH 准确度 | 定量对标基线、写作引文锚点（范围含 X 光，非纯超声） | [10.1111/jpc.70172](https://doi.org/10.1111/jpc.70172) |
| 01-12 | Artificial intelligence in the diagnostic imaging of DDH: a systematic review | 2026 | EFORT Open Reviews | PRISMA 系统评价（32 篇初筛，19 篇纳入） | 目前最全的 AI×DDH 文献地图，用于证明"我们的问题确实没人正面做" | [10.1530/EOR-2024-0165](https://doi.org/10.1530/EOR-2024-0165) |
| 01-13 | Deep learning algorithms for identifying DDH based on sonographic images: retrospective, prospective, multicenter study in China (HipSonoNeuNet) | 2025 | eClinicalMedicine | 22 家医院、7286 张图的诊断模型；三阶段人机对比 | 中国多中心**泛化性**证据；说明"AI 辅助提升初级医生"的评价口径 | [10.1016/j.eclinm.2025.103552](https://doi.org/10.1016/j.eclinm.2025.103552)（PMC12547208） |
| 01-14 | Multihipposenet: multi-task deep learning for automated DDH assessment in infant hip ultrasound | 2026 | Front Pediatr | ResNet-50+FPN 多任务（分割/关键点/角度） | **最新 SOTA 数值（α 97.97%、β 96.81% 落在 ±5° 内）**，作为性能上界参照 | [10.3389/fped.2026.1939157](https://doi.org/10.3389/fped.2026.1939157) |
| 01-15 | Artificial Intelligence-Assisted Wireless Handheld Ultrasound for Screening DDH in Infants (AI-HUDs) | 2026 | J Ultrasound Med | YOLOv8 检测 6 个关键点 → 测 α/β | 证明**"YOLO 直接出点"在弱设备/新手操作下也能用**，是我们检测器的可选升级 | [10.1002/jum.70318](https://doi.org/10.1002/jum.70318) |
| 01-16 | Artificial Intelligence-Assisted Standard Plane Detection in Hip Ultrasound (AI-SPS) | 2025 | J Orthop Res | SSD-MobileNetV2 vs YOLOv11n 检测"标准帧 / 非标准帧" | 把"标准切面"当**检测任务**做，精度 86.3%——提醒"标准切面"本身是软标签 | [10.1002/jor.70020](https://doi.org/10.1002/jor.70020) |
| 01-17 | Detection of standard plane from ultrasound scans by deep learning methods for DDH diagnosis | 2022 | Mühendislik Bilimleri ve Tasarım Dergisi | 深度网络做标准平面检测 | 土耳其 OA 期刊，低成本可复现的对照实验（**仅元数据**） | [10.21923/jesd.1064904](https://doi.org/10.21923/jesd.1064904) |
| 01-18 | Toward automatic diagnosis of hip dysplasia from 2D ultrasound | 2017 | IEEE ISBI | 早期 2D US 自动 DDH 诊断 | 历史基线（**仅元数据**） | [10.1109/ISBI.2017.7950680](https://doi.org/10.1109/ISBI.2017.7950680) |
| 01-19 | Diagnosis on Ultrasound Images for DDH with a Deep Learning-Based Model Focusing on Signal Heterogeneity in the Bone Region | 2025 | Diagnostics | HigherHRNet-W48 做顶点估计 + 髂骨信号异质性分析 + 2D→3D 变换 | **最接近"上边界为什么难"的一篇**：明确把"髂骨高亮度区域 + 旋转"作为模型输入 | [10.3390/diagnostics15040403](https://doi.org/10.3390/diagnostics15040403)（PMC11854700） |
| 01-20 | Automated Neonatal Hip Ultrasound System for Diagnosing DDH Using Assistive AI | 2025/2026 | J Imaging Inform Med | NASNetMobile/MobileNetV1/DenseNet121/EfficientNetV2B0/ResNet50 分类 + UnestED/DeepLabV3Plus 分割；Graf 角度误差 0.21 | 提供"分类主干 + 分割主干"的逐模型对比表，可直接抄基线选型 | [10.1007/s10278-025-01498-3](https://doi.org/10.1007/s10278-025-01498-3)（PMC12920978） |
| 01-21 | A Deep Learning–Based Framework for Automatic Determination of DDH from Graf Angles | 2025 | J Imaging Inform Med | 从 Graf 角度出发的自动判定框架 | 与"测量口径→诊断"链条对齐（**仅元数据**） | [10.1007/s10278-025-01518-2](https://doi.org/10.1007/s10278-025-01518-2) |
| 01-22 | Can AI Automatically Assess Scan Quality of Hip Ultrasound? | 2022 | Applied Sciences | CNN 把 3D US 髋扫描分「adequate / inadequate」 | **可用性判断的量化范例**（有两套专家标注集，准确率 96%/91%） | [10.3390/app12084072](https://doi.org/10.3390/app12084072) |
| 01-23 | Real Time RNN Based 3D Ultrasound Scan Adequacy for DDH | 2018 | MICCAI 2018（LNCS） | 3D US 扫描充分性实时判断 | "可用性"从后处理搬到实时（**仅元数据**） | [10.1007/978-3-030-00928-1_42](https://doi.org/10.1007/978-3-030-00928-1_42) |
| 01-24 | Comparison of quality and interpretation of newborn US screening for DDH by basically trained nurses and junior physicians | 2024 | PLOS ONE | 6 护士 vs 6 初级医生各 400 张标准图，双瑞士督导定金标准 | **给了"常见错误清单：缺下缘、测量线画错、技术问题"**——正是我们上/下边界失败的临床对应 | [10.1371/journal.pone.0300753](https://doi.org/10.1371/journal.pone.0300753) |
| 01-25 | SegFormer3D: Improving the Robustness of DL Segmentation in US Volumes of the Pediatric Hip | 2025 | Ultrasound Med Biol | 3D US 骨分割鲁棒性系统评估 + SegFormer3D | **"分辨率/FOV 小变化就让模型崩"**的量化证据，直接解释我们的边界漂移 | [10.1016/j.ultrasmedbio.2024.12.013](https://doi.org/10.1016/j.ultrasmedbio.2024.12.013) |
| 01-26 | Morphological Development of the Hip in Normal Infants Under Six Months of Age by the Graf US Method | 2022 | Front Pediatr | 3067 名婴儿 6134 髋的 α/β/FHC 参考值 | 先验：正常婴儿 α≈62–63°、β≈55–56°，用于 sanity check 和窗口内尺度先验 | [10.3389/fped.2022.914545](https://doi.org/10.3389/fped.2022.914545) |
| 01-27 | A method for early ultrasound classification of DDH in infants: combining YOLOv8-based keypoint detection with radiomics | 2026 | BMC Pediatrics | YOLOv8 关键点 + 影像组学 | 中国团队，YOLO 关键点路线的又一实例 | [10.1186/s12887-026-07150-5](https://doi.org/10.1186/s12887-026-07150-5) |
| 01-28 | Automated Diagnosis of DDH in Ultrasound Using an End-to-End Deep Learning Approach | 2026 | Research Square（**预印本**） | 端到端深度学习 | 只看作预印本，不作证据 | [10.21203/rs.3.rs-9439534/v1](https://doi.org/10.21203/rs.3.rs-9439534/v1) |
| 01-29 | Deep Convolutional Neural Network-Based Automatic Classification of Neonatal Hip Ultrasound Images: A Novel Data Augmentation Approach with Speckle Noise Reduction | 2019/2020 | Ultrasound Med Biol |  speckle 噪声抑制 + 数据增强分类 | 超声散斑是我们上边界模糊的物理根源之一（**摘要未取得**） | [10.1016/j.ultrasmedbio.2019.09.018](https://doi.org/10.1016/j.ultrasmedbio.2019.09.018) |
| 01-30 | Shadow Peak: Accurate Real-time Bone Segmentation for Ultrasound and DDH | 2019 | EPiC Series in Health Sciences | 基于声影的实时光骨分割 | 若能把"骨声影起点"变成上边界特征，值得一试（**仅元数据**） | [10.29007/w172](https://doi.org/10.29007/w172) |

> 说明：编号 01-07 是我在早期检索中因 DOI 误配而**主动作废**的条目，保留占位是为了避免后续重复踩坑，正文见「四、未确认/存疑」A6。表格行 30 行，去掉作废的 01-07 后**有效条目 29 篇**。

---

## 二、逐篇要点

### [01-01] Fully Automating Graf's Method for DDH Diagnosis Using Deep Convolutional Neural Networks
- **做了什么**：Golan D, Donner Y, Mansi C, Jaremko J, Ramachandran M（2016，MICCAI 2016 Workshop "Deep Learning and Data Labeling for Medical Applications"，LNCS，pp.130–141）。据同刊引用描述（见 [01-04] 参考文献 [10]、[01-02] 参考文献 [21][22]），把 Graf 法全流程交给 CNN 自动化。
- **方法**：**本轮未取得摘要/正文**（Springer 付费，OpenAlex 无 abstract），不据标题展开。
- **结果**：未取得。
- **对我们的价值**：只作"路线源头"引用——所有后续两阶段 / 分割+关键点工作都把它当参照。**引用时只写"最早提出全自动 Graf 法的深度学习框架"，不要描述其具体网络结构。**
- **局限**：无开放全文，且 2016 年的标注规模与现代工作不可比。

### [01-02] Accuracy of New Deep Learning Model-Based Segmentation and Key-Point Multi-Detection Method for Ultrasonographic DDH Screening（Lee SW et al., Diagnostics 2021;11(7):1174）
- **做了什么**：单中心 1243 张髋超声（168 名婴儿），Mask R-CNN 分割髂骨/髋臼/盂唇等结构，再用第二阶段从分割图上取点；用"check angle"（髂骨翼与基线的夹角）判断图像是否可用，只有 ±5° 内的图才进入 α/β 评价。
- **方法**：两阶段训练（先区域分割，转黑白掩膜，再在其上标 4 个点）；数据增强含 ±15° 旋转、75–100% 亮度、高斯噪声；训练 40k 步。
- **结果**：921 张测试图中仅 320 张（34.7%）被医生与 AI 共同判为"可用"；check angle 一致率 80.9%（κ=0.525）；α 角 ICC=0.764，β 角 ICC=0.743；α 平均绝对偏差 3.47°，β 4.50°；74.7% 的图 α 角人机差 <5°；正常/异常二分类一致率 84.3%。
- **对我们的价值**：**评价口径**——"可用性判断（两阶段门控）+ 角度误差 ICC/一致性"这套做法可以直接照搬成我们的 evaluation contract；也侧面说明**真实数据里超过一半的图"不达标"**，我们的裁剪工具必须能对"不该裁的图"给出拒绝或告警，而不是硬裁。
- **局限**：单中心、单一机型（Philips HD15/HD7 XE）、无外部验证；作者自述样本来源少。另有 Comment（Diagnostics 2022;12:1738）对其方法提出质疑，引用时建议一并提及。

### [01-03] Joint Landmark and Structure Learning for Automatic Evaluation of DDH（Hu X. et al., IEEE JBHI 2022;26(1):345-358）
- **做了什么**：南京医科大学 / 深圳大学团队。Mask R-CNN 为基座，加一条 landmark 检测分支，构成多任务框架；引入 **shape similarity loss** 修正不完整的解剖结构预测；再引入 **landmark-structure 一致性先验**，保证"从分割结构估计出的 bony rim"与"直接检测出的 landmark"一致。
- **方法**：多任务回归 α/β 角；1231 张图（632 例），247 张（126 例）测试。
- **结果**：α 角平均误差 2.221°，β 角 2.899°；α 有 93%、β 有 85% 的估计误差 <5°。
- **对我们的价值**：**直接命中上边界**。我们的上边界本质是"骨性轮廓与软组织交界"，单靠回归头容易被骨影/声影带偏；这篇的做法是**先分割出结构，再用结构一致性约束点**。落地方式：在 ResNet18 回归之外并联一个轻量分割头（髂骨高亮区 / 骨-软骨边界），用分割边缘作为上边界的强先验做后处理修正（例如把回归线吸附到分割边界梯度最大处）。
- **局限**：付费期刊（有 arXiv 预印本 2106.05458 可读）；2D 单帧，未处理弱质图。

### [01-04] Deep learning-based automated guide for defining a standard imaging plane for DDH screening（Ahn KS, Choi JH, … Cho Y, Jang WY; BMC Med Inform Decis Mak 2025;25:91）
- **做了什么**：韩国高丽大学安岩医院，2823 张冠状面髋超声（532 例）。定义 5 个 landmark（髂骨下缘、髂骨中点、髋臼骨缘、三叉软骨、盂唇末端），前 3 点决定平面是否标准：∠(点1-点2) 与 ∠(点2-点3) 各自与水平线夹角都 <5° 即视为"髂骨线直、平面可用"；后 2 点用于算 α/β。
- **方法**：**global network + local network 两阶段**：global 网络（ResNet50 / EfficientNetB2 / YOLOv8 三选一）先在 256×256 图上预测 32×32 的 ROI 框（IoU loss），local 网络再在 32×32 patch 上回归点坐标（RMSE loss）。几何增强 zoom/rotation/shift，z-score 归一化。
- **结果**：整合后 RMSE 总体 4.02±3.72 像素（EfficientNetB2）；逐点 RMSE 从第 1 点 1.69±1.26 递增到第 4 点 5.92±4.25、第 5 点 6.61±4.82；mAP 0.67/0.74/0.83。**结论很关键：越靠外周/越模糊的点越不准**。
- **对我们的价值**：**window 定位 + 上边界 + 评价口径三处都直接可用**。(1) 它的 "global 出框 → local 出点" 就是我们 "检测器出 window → ResNet18 出边界线" 的完全同构实现，可作为架构正当性的最强引用；(2) 它量化了"误差随点序递增"，可作为我们"上边界注定比下边界难"的文献支撑；(3) 平面可用性用**角度阈值 5°** 表达，我们在上边界可用性上也可以给出类似的软阈值。
- **局限**：单中心；作者承认数据里前/后切面样本少（临床多在中线位采集）；髂骨与探头夹角未进模型。

### [01-05] Diagnosis of DDH by Ultrasound Imaging Using Deep Learning（J Pediatr Orthop 2023）
- **做了什么**：CNN 用于髋超声 DDH 诊断。
- **方法/结果**：**摘要未取得**（付费），仅元数据经 Crossref/OpenAlex 核对。
- **对我们的价值**：仅作"分类路线"背景引文，不描述其细节。
- **局限**：非开放获取。

### [01-06] The Diagnosis of DDH From Hip Ultrasonography Images With Deep Learning Methods（Atalar H et al., J Pediatr Orthop 2023;43(2):e132-e137）
- **做了什么**：土耳其团队，深度学习做髋超声诊断。
- **方法/结果**：**摘要未取得**（付费），元数据已核对（含 Volume/Issue/页码）。
- **对我们的价值**：仅背景引文。
- **局限**：非开放获取。

### [01-07] （占位，作废）——见「四、未确认/存疑」A6
- 原计划收录一篇印度团队（JMAU）的 "Landmark-driven…" 论文，但按 DOI `10.4103/jmau.jmau_63_21` 检索，OpenAlex/Crossref 返回的是**完全无关的鼻窦真菌论文**（Bapodra MK et al.）。判定为该 DOI 与标题不匹配，**本轮不收录**，此编号留空。

### [01-08] Artificial Intelligence Algorithm Supporting the Diagnosis of DDH: Automated Ultrasound Image Segmentation（Pulik Ł et al., J Clin Med 2025;14(17):6332）
- **做了什么**：波兰团队，10767 张髋超声（311 例），按 Graf 法标注 **8 个结构**；训练/验证/测试 75/9.5/15.5%。
- **方法**：对比多种分割架构，最优为 **SegNeXt (MSCAN_L backbone)**；对"点式 landmark"（bony rim、lower limb）额外报告欧氏距离，对"基线"额外报告偏转角。
- **结果**：IoU/DSC：软骨-骨边界 0.632/0.774，股骨头 0.916/0.956，盂唇 0.625/0.769，软骨顶 0.672/0.804，骨性顶 0.725/0.841；bony rim 点平均欧氏距离 **4.8 px**，lower limb **4.5 px**，基线偏转角 **1.7°**。
- **对我们的价值**：**评价口径**。它把"点误差（像素）"和"角度误差（度）"分开报，这正是我们上边界该用的双指标：像素误差说明"线画偏多少"，角度误差说明"偏多少才影响下游 α 角"。另外它的数据量（1 万张）说明分割路线在髋超声上已经足够成熟，可作为我们并联分割头的风险背书。
- **局限**：结构类分割仍然是弱项（软骨-骨边界 IoU 仅 0.63），恰恰是我们上边界所在位置——说明这条边界在数据层面本身就模糊。

### [01-09] Developmental dysplasia of the hip: ultrasound evaluation（Krauss J, Straus Takahashi M, Samet J; Pediatr Radiol 2025）
- **做了什么**：综述，系统性讲 Graf 法髋超声的评估方法。
- **方法/结果**：**摘要未取得**（OpenAlex/PubMed 均无 abstract）。
- **对我们的价值**：作为"Graf 标准切面定义"的**权威临床出处**备选；引用前需实际下载原文核对，不能凭标题写内容。
- **局限**：付费；内容未核实。

### [01-10] Game-Changing Advances in Infant Hip Dysplasia Evaluation With Ultrasound（Fadell M, … Jaremko JL; J Ultrasound Med 2025;45(6):1129-1137）
- **做了什么**：综述，三条线：(1) 3D US 体积成像 + 回溯选平面；(2) **AI 辅助的 2D cine sweep，为"轻度训练的操作者"提供自动质量控制与分类**；(3) 开源软件与数据集，支持外部验证与标准化。提出务实路径：**普筛用 AI 辅助的短 2D 扫查，异常/不确定者转 3D US 确认**。
- **方法/结果**：综述，无原始数值。
- **对我们的价值**：**最贴合本项目定位的综述**。我们的工具正属于"AI 辅助 2D 图的自动质量控制"，这篇给了行业级别的问题定义与正当性；"cine sweep + 自动质控"也暗示：如果我们的 window 定位在**帧序列**上跑，稳定性会比单帧大幅提升（与本文 [02 主题] 的窗口定位思路互补）。
- **局限**：观点性综述，无定量证据；未涉及"裁剪线"这一具体输出形态。

### [01-11] Artificial Intelligence to Detect DDH: A Systematic Review（J Paediatr Child Health 2025）
- **做了什么**：检索 PubMed/Medline/EMBASE/EMCARE/ClinicalTrials/IEEE Xplore/Cochrane（2024-10），按 Cochrane DTA 工作组指南做系统评价，评估深度学习在**髋超声与 X 光**上的诊断准确度与偏倚风险。
- **方法/结果**：摘要只交代方法与范围，未在可获取部分给出合并效应量。
- **对我们的价值**：写论文/报告时的"AI×DDH 已有多少证据"锚点；注意它**混合了 X 光**，不能当作纯超声证据。
- **局限**：范围混杂；风险偏倚结果需读全文。

### [01-12] Artificial intelligence in the diagnostic imaging of DDH: a systematic review（Kamath AG et al., EFORT Open Rev 2026）
- **做了什么**：PRISMA 系统评价，Prospero 注册 CRD42024563606，检索 Ovid MEDLINE / PubMed / Embase / CENTRAL / CDSR；初筛 32 篇，**纳入 19 篇，其中 15 篇进入定量分析**。
- **方法/结果**：摘要未在可获取部分给出合并敏感性/特异性数值。
- **对我们的价值**：目前最新、覆盖最全的 AI×DDH 文献地图。**用它可以回答"我们的问题有没有被人正面做过"**——从纳入研究的主题分布看，主流是"诊断/分型/测角"，"裁剪 / 边界线"这一类**输出形态**几乎无人涉及，这是我们差异化的依据（引用前建议读全文确认这一判断）。
- **局限**：2026 年新刊，纳入研究以分类/测角为主，与"裁剪"跨领域。

### [01-13] Deep learning algorithms for identifying DDH based on sonographic images（Xu N, …, Ni D, Yang X, Zhou L; eClinicalMedicine 2025;89:103552）
- **做了什么**：HipSonoNeuNet（HSNN），**中国 22 家医院**（2022-09 至 2025-01），3082 例受试者，7286 张图（动态 1429 / 静态 5857）。三阶段：I 训练（2431 例）；II 与不同经验放射科医生对比（500 例）；III 前瞻性验证（151 例）。
- **方法**：深度 CNN 做 DDH 诊断（非测角）。
- **结果**：内部测试 AUC 0.99、敏感度 1.00、特异度 0.91；人机对比准确率 0.94、AUC 0.99、与专家一致性 κ（文中记作魏）0.77；**AI 辅助让 7 名初级放射科医生准确率 0.90→0.93、AUC 0.80→0.95、敏感度 0.69→0.97**，并缩短检查时间、提升观察者间一致性；前瞻验证准确率 0.92。
- **对我们的价值**：**评价口径 + 泛化性论据**。"AI 辅助提升初级人员"这一评价框架，正是我们工具的商业/临床价值叙事；多中心设计也提醒我们跨机型（设备差异）泛化必须自测。
- **局限**：只做诊断分类，不做测角与裁剪；注意 AUC≈0.99 很可能受数据集难度影响，不宜当作"任务已解决"的证据。

### [01-14] Multihipposenet: multi-task deep learning for automated DDH assessment in infant hip ultrasound（Chi W et al., Front Pediatr 2026）
- **做了什么**：端到端多任务网络，1355 张标注图（566 例婴儿），两个数据集。
- **方法**：ResNet-50 + FPN + 上下文语义融合模块提多尺度特征；**ME-GCT**（多专家门控通道变换）建模"任务-通道"关系以权衡协同/冲突；**MT-GALM**（多任务图自适应学习）挖掘相邻通道跨任务的隐式依赖。
- **结果**：患者级交叉验证中，**α 角 97.97%、β 角 96.81% 的估计落在专家参考 ±5° 内**。
- **对我们的价值**：**性能上界参照**。若我们的上边界最终希望"下游角度不偏离专家"，可以用"落入 ±5° 的比例"而不是像素误差作为对外指标。另外它的多任务思路说明：同时预测上/下两条线 + 角度代理量，可能比单独预测上边界更容易收敛（多任务正则化）。
- **局限**：作者自己明确：**只验证了 Graf I/II 型**，属技术可行性，III/IV 型未验证；数据量小（1355 张）。

### [01-15] AI-Assisted Wireless Handheld Ultrasound for Screening DDH in Infants（Zhang D et al., J Ultrasound Med 2026）
- **做了什么**：AI-HUDs 系统，用 **6 个解剖关键点** + Graf 法算 α/β；数据 1192 张静态图 + 498 段动态视频；用 **YOLOv8** 训练。
- **结果**：静态模式相对专家手测，α/β 角差的标准差 1.38°/2.14°，MAE 1.05°/1.65°，ICC 0.82/0.64；动态模式 MAE 1.18°/1.86°，ICC 0.77/0.51；由住院医师操作设备、与专家手测对照仍保持良好一致性。
- **对我们的价值**：**window 定位 / 关键点检测**。它证明"YOLO 单阶段直接出 6 个关键点"在弱设备和弱操作者下可用，且误差量级（α MAE≈1°）比两阶段方案更好。**我们的检测器若目前只输出一个框，可以试改成多类别检出（窗口框 + 上/下边界锚点），一次前向同时给出粗定位。**
- **局限**：β 角 ICC 偏低（0.51–0.64），说明β侧本就不稳；纯关键点方案对"线"的位置约束弱。

### [01-16] AI-SPS: Artificial Intelligence-Assisted Standard Plane Detection in Hip Ultrasound（Darilmaz MF et al., J Orthop Res 2025）
- **做了什么**：实时标准切面检测软件 AI-SPS。2737 个标注帧（1737 标准 / 1000 非标准，来自 45 段临床超声视频）训练；另用 934 帧（347 标准 / 587 非标准）独立验证。对比 SSD-MobileNetV2 与 YOLOv11n。
- **结果**：YOLOv11n 准确率 86.3%、精确率 0.78、召回 0.88、F1 0.83；SSD-MobileNetV2 75.2%。
- **对我们的价值**：**上边界 / 评价口径的认知校正**。它说明"标准切面"的判定在**帧级别**本身就只有 ~86% 的可靠度——即"标准"是个**软标签**。推论：我们不要为上边界追求亚像素级的"绝对正确"，而应把"专家标注的边界离散度"当作可达误差下限来设验收目标（例如先测同一图上 3 位专家的上边界标准差）。
- **局限**：数据来自同一批视频源（同中心、同设备），泛化未知；只判"是不是标准帧"，不给出边界。

### [01-17] Detection of standard plane from ultrasound scans by deep learning methods for DDH diagnosis（Çevik KK, Andaç Ş; Mühendislik Bilimleri ve Tasarım Dergisi 2022:1014-1026）
- **做了什么**：深度学习方法做标准平面检测。
- **方法/结果**：**摘要未取得**（土耳其 OA 期刊，DergiPark 有 PDF，未下载）。
- **对我们的价值**：若需要"非英文但 OA 可复现"的对照实验来源，可后续补读。
- **局限**：期刊影响因子低，宜作补充证据。

### [01-18] Toward automatic diagnosis of hip dysplasia from 2D ultrasound（Hareendranathan AR et al., IEEE ISBI 2017）
- **做了什么**：早期 2D 超声自动 DDH 诊断。
- **方法/结果**：**摘要未取得**（IEEE 付费）。
- **对我们的价值**：历史基线引文。
- **局限**：非开放获取。

### [01-19] Diagnosis on Ultrasound Images for DDH with a DL Model Focusing on Signal Heterogeneity in the Bone Region（Shimizu H et al., Diagnostics 2025;15(4):403）
- **做了什么**：北海道大学团队，417 名高危婴儿的髋超声；用 **HigherHRNet-W48** 做顶点（apex point）估计，并分析"髂骨高亮度区域"的信号异质性、评估髂骨旋转，再构建 α 角做整体诊断；含 2D→3D 变换以抑制旋转/噪声。
- **结果**：(1) HigherHRNet-W48 顶点估计优于骨科住院医师；(2) 对图像做**"合格/不合格"质量评估**后，合格图误差显著更低；(3) 合格图上模型 DDH 检出 AUC 0.92，超过住院医师水平。
- **对我们的价值**：**上边界环节的核心参考**。它把"骨区亮度统计 + 旋转量"显式喂进模型，这正是我们规则算法可能失效的地方（骨影边界不是一个锐利边缘，而是一段强度渐变）。可落地的三条：(a) 在 ResNet18 回归头之外，把 **骨区强度直方图 / 梯度峰值位置**作为手工特征拼接进 MLP 头；(b) 先估"髂骨旋转角"，再把上边界在**旋转对齐的坐标系**里回归（消除姿态带来的误差）；(c) 借鉴其"先质量门控再测量"，我们可先判该图 upper-bound 是否可裁。
- **局限**：417 例、单中心；顶端估计与"整条上边界线"并不等价（点 vs 线）。

### [01-20] Automated Neonatal Hip Ultrasound System for Diagnosing DDH Using Assistive AI（Lee YS et al., J Imaging Inform Med 2025/39(1):518-531）
- **做了什么**：按 Graf 算法搭建自动化诊断流水线，含分类与分割两部分。
- **方法**：分类对比 NASNetMobile、MobileNetV1、DenseNet121、EfficientNetV2B0、ResNet50；分割对比 UnestED（打通 Unest+UNet）与 DeepLabV3Plus。文中自述 Grad 角与专家相比平均误差率 0.21。
- **结果**：NASNetMobile AUC 0.864（95%CI 0.850–0.878）为分类最佳；UnetED 分割 Dice 0.794、单张推理 40.078 ms；DeepLabV3Plus 在"手机+掌上超声"场景下稳健。
- **对我们的价值**：**选型基线 + 速度参照**。40 ms/张说明分割头在 Windows 桌面端（CPU/入门 GPU）也够快；分类主干对比表可直接用于我们"ResNet18 是否够用"的论证。
- **局限**：摘要口径较粗（"0.21"的"误差率"定义需读全文）；未提供边界线误差。

### [01-21] A Deep Learning–Based Framework for Automatic Determination of DDH from Graf Angles（Taşolar S et al., J Imaging Inform Med 2025;39(1):250-264）
- **做了什么**：从 Graf 角度自动判定 DDH 的框架。
- **方法/结果**：**摘要未取得**（OpenAlex 记录为 null abstract，PubMed 无对应题名）。
- **对我们的价值**：与"测角 → 判定"链条对齐，若我们需要把裁剪结果接到下游诊断，可参考。
- **局限**：内容未核实，引用前必须读原文。

### [01-22] Can AI Automatically Assess Scan Quality of Hip Ultrasound?（Hareendranathan AR, Mabee M, … Jaremko JL; Applied Sciences 2022;12(8):4072）
- **做了什么**：把 3D US 髋扫描的"扫描质量"作为二分类（adequate / inadequate）问题。Dataset1 = 2187 张 3DUS 图（单阅片者 1–5 分），Dataset2 = 107 张（4 位阅片者 10 分制半定量）。
- **结果**：二分类准确率 **96%（DS1）/ 91%（DS2）**，与专家读数高度一致。
- **对我们的价值**：**评价口径 / 可用性判断**。它证明"可用性"用简单 CNN 就能做到 90%+，但代价是**必须有分级标注体系（1–5 分或 10 分制）**。我们的数据目前是"二值可裁/不可裁"还是"连续好坏"？建议补一个 3 级或 5 级质量标注，这会同时决定 evaluation、决定训练目标。
- **局限**：3D US 数据，与我们的 2D 裁剪图域不同；"adequate"的定义依赖阅片者。**注意作者姓名在不同库中拼写不一致（Hareendranathan / Hareendrananthan），引用时以期刊版面为准。**

### [01-23] Real Time RNN Based 3D Ultrasound Scan Adequacy for DDH（Paserin O, Mulpuri K, Cooper A, Hodgson AJ, Garbi R; MICCAI 2018, LNCS pp.365-373）
- **做了什么**：RNN 实时判断 3D US 扫描是否充分。
- **方法/结果**：**摘要未取得**。
- **对我们的价值**：提示"充分性判断"可以做成**序列模型**（对帧序列打分）。如果我们有连续帧数据，可考虑时序平滑上边界（抑制单帧抖动）。
- **局限**：3D US 场景，非开放获取。

### [01-24] Comparison of quality and interpretation of newborn US screening for DDH by basically trained nurses and junior physicians（Ulziibat M et al., PLOS ONE 2024;19(4):e0300753）
- **做了什么**：蒙古国乡村场景，6 名护士 + 6 名初级医生，均无超声经验、接受 Graf 标准培训；201 名新生儿（402 髋，804 张标准图）；以两位瑞士督导的 Graf 判定为金标准。
- **结果**：两组标准切面图像质量**无统计学差异**；正确判定 I 型/非 I 型的比例 92.0%（医生）vs 89.1%（护士，p=0.484）。**最常见错误：缺失下缘（missing lower limb）、测量线画错、技术问题**。结论强调"基础培训后必须定期质量控制与复训，护士尤其需要解剖结构识别训练"。
- **对我们的价值**：**上/下边界环节的临床错误 taxonomy**。"missing lower limb（缺髂骨下缘/下角）"和"wrong measurement lines（测量线画错）"正好对应我们两条裁剪线的两类失败模式。**建议直接把这几个术语写进我们的错误分析表**，并在人工微调 UI 上把它们做成快捷错误标签——这样积累的数据能直接回答"上边界到底是画错、还是根本没看到"。
- **局限**：样本量偏小（每组约 400 张图）；仅 1 型/非 1 型二分类；蒙古国的资源环境与国内三甲不同。

### [01-25] SegFormer3D: Improving the Robustness of DL Model-Based Image Segmentation in Ultrasound Volumes of the Pediatric Hip（Ultrasound Med Biol 2025）
- **做了什么**：(1) 对 3D US 骨分割的 SOTA 做**定量鲁棒性评估**，给出 CNN 与 ViT 模型在"分辨率和解剖视野 FOV 的小变动"下失败/不合理分割的实例；(2) 提出 SegFormer3D 改进鲁棒性。
- **对我们的价值**：**“为什么上边界会漂”的最直接文献证据**：模型对 FOV/分辨率的小变化高度敏感。我们的 window 裁剪本质上就是在改变 FOV——**同一张原图，window 定位多/少几个像素，喂进 ResNet18 的上边界输入就变了**。据此可提出一个明确的实验：固定边界真值、把 window 框在 ±5% 内随机扰动，测上边界误差的分布（这将把"上边界不准"拆成"window 不准"与"回归本身不准"两份误差，是我们最该先做的诊断实验）。
- **局限**：3D US 数据；作者自己指出当前技术鲁棒性不足以直接上临床。

### [01-26] Morphological Development of the Hip in Normal Infants Under Six Months of Age by the Graf US Method（Liu B, Hu X, Li L, Gao S; Front Pediatr 2022;10:914545）
- **做了什么**：中国 3067 名婴儿 6134 髋（仅 I 型/IIa 型），按年龄分组比较 α、β 角与股骨头覆盖率（FHC）。
- **结果**：平均 α 角 左 62.4±3.6° / 右 63.2±3.5°；平均 β 角 左 55.6±4.5° / 右 54.8±4.5°；FHC 约 54.2%±4.6% / 54.8%±4.2%。
- **对我们的价值**：**几何先验**。可用于 (a) 在 window 内做尺度归一化时设定期望的髋臼宽度/深度比例；(b) 对裁剪结果做 sanity check（如果由裁剪图算出的 α 明显偏离 62±4°，说明上边界大概率画错了——可做成一个启发式告警）。
- **局限**：仅正常髋（I/IIa），不覆盖病理形态；α/β 的分布不能直接当成"该图的上边界应在哪"。

### [01-27] A method for early ultrasound classification of DDH in infants: combining YOLOv8-based keypoint detection with radiomics（Zhu Z et al., BMC Pediatrics 2026）
- **做了什么**：YOLOv8 关键点检测 + 影像组学特征，做 DDH 早期分类。
- **方法/结果**：**摘要未取得**（OpenAlex 无 abstract；BMC 为 OA，可后续下载 PDF）。
- **对我们的价值**：**YOLOv8 关键点**路线的又一独立复现证据，可作为我们检测器升级的旁证。
- **局限**：内容未核实。

### [01-28] Automated Diagnosis of DDH in Ultrasound Using an End-to-End Deep Learning Approach（Pulik Ł et al., Research Square 2026，**预印本**）
- **做了什么**：端到端深度学习诊断 DDH。
- **方法/结果**：**未取得摘要**；且为**未经同行评议的预印本**。
- **对我们的价值**：仅作趋势参考，**不得作为证据引用**。
- **局限**：预印本。

### [01-29] Deep CNN-Based Automatic Classification of Neonatal Hip Ultrasound Images: A Novel Data Augmentation Approach with Speckle Noise Reduction（Sezer A, Sezer HB; Ultrasound Med Biol 2019/2020）
- **做了什么**：针对超声散斑噪声做抑制 + 数据增强的髋超声分类。
- **方法/结果**：**摘要未取得**；元数据经 Crossref 与 OpenAlex 双重核对（发表 2020-03，OpenAlex 标 2019）。
- **对我们的价值**：**物理层面的解释**——超声散斑是上边界"看起来有 10 个候选位置"的原因之一。若我们上边界误差集中在低 SNR 图上，可试"先做散斑抑制再回归"作为消融。
- **局限**：分类任务，不是边界任务；内容未核实。

### [01-30] Shadow Peak: Accurate Real-time Bone Segmentation for Ultrasound and DDH（EPiC Series in Health Sciences 2019）
- **做了什么**：基于骨产生的**声影（shadow）**做实时骨分割。
- **方法/结果**：**摘要未取得**（OpenAlex 无 abstract）。
- **对我们的价值**：一个具体的**上边界新特征候选**："骨声影的上起点"往往比骨表面高亮带更稳定。可在规则算法里加一路"向下扫描首个持续暗区"的检测，与神经网络输出融合。
- **局限**：会议短文；需读原文确认方法细节。

---

## 三、综合结论

### 1. 我们的两阶段路线（检测器找窗口 + 回归找边界）与文献主流是否一致？

**方向上一致，但"第二阶段的输出形态"不一致。**

- **一致的部分（强）**：DDH 髋超声自动化的主流解法确实是"**先定位区域/结构 → 再在局部做精细任务**"。
  - [01-04] 的结构几乎是我们的镜像：**global 网络预测 32×32 ROI 框 → local 网络在 patch 内回归 5 个点**，作者报告的 mAP 0.83、总体 RMSE 4.02 px。
  - [01-02] 是"分割结构 → 在分割图上取点"的两阶段；[01-15] 用 YOLOv8 一次性出 6 个点；[01-14] 用多任务网络同时学分割与关键点。
  - 也就是说：**"窗口/ROI 先行"这一点，文献支持度很高**，我们的第一阶段不用怀疑路线。
- **不一致的部分（这是我们的差异化，也是风险）**：文献第二阶段的输出几乎无一例外是**离散关键点**（5–6 个点），再由点连成线算 α/β角；而我们输出的是**两条水平裁剪线的绝对位置**。后果有三：
  1. 学术界几乎没有"上边界线位置误差"的公开基线，我们没有现成的对比数字（只能自建评价集）；
  2. 但**可以移植它们最有效的技术工具**——尤其是 [01-03] 的"结构→点一致性约束"和 [01-19] 的"骨区信号异质性特征"；
  3. 更有利的一点：关键点任务的误差评价（像素距离）在我们这里是**天然对齐**的（[01-08] 报 bony rim 4.8 px、lower limb 4.5 px），**我们的上边界误差应当先对齐到"同级像素误差"这个量级再谈目标**。
- **一个必须吸收的方法论共识**：几乎所有严谨工作都**先把"图像是否可用/是否标准切面"作为前置门控**（[01-02] 的 34.7% 可用率、[01-19] 的合格/不合格、[01-22] 的 96%/91%、[01-16] 的 86.3%）。我们的 pipeline 也应当把"这张图能不能裁"作为一级分类，而不是对每张图都强行输出边界线。

### 2. 有哪几篇的做法能直接改善**上边界预测**？（按优先级）

| 优先级 | 文献 | 可直接落地的动作 |
|---|---|---|
| ★★★ | **[01-03]** Hu et al. JBHI | 并联轻量分割头（髂骨/骨-软骨边界），用**分割边缘作为上边界强先验**做后处理：把回归线吸附到分割边界梯度最大处；若二者差 > 阈值则告警。配套 shape-similarity / landmark-structure 一致性 loss |
| ★★★ | **[01-25]** SegFormer3D | **先做误差归因实验**：固定上边界真值，把 window 框在 ±5% 随机扰动，量化上边界误差的方差增量——把"window 定位误差"与"回归误差"分离（这一步成本最低，收益最大） |
| ★★☆ | **[01-19]** Shimizu et al. | 把**骨区强度直方图/梯度峰**作为手工特征拼进回归头；先估髂骨旋转角，在**旋转对齐坐标系**中回归上边界 |
| ★★☆ | **[01-08]** Pulik et al. | 上边界验收指标改双轨：**像素距离 + 下游 α 角偏差**；并注意"软骨-骨边界 IoU 仅 0.63"说明该边界本身模糊，需先测**专家间离散度**作为误差下限 |
| ★★☆ | **[01-24]** Ulziibat et al. | 把上/下边界失败接入临床错误 taxonomy（缺下缘 / 测量线画错 / 技术问题），在人工微调 UI 里做快捷标签 |
| ★☆☆ | **[01-29]** Sezer & Sezer | 低 SNR 图上先做散斑抑制再做回归（消融项） |
| ★☆☆ | **[01-30]** Shadow Peak | 加一路"骨声影起点"检测，与网络输出融合 |
| ★☆☆ | **[01-16]** AI-SPS | 认知校正：标准切面标注本身只有 ~86% 帧级一致性 → **不要以"绝对正确"设验收标准**，以专家离散度为基准 |

> 如果只能改一件事：在 **[01-25] 的窗口扰动实验**（拆误差归因）与 **[01-03] 的结构约束**之间，先做前者。因为若上边界误差主要来自 window 抖动，那么任何回归头改进都是徒劳。

### 3. 只精读 3 篇，读哪 3 篇、为什么？

1. **[01-04] Ahn et al. 2025, BMC Med Inform Decis Mak（开放全文，PMC11837398）**
   为什么：**它就是我们架构的镜像实现**——global 出框 + local 在框内回归点，且给出了逐点误差表（第 1 点 1.69 px → 第 5 点 6.61 px）。读它最少的时间就能拿到三样东西：两阶段训练细节（ROI 32×32、IoU loss + RMSE loss 的组合）、逐点误差的分布规律（证明"越靠外周越不准"，直接支持"上边界比下边界难"）、以及"平面是否标准"的角度化判据（±5°）。免费、可全文复现。

2. **[01-03] Hu et al. 2022, IEEE JBHI（arXiv 2106.05458 可免费读）**
   为什么：**唯一一篇系统解决"点被结构干扰"的论文**，而这正是我们上边界不准的机制所在。它的 shape similarity loss 和 landmark-structure 一致性先验是**方法论可移植**的（不必复现其全部网络，只要在我们的 ResNet18 旁边加一个分割分支做约束）。同时它是该方向上精度标杆（α 平均误差 2.221°）。

3. **[01-25] SegFormer3D 2025, Ultrasound Med Biol**
   为什么：**它是唯一一篇把"为什么模型在 FOV/分辨率变动下会崩"量化的论文**。我们整个项目的输入就是"被裁出来的 window"，这段实验直接告诉我们"window 边界抖动多少 → 下游分割/边界误差涨多少"。读它的价值不在模型本身，而在**它给出的鲁棒性测试协议**，可以直接改造成我们的误差归因实验。若时间只够读两篇，第 3 篇可退为"读方法部分即可"。

> 备选第 4 篇（若关心可用性/评价口径）：**[01-02] Lee et al. Diagnostics 2021**（开放全文，含数据分类流程与 ICC 统计），是所有"可用性门控 + 评价指标"的模板；以及 **[01-24] PLOS ONE 2024**（从未受训操作者的错误清单，最接近硅基产品要面对的真实使用场景）。

---

## 四、未确认/存疑

以下条目**未能核实到确切的论文出处**，一律不进正式清单；如需使用，必须先补证。

### A. 检索到标题但无法确认元数据的（**不要引用**）

| 编号 | 内容 | 问题 |
|---|---|---|
| A1 | "Improving clinician trust in automated hip screening: Utilising heatmaps to express confidence in anatomical landmarks"（推断来自牛津大学 ORA 库） | 只见到 ORA 页面标题，未取得 DOI / 期刊 / 年份 / 作者；无法确认真实存在性。若要收录，需先从 ORA 页面下载并核对。 |
| A2 | "Retuve: Automated multi-modality analysis of hip dysplasia with open source AI" | 检索到 ScienceDirect 链接片段，但未取得可核实的 DOI/卷期；未确认是否为经同行评议的正式文章。 |
| A3 | "Explainable Quality Assessment and Measurement from Real-World Hip Ultrasound Cine Sweeps"（MDPI Bioengineering 2026 方向） | 仅有搜索结果片段（mdpi.com/2306-5354/13/6/667），未通过 Crossref/PubMed 核对；**疑似相关度很高（cine sweep + 质量评估）**，建议下一轮优先核实。 |
| A4 | "Computer-aided diagnosis of DDH using ultrasound: deep learning for segmentation and accurate angle measurement aligned with radiologist's clinical workflow"（Yilmaz ME, Colak E, Guzel MS；*Medical Ultrasonography*；doi:10.11152/mu-4535；PMID 40789016） | PMID 与 DOI 已核对，但**摘要未取得**（PubMed 返回空 ABSTRACT 字段），且元数据显示 "2026 Mar 18"、内容疑为 ahead-of-print，作者列表以 PubMed 与 OpenAlex 为准。因未读到内容，本轮未收录进清单——**若需要，这是最值得补的一篇（"对齐放射科医生工作流"与我们的裁剪流程高度相关）**。 |
| A5 | "A Deep Learning–Based Framework … from Graf Angles"（01-21） | DOI 已核实，但摘要为空；清单中已标"仅元数据"。 |
| A6 | 一篇印度团队（JMAU）标注关键点的论文 | 用 DOI `10.4103/jmau.jmau_63_21` 在 OpenAlex 与 Crossref 检索，**实际指向一篇鼻窦真菌论文**（Bapodra MK et al.）。判定 DOI 与标题不匹配，**整条作废**，不作引用。若确有此文，其 DOI 应为另一个，须重新检索确认。 |

### B. 关于"Missing hips"等具体数字的提醒
- 我**没有**在本轮核实到任何关于"Graf 筛查中臀部漏扫比例"的权威数值，因此不写入正文。若报告需要此类数字，请单独核实（例如 [01-11]/[01-12] 两篇系统评价的全文）。
- [01-13] 摘要中专家一致性符号在原文被转写为"魏"（应为 κ），引用时请以原文 PDF 为准。

### C. 关于"我们的问题是否已有人做过"的判断
- 依据 [01-12] 与 [01-11] 的**摘要级信息**（纳入研究以分类/分型/测角为主），我判断"整图自动裁剪输出（window + 上/下裁剪线）"在文献中**没有直接对应工作**。但这只是摘要级推断，**建议在做正式文献综述（systematic review 级别）时用全文核实**，不要直接引为"尚无人研究"。
- 与"裁剪"最接近的跨域概念是**超声图像 ROI / 成像窗口自动定位**（例如胎儿超声标准平面与 ROI），本轮**尚未检索该方向**——这部分应放在本目录的 `03-超声ROI与窗口定位.md`，本文件不越界推断。

### D. 元数据不一致的已知点
- [01-22] 作者姓氏在 OpenAlex("Hareendrananthan") 与 Crossref("Hareendranathan") 不一致，且与 [01-18] 第一作者同一人；引用以期刊版面为准。
- [01-29] 年份在两库中分别为 2019（OpenAlex）与 2020-03（Crossref 出版日期），引用建议写 "2019/2020（在线/正式）" 或按卷期。
- [01-20] 发表年份 Crossref 为 2025-04-23（Epub），PubMed 卷期为 2026 Feb;39(1):518-531；清单里标 "2025/2026"。