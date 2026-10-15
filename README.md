# DDH 图像裁剪

Windows 桌面工具，用于超声图像的批量裁剪与人工审核。模型在本机预测成像窗口和上下边界，人工可调整最终裁剪框的四条边；支持草稿恢复、成品预览和浏览器协作审核。

## 下载与安装

**普通用户请下载安装包，不要下载 GitHub 的 Code → Download ZIP 来当作软件安装。**

- [Windows CPU 安装包，推荐](https://github.com/L1NXhang/ddh-ultrasound-crop/releases/download/v0.5.0/DDH_Crop_Setup_0.5.0_CPU_x64.exe)：单个文件，下载后双击安装，包含 Python 推理环境、模型与 WebView2，无需独立显卡。
- [全部下载与 GPU 版本](https://github.com/L1NXhang/ddh-ultrasound-crop/releases/latest)：GPU 自动版同样是单个完整安装 EXE，兼容 NVIDIA CUDA 时使用 GPU，否则使用 CPU。
- [CPU 便携 ZIP](https://github.com/L1NXhang/ddh-ultrasound-crop/releases/download/v0.5.0/DDH_Crop_Portable_0.5.0_CPU_x64.zip)：完整解压后运行其中的 DDH裁剪工具.exe，保留整个目录结构。

支持 Windows 10/11 64 位，系统需具备 .NET Framework 4.8。安装到当前用户目录，无需管理员权限，可从 Windows 应用设置卸载。程序尚未配置代码签名证书，安装文件可能显示未知发布者；请核对本仓库发布地址和附带的 SHA256 清单。

应用目录中的小 EXE 是启动入口，不能单独复制给其他电脑。CPU 安装 EXE 则包含完整运行文件，可以单独分发。本次 CPU 和 GPU 安装包均为单个完整 EXE。

## 开始裁剪

1. 打开软件，选择图片文件夹和结果保存位置，两个目录分开选择。
2. 选择逐张审核或批量裁剪。首次预测需加载模型。
3. 红框是完整成像窗口，绿框是最终裁剪范围；拖动绿框四边，保留需要的解剖结构。
4. Enter 保存并下一张，K 稍后处理，P 查看成品预览，Ctrl+Z 撤销。
5. 红框需要调整时按 W，调整后按 R 重新预测，按 C 返回裁剪范围。
6. 结果即时保存到选择的工作目录，完成后可选是否另存 ZIP。

原图不被覆盖。未经确认的调整保存为草稿，确认后才生成成品并记为人工审核。自动批量结果会明确保留未审核状态。

## 让组员通过浏览器审核

负责人打开图片项目，点击右上角“协作审核”，为组员设置独立账号和密码。待审核图片会分配给各账号，组员在网页上调整和确认，结果直接存回负责人的电脑。

局域网使用主机 IP 与审核端口；外网需配置加密组网或 HTTPS 代理。主机保持开机、软件运行及网络连通。组员只获得审核权限，无法切换主机目录或管理服务。密码私下提供，不写入 GitHub 文档。

详见[协作审核说明](docs/协作审核使用说明_20261015.md)和[给组员的一页纸](docs/给同学的一页纸.md)。

## 数据、模型与训练

仓库只保存源代码、说明、测试和合成示例，不上传原始医学图片、实际标注、逐图评估记录、本机配置或训练目录。发行安装包包含已训练的推理权重与第三方依赖许可，不包含项目图片或用户审核数据。

人工确认会记录原图 SHA256、原始预测、最终坐标、模型版本与审核人，生成训练候选。候选仍需复核、去重和合理划分后才能用于训练；软件目前不会一键重新训练或自动替换模型。

现有裁剪模型学习上下边界。成像区内部文字或婴儿图标的左右清理目前由人工框选完成，自动四边裁剪模型尚未训练。研究阶段的训练脚本仍保留，不能直接等同于已集成的桌面训练功能。

本工具用于图像整理与研究，不提供临床诊断结论。功能测试通过不代表跨设备或临床准确率已得到验证。

## 开发与构建

使用 Python 3.11，先按 CPU 或 CUDA 环境安装匹配的 PyTorch，再安装 requirements.txt。桌面外壳使用 .NET Framework 4.8 和 Microsoft WebView2。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python tools/start_studio.py
```

模型文件不在 Git 仓库内。开发时需另行提供兼容的 runs_v2/crop.pt；普通使用请直接下载完整发行包。

```powershell
python -m unittest discover -s tests -p "test_*.py"
python tools/audit_publication.py --include-untracked
```

完整打包命令、安装器构建及发布流程见[构建与发布](docs/构建与发布.md)。软件模型代码位于 ddh.py，当前产品入口为 tools/crop_studio.py，桌面外壳位于 desktop/，前端位于 tools/webapp/。

## 文档

[安装与分发](docs/安装与分发.md) · [全部文档索引](docs/README.md) · [裁剪工作台使用说明](docs/裁剪工作台使用说明.md) · [CPU/GPU 速度实测](docs/CPU_GPU速度实测_20261015.md)

历史研究文档保留原来的时间点与指标，详见[早期研究 README](docs/研究阶段README_20260921.md)；当前软件能力以本页和最新发行说明为准。
