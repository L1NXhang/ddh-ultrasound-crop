#ifndef AppSource
  #error AppSource is required
#endif
#ifndef InstallerOutput
  #error InstallerOutput is required
#endif
#ifndef AppVersion
  #define AppVersion "0.6.0"
#endif
#ifndef Variant
  #define Variant "CPU"
#endif
#ifndef DiskParts
  #define DiskParts "0"
#endif

[Setup]
AppId=DDHCrop-{#Variant}
AppName=DDH 图像裁剪 ({#Variant})
AppVersion={#AppVersion}
AppPublisher=DDH Crop Project
AppPublisherURL=https://github.com/L1NXhang/ddh-ultrasound-crop
AppSupportURL=https://github.com/L1NXhang/ddh-ultrasound-crop/issues
AppUpdatesURL=https://github.com/L1NXhang/ddh-ultrasound-crop/releases/latest
DefaultDirName={localappdata}\Programs\DDHCrop-{#Variant}
DefaultGroupName=DDH 图像裁剪
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#InstallerOutput}
OutputBaseFilename=DDH_Crop_Setup_{#AppVersion}_{#Variant}_x64
SetupIconFile=app.ico
UninstallDisplayIcon={app}\DDH裁剪工具.exe
WizardStyle=modern
WizardResizable=yes
ShowLanguageDialog=no
Compression=lzma2/normal
SolidCompression=yes
LZMANumBlockThreads=4
DisableProgramGroupPage=yes
DisableWelcomePage=no
CloseApplications=no
RestartApplications=no
Uninstallable=yes
#if DiskParts == "1"
DiskSpanning=yes
DiskSliceSize=1900000000
SlicesPerDisk=1
#endif

[Languages]
; ShowLanguageDialog=no：默认使用第一个语言（简体中文）
Name: "zh"; MessagesFile: "ChineseSimplified.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式(&D)"; Flags: unchecked

[Files]
Source: "{#AppSource}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\DDH 图像裁剪 ({#Variant})"; Filename: "{app}\DDH裁剪工具.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\DDH 图像裁剪 ({#Variant})"; Filename: "{app}\DDH裁剪工具.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\DDH裁剪工具.exe"; Description: "打开 DDH 图像裁剪"; Flags: nowait postinstall skipifsilent

[Code]
var
  TrainingPage: TInputOptionWizardPage;
  TrainingChosen: Boolean;

function InitializeSetup: Boolean;
var
  FrameworkRelease: Cardinal;
begin
  Result := RegQueryDWordValue(HKLM, 'SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full', 'Release', FrameworkRelease) and
    (FrameworkRelease >= 528040);
  if not Result then
    MsgBox('DDH 图像裁剪需要 .NET Framework 4.8。请先安装 .NET Framework 4.8，再重新运行本安装程序。', mbError, MB_OK);
end;

procedure InitializeWizard;
begin
  // 自定义向导页：是否安装训练组件（本轮仅留入口，训练功能后续版本提供）
  TrainingPage := CreateInputOptionPage(wpSelectTasks,
    '训练组件',
    '是否安装模型训练组件？',
    '训练组件用于在本机用新标注数据重新训练裁剪模型，需要 NVIDIA 显卡（CUDA）并占用较大磁盘空间。日常裁剪不需要它。',
    True, False);
  TrainingPage.Add('安装训练组件（需要 NVIDIA 显卡，后续需单独下载）');
  TrainingPage.Add('暂不安装，仅进行图像裁剪（推荐）');
  TrainingPage.Values[1] := True;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = TrainingPage.ID then
  begin
    TrainingChosen := TrainingPage.Values[0];
    // 写入 HKCU 供后续版本的训练入口读取；本版本不据此安装任何组件
    RegWriteDWordValue(HKCU, 'Software\DDH Crop', 'InstallTraining', Integer(TrainingChosen));
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and TrainingChosen then
    WizardForm.FinishedLabel.Caption := WizardForm.FinishedLabel.Caption + #13#10 +
      '提示：已选择安装训练组件。训练功能需要单独下载 CUDA 版组件包，本版本暂未包含；请关注项目发布页的更新。';
end;
