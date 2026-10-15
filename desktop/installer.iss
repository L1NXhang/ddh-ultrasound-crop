#ifndef AppSource
  #error AppSource is required
#endif
#ifndef InstallerOutput
  #error InstallerOutput is required
#endif
#ifndef AppVersion
  #define AppVersion "0.5.0"
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
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "{#AppSource}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\DDH 图像裁剪 ({#Variant})"; Filename: "{app}\DDH裁剪工具.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\DDH 图像裁剪 ({#Variant})"; Filename: "{app}\DDH裁剪工具.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\DDH裁剪工具.exe"; Description: "Open DDH Crop"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup: Boolean;
var
  FrameworkRelease: Cardinal;
begin
  Result := RegQueryDWordValue(HKLM, 'SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full', 'Release', FrameworkRelease) and
    (FrameworkRelease >= 528040);
  if not Result then
    MsgBox('DDH Crop requires .NET Framework 4.8. Please install it and run setup again.', mbError, MB_OK);
end;
