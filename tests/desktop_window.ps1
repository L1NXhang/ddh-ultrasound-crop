param(
    [Parameter(Mandatory=$true)][int]$TargetPid,
    [Parameter(Mandatory=$true)][string]$ExpectedHash,
    [string]$ExpectedPath,
    [ValidateSet('Inspect','CancelDialog','AcceptDirectory','DismissStartupError','Close','Capture')][string]$Action = 'Inspect',
    [string]$OutputPath
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$expectedPath = if ($ExpectedPath) { [IO.Path]::GetFullPath($ExpectedPath) } else { [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\DDH裁剪工具.exe')) }
$workspacePath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if (!$expectedPath.StartsWith($workspacePath + '\', [StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetFileName($expectedPath) -ne 'DDH裁剪工具.exe') { throw 'Target path is outside the crop workspace' }
$target = Get-Process -Id $TargetPid
if ($target.Path -ne $expectedPath) { throw 'Target executable does not match this build' }
if ((Get-FileHash -LiteralPath $expectedPath -Algorithm SHA256).Hash -ne $ExpectedHash) { throw 'Executable hash changed' }
if ($target.SessionId -ne (Get-Process -Id $PID).SessionId) { throw 'Target belongs to another desktop session' }
$logFolder = Join-Path $PSScriptRoot '..\output\playwright'
New-Item -ItemType Directory -Path $logFolder -Force | Out-Null
@{time=[DateTime]::UtcNow.ToString('o');pid=$TargetPid;action=$Action;sha256=$ExpectedHash;tier='standard'} | ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $logFolder 'desktop_automation.jsonl') -Encoding UTF8
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
$condition = New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::ProcessIdProperty, $TargetPid)
$windows = [System.Windows.Automation.AutomationElement]::RootElement.FindAll([System.Windows.Automation.TreeScope]::Children, $condition)
$dialog = @($windows | Where-Object { $_.Current.ClassName -eq '#32770' }) | Select-Object -First 1
$main = @($windows | Where-Object { $_.Current.Name -eq 'DDH 裁剪工具' -and $_.Current.ClassName -like 'WindowsForms10*' }) | Select-Object -First 1
if (!$dialog -and $main) {
    $dialogCondition = New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::ClassNameProperty, '#32770')
    $dialog = $main.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $dialogCondition)
}
if ($Action -eq 'Inspect') {
    @($windows | ForEach-Object { @{name=$_.Current.Name;class=$_.Current.ClassName;handle=$_.Current.NativeWindowHandle;enabled=$_.Current.IsEnabled} }) | ConvertTo-Json -Compress
    if ($dialog) {
        $buttonsCondition = New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::ControlTypeProperty, [System.Windows.Automation.ControlType]::Button)
        @{dialog=$dialog.Current.Name;buttons=@($dialog.FindAll([System.Windows.Automation.TreeScope]::Descendants,$buttonsCondition) | ForEach-Object { @{name=$_.Current.Name;id=$_.Current.AutomationId;handle=$_.Current.NativeWindowHandle} })} | ConvertTo-Json -Compress -Depth 4
    }
    exit 0
}
if ($Action -eq 'Close') {
    if (!$main) { throw 'Owned application window not found' }
    $main.GetCurrentPattern([System.Windows.Automation.WindowPattern]::Pattern).Close()
    exit 0
}
if ($Action -eq 'Capture') {
    if (!$main -or !$OutputPath) { throw 'Missing owned window or output path' }
    Add-Type -AssemblyName System.Drawing
    Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class DDHCapture { [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr hwnd, IntPtr dc, uint flags); }'
    $bounds = $main.Current.BoundingRectangle
    $bitmap = New-Object System.Drawing.Bitmap([int]$bounds.Width, [int]$bounds.Height)
    $graphics = [Drawing.Graphics]::FromImage($bitmap)
    $dc = $graphics.GetHdc()
    try {
        if (![DDHCapture]::PrintWindow([IntPtr]$main.Current.NativeWindowHandle, $dc, 2)) { throw 'Window capture failed' }
    } finally { $graphics.ReleaseHdc($dc) }
    $bitmap.Save($OutputPath, [Drawing.Imaging.ImageFormat]::Png)
    $graphics.Dispose(); $bitmap.Dispose()
    exit 0
}
if (!$dialog) { throw 'Owned dialog not found' }
$automationId = '2'
if ($Action -eq 'AcceptDirectory') {
    $treeCondition = New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::ClassNameProperty, 'SysTreeView32')
    if (!$dialog.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $treeCondition)) { throw 'This is not a folder picker' }
    $automationId = '1'
}
if ($Action -eq 'DismissStartupError') {
    $text = $dialog.FindAll([System.Windows.Automation.TreeScope]::Descendants, [System.Windows.Automation.Condition]::TrueCondition)
    if (!(@($text | ForEach-Object {$_.Current.Name}) -match '桌面会话信息无效')) { throw 'Unexpected startup dialog' }
    $automationId = '1'
}
$idCondition = New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::AutomationIdProperty, $automationId)
$typeCondition = New-Object System.Windows.Automation.PropertyCondition([System.Windows.Automation.AutomationElement]::ControlTypeProperty, [System.Windows.Automation.ControlType]::Button)
$buttonCondition = New-Object System.Windows.Automation.AndCondition($idCondition, $typeCondition)
$button = $dialog.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $buttonCondition)
if (!$button) { throw 'Expected dialog action not found' }
$button.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
