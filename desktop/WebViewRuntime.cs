using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace DDHDesktop
{
    static class WebViewRuntime
    {
        internal static async Task<string> FindAsync()
        {
            string folder = Path.Combine(Program.Root, "desktop", "WebView2Runtime");
            if (!Directory.Exists(folder)) return null;
            if (!File.Exists(Path.Combine(folder, "msedgewebview2.exe")))
                throw new FileNotFoundException("浏览器运行组件不完整，请完整解压软件文件夹。");
            await EnsureReadAccessAsync(folder);
            return folder;
        }

        static async Task EnsureReadAccessAsync(string folder)
        {
            string marker = Path.Combine(Program.RuntimePath, "webview_permissions.txt");
            if (File.Exists(marker) && File.ReadAllText(marker) == folder) return;
            foreach (string sid in new[] { "*S-1-15-2-2", "*S-1-15-2-1" })
            {
                var info = new ProcessStartInfo(Path.Combine(Environment.SystemDirectory, "icacls.exe"),
                    BackendProcess.Quote(folder) + " /grant " + sid + ":(OI)(CI)(RX)") {
                    UseShellExecute = false, CreateNoWindow = true
                };
                using (var process = Process.Start(info))
                {
                    bool ended = await Task.Run(() => process.WaitForExit(10000));
                    if (!ended || process.ExitCode != 0) throw new IOException("无法设置浏览器组件读取权限，请将软件解压到本人可写的本地文件夹。");
                }
            }
            File.WriteAllText(marker, folder);
        }
    }
}
