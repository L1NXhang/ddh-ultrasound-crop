using System;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.Web.WebView2.Core;

namespace DDHDesktop
{
    static class WebViewRuntime
    {
        const string BootstrapperUrl = "https://go.microsoft.com/fwlink/p/?LinkId=2124703";

        internal static async Task<string> FindAsync()
        {
            string folder = Path.Combine(Program.Root, "desktop", "WebView2Runtime");
            if (!Directory.Exists(folder)) return null;
            if (!File.Exists(Path.Combine(folder, "msedgewebview2.exe")))
                throw new FileNotFoundException("浏览器运行组件不完整，请完整解压软件文件夹。");
            await EnsureReadAccessAsync(folder);
            return folder;
        }

        /// <summary>引导版不带固定运行时：系统已有 Evergreen 就直接用；
        /// 没有则下载微软官方约 2MB 的引导器静默安装。</summary>
        internal static async Task EnsureEvergreenAsync(Action<string> status)
        {
            try
            {
                CoreWebView2Environment.GetAvailableBrowserVersionString();
                return;
            }
            catch (Exception) { /* 系统没有 WebView2，走在线安装 */ }
            if (status != null) status.Invoke("正在下载浏览器组件（约 2 MB）…");
            string bootstrapper = Path.Combine(Path.GetTempPath(), "MicrosoftEdgeWebview2Setup.exe");
            using (var client = new HttpClient(new HttpClientHandler { UseProxy = false }) { Timeout = TimeSpan.FromMinutes(5) })
            using (var response = await client.GetAsync(BootstrapperUrl, HttpCompletionOption.ResponseHeadersRead))
            {
                response.EnsureSuccessStatusCode();
                using (var input = await response.Content.ReadAsStreamAsync())
                using (var output = new FileStream(bootstrapper, FileMode.Create, FileAccess.Write))
                    await input.CopyToAsync(output);
            }
            if (status != null) status.Invoke("正在安装浏览器组件…");
            using (var process = Process.Start(new ProcessStartInfo(bootstrapper, "/silent /install")
            {
                UseShellExecute = false,
                CreateNoWindow = true
            }))
            {
                bool ended = await Task.Run(() => process.WaitForExit(300000));
                if (!ended) { process.Kill(); throw new IOException("浏览器组件安装超时，请检查网络后重试。"); }
                if (process.ExitCode != 0) throw new IOException("浏览器组件安装失败（代码 " + process.ExitCode + "）。");
            }
            try
            {
                CoreWebView2Environment.GetAvailableBrowserVersionString();
            }
            catch (Exception error)
            {
                throw new IOException("浏览器组件已安装但仍不可用，请重启电脑后再试。" + error.Message);
            }
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
