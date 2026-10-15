using System;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;

namespace DDHDesktop
{
    sealed class StudioWindow : Form
    {
        readonly WebView2 browser = new WebView2 { Dock = DockStyle.Fill, Visible = false, DefaultBackgroundColor = Color.FromArgb(250, 249, 245) };
        readonly Label splash = new Label { Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleCenter,
            ForeColor = Color.FromArgb(120, 118, 109), Font = new Font("Segoe UI", 12), Text = "正在打开…" };
        readonly BackendProcess backend;
        readonly string source;
        readonly int debugPort;
        bool closing;
        bool stopped;
        [DllImport("dwmapi.dll")] static extern int DwmSetWindowAttribute(IntPtr window, int key, ref int value, int size);

        internal StudioWindow(string[] args)
        {
            Text = "DDH 裁剪工具";
            BackColor = Color.FromArgb(250, 249, 245);
            AutoScaleMode = AutoScaleMode.Dpi;
            ClientSize = new Size(1440, 940);
            MinimumSize = new Size(1120, 760);
            StartPosition = FormStartPosition.CenterScreen;
            WindowState = FormWindowState.Maximized;
            Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath);
            string session = Path.Combine(Program.RuntimePath, "desktop_" + Guid.NewGuid().ToString("N") + ".json");
            foreach (string arg in args)
            {
                if (arg.StartsWith("--debug-port=")) int.TryParse(arg.Substring(13), out debugPort);
                else if (arg.StartsWith("--session-file=")) session = Path.GetFullPath(arg.Substring(15));
                else if (Directory.Exists(arg)) source = Path.GetFullPath(arg);
            }
            backend = new BackendProcess(session);
            Controls.Add(browser); Controls.Add(splash);
            Shown += async delegate { await InitializeAsync(); };
            FormClosing += OnClosing;
            FormClosed += delegate { backend.Dispose(); };
        }

        protected override void OnHandleCreated(EventArgs args)
        {
            base.OnHandleCreated(args);
            int enabled = 0;
            DwmSetWindowAttribute(Handle, 20, ref enabled, sizeof(int));
            int caption = 0x00f5f9fa, text = 0x00202728;
            DwmSetWindowAttribute(Handle, 35, ref caption, sizeof(int));
            DwmSetWindowAttribute(Handle, 36, ref text, sizeof(int));
        }

        async Task InitializeAsync()
        {
            try
            {
                await backend.StartAsync();
                string profile = Path.Combine(Program.RuntimePath, "webview_profile");
                string flags = debugPort > 0 && debugPort <= 65535 ? "--remote-debugging-port=" + debugPort + " --remote-debugging-address=127.0.0.1" : "";
                string fixedRuntime = await WebViewRuntime.FindAsync();
                var environment = await CoreWebView2Environment.CreateAsync(fixedRuntime, profile, new CoreWebView2EnvironmentOptions(flags));
                await browser.EnsureCoreWebView2Async(environment);
                browser.CoreWebView2.Settings.AreDefaultContextMenusEnabled = false;
                browser.CoreWebView2.Settings.AreBrowserAcceleratorKeysEnabled = false;
                browser.CoreWebView2.Settings.AreDevToolsEnabled = debugPort > 0;
                browser.CoreWebView2.Settings.IsStatusBarEnabled = false;
                new NativeBridge(this, browser.CoreWebView2, backend.Url);
                browser.CoreWebView2.NavigationCompleted += delegate(object sender, CoreWebView2NavigationCompletedEventArgs args)
                {
                    if (args.IsSuccess) { browser.Visible = true; splash.Visible = false; browser.Focus(); }
                    else splash.Text = "界面未能加载，请重新打开软件。\n" + args.WebErrorStatus;
                };
                string url = backend.Url + (source == null ? "" : "?source=" + Uri.EscapeDataString(source));
                browser.CoreWebView2.Navigate(url);
            }
            catch (Exception error)
            {
                DesktopLog.Write(error.ToString());
                splash.Text = "启动失败\n\n" + error.Message + "\n\n日志：data/studio_runtime/desktop.log";
                MessageBox.Show(this, error.Message, "DDH 裁剪工具", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        async void OnClosing(object sender, FormClosingEventArgs args)
        {
            if (stopped) return;
            args.Cancel = true;
            if (closing) return;
            closing = true;
            Exception closeError = null;
            try
            {
                await FlushDraftAsync();
                browser.Visible = false; splash.Visible = true;
                splash.Text = "正在结束当前任务…";
                await backend.StopAsync();
                stopped = true;
                browser.Dispose();
                Close();
            }
            catch (Exception error)
            {
                closing = false;
                DesktopLog.Write(error.ToString());
                browser.Visible = true; splash.Visible = false;
                closeError = error;
            }
            if (closeError != null)
            {
                try { if (browser.CoreWebView2 != null) await browser.CoreWebView2.ExecuteScriptAsync("if(typeof setBusy==='function')setBusy(false);"); }
                catch (Exception resetError) { DesktopLog.Write(resetError.ToString()); }
                MessageBox.Show(this, closeError.Message, "暂时无法退出", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
        }

        async Task FlushDraftAsync()
        {
            if (browser.CoreWebView2 == null || !browser.Visible) return;
            await browser.CoreWebView2.ExecuteScriptAsync("window.ddhCloseReady=null;if(window.prepareDesktopClose){window.prepareDesktopClose();}else{window.ddhCloseReady=true;}");
            DateTime deadline = DateTime.UtcNow.AddSeconds(55);
            while (DateTime.UtcNow < deadline)
            {
                string result = await browser.CoreWebView2.ExecuteScriptAsync("window.ddhCloseReady");
                if (result == "true") return;
                if (result == "false") throw new IOException("草稿尚未保存。请检查保存位置，重试后再关闭软件。");
                await Task.Delay(100);
            }
            throw new TimeoutException("当前操作尚未完成，请稍后再关闭软件。");
        }
    }
}
