using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using System.Web.Script.Serialization;

namespace DDHDesktop
{
    sealed class BackendProcess : IDisposable
    {
        readonly JavaScriptSerializer json = new JavaScriptSerializer();
        readonly string session;
        readonly HttpClient client = new HttpClient(new HttpClientHandler { UseProxy = false });
        Process process;
        internal string Url { get; private set; }
        internal string Token { get; private set; }
        internal int Pid { get { return process == null ? 0 : process.Id; } }

        internal BackendProcess(string handshake)
        {
            session = handshake;
            client.Timeout = TimeSpan.FromSeconds(5);
        }

        internal async Task StartAsync()
        {
            string python = Path.Combine(Program.Root, ".venv", "Scripts", "pythonw.exe");
            string bundled = Path.Combine(Program.Root, "backend", "DDHBackend.exe");
            bool portable = File.Exists(bundled);
            string executable = portable ? bundled : python;
            if (!File.Exists(executable)) throw new FileNotFoundException("找不到模型运行环境，请完整解压软件文件夹。");
            Directory.CreateDirectory(Path.GetDirectoryName(session));
            var info = new ProcessStartInfo(executable) {
                Arguments = (portable ? "" : Quote(Path.Combine(Program.Root, "tools", "crop_studio.py"))) +
                    " --desktop --port 0 --parent-pid " + Process.GetCurrentProcess().Id + " --session-file " + Quote(session) +
                    " --state-dir " + Quote(Program.RuntimePath),
                WorkingDirectory = Program.Root, UseShellExecute = false, CreateNoWindow = true,
                RedirectStandardOutput = true, RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8
            };
            info.EnvironmentVariables["PYTHONUTF8"] = "1";
            info.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";
            info.EnvironmentVariables["PYTHONNOUSERSITE"] = "1";
            if (portable) { info.EnvironmentVariables.Remove("PYTHONPATH"); info.EnvironmentVariables.Remove("PYTHONHOME"); }
            process = new Process { StartInfo = info };
            process.OutputDataReceived += delegate(object sender, DataReceivedEventArgs data) { if (data.Data != null) DesktopLog.Write(data.Data); };
            process.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs data) { if (data.Data != null) DesktopLog.Write(data.Data); };
            process.Start();
            process.BeginOutputReadLine(); process.BeginErrorReadLine();
            for (int count = 0; count < 200; count++)
            {
                if (process.HasExited) throw new InvalidOperationException("模型服务未启动，请查看 data/studio_runtime/desktop.log。");
                if (File.Exists(session)) { ReadSession(); return; }
                await Task.Delay(150);
            }
            throw new TimeoutException("模型服务启动超时，请查看 data/studio_runtime/desktop.log。");
        }

        void ReadSession()
        {
            var value = json.Deserialize<Dictionary<string, object>>(File.ReadAllText(session, Encoding.UTF8));
            Uri uri;
            bool owned = Convert.ToInt32(value["pid"]) == process.Id || Convert.ToInt32(value["parent_pid"]) == process.Id;
            if (!owned || !Uri.TryCreate(Convert.ToString(value["url"]), UriKind.Absolute, out uri)
                || uri.Scheme != "http" || uri.Host != "127.0.0.1" || uri.Port <= 0)
                throw new InvalidDataException("桌面会话信息无效。");
            Url = uri.AbsoluteUri;
            Token = Convert.ToString(value["token"]);
        }

        internal async Task StopAsync()
        {
            if (process == null || process.HasExited) return;
            if (Url != null)
            {
                using (var request = new HttpRequestMessage(HttpMethod.Post, Url + "api/shutdown"))
                {
                    request.Headers.Add("X-DDH-Token", Token);
                    request.Content = new StringContent("{}", Encoding.UTF8, "application/json");
                    using (var response = await client.SendAsync(request)) response.EnsureSuccessStatusCode();
                }
            }
            else { process.Kill(); }
            for (int count = 0; count < 600; count++)
            {
                if (process.HasExited) return;
                await Task.Delay(100);
            }
            throw new TimeoutException("当前图片仍在处理，请等待完成后再关闭。");
        }

        internal static string Quote(string value)
        {
            if (value.IndexOf('"') >= 0) throw new ArgumentException("路径包含无效字符。");
            return "\"" + value + "\"";
        }

        public void Dispose()
        {
            client.Dispose();
            if (process != null) process.Dispose();
        }
    }
}
