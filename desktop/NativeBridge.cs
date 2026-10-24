using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using System.Windows.Forms;
using Microsoft.Web.WebView2.Core;

namespace DDHDesktop
{
    sealed class NativeBridge
    {
        readonly Form owner;
        readonly CoreWebView2 view;
        readonly Uri origin;
        readonly JavaScriptSerializer json = new JavaScriptSerializer();

        internal NativeBridge(Form window, CoreWebView2 browser, string url)
        {
            owner = window; view = browser; origin = new Uri(url);
            view.WebMessageReceived += Receive;
            view.NavigationStarting += delegate(object sender, CoreWebView2NavigationStartingEventArgs args) { args.Cancel = !IsLocal(args.Uri); };
            view.NewWindowRequested += delegate(object sender, CoreWebView2NewWindowRequestedEventArgs args) { args.Handled = true; };
            view.PermissionRequested += delegate(object sender, CoreWebView2PermissionRequestedEventArgs args) { args.State = CoreWebView2PermissionState.Deny; };
            view.DownloadStarting += Download;
        }

        bool IsLocal(string source)
        {
            Uri uri;
            return Uri.TryCreate(source, UriKind.Absolute, out uri) && uri.Scheme == origin.Scheme && uri.Authority == origin.Authority;
        }

        async void Receive(object sender, CoreWebView2WebMessageReceivedEventArgs args)
        {
            if (!IsLocal(args.Source) || !IsLocal(view.Source)) return;
            string id = "";
            try
            {
                string raw = args.WebMessageAsJson;
                if (raw.Length > 8192) throw new ArgumentException("目录请求过长。");
                var message = json.Deserialize<Dictionary<string, object>>(raw);
                id = Convert.ToString(message["id"]);
                if (id.Length == 0 || id.Length > 80)
                    throw new ArgumentException("无效的桌面操作。");
                string action = Convert.ToString(message["action"]);
                if (action == "saveArchive")
                {
                    string saved = await SaveArchiveAsync(Convert.ToString(message["archive_id"]));
                    view.PostWebMessageAsJson(json.Serialize(new { id = id, result = new { path = saved } }));
                    return;
                }
                if (action == "pickFile")
                {
                    string fileInitial = message.ContainsKey("initial") && !string.IsNullOrWhiteSpace(Convert.ToString(message["initial"]))
                        ? Convert.ToString(message["initial"]) : Program.Root;
                    using (var picker = new OpenFileDialog { Title = "选择标注清单", Filter = "标注清单 (*.jsonl)|*.jsonl|所有文件 (*.*)|*.*" })
                    {
                        string directory = File.Exists(fileInitial) ? Path.GetDirectoryName(fileInitial) : fileInitial;
                        if (Directory.Exists(directory)) picker.InitialDirectory = directory;
                        string path = picker.ShowDialog(owner) == DialogResult.OK ? picker.FileName : "";
                        view.PostWebMessageAsJson(json.Serialize(new { id = id, result = new { path = path } }));
                    }
                    return;
                }
                if (action != "pickFolder") throw new ArgumentException("无效的桌面操作。");
                string initial = message.ContainsKey("initial") ? Convert.ToString(message["initial"]) : Program.Root;
                using (var picker = new FolderBrowserDialog { Description = "选择工作目录或待裁图片文件夹", ShowNewFolderButton = true })
                {
                    if (Directory.Exists(initial)) picker.SelectedPath = initial;
                    string path = picker.ShowDialog(owner) == DialogResult.OK ? picker.SelectedPath : "";
                    view.PostWebMessageAsJson(json.Serialize(new { id = id, result = new { path = path } }));
                }
            }
            catch (Exception error)
            {
                DesktopLog.Write(error.ToString());
                view.PostWebMessageAsJson(json.Serialize(new { id = id, error = "目录选择失败：" + error.Message }));
            }
        }

        async Task<string> SaveArchiveAsync(string archiveId)
        {
            long key;
            if (!long.TryParse(archiveId, out key) || key <= 0) throw new ArgumentException("无效的压缩包标识。");
            using (var client = new HttpClient(new HttpClientHandler { UseProxy = false }))
            using (var response = await client.GetAsync(new Uri(origin, "api/archive/download?id=" + archiveId), HttpCompletionOption.ResponseHeadersRead))
            {
                response.EnsureSuccessStatusCode();
                var header = response.Content.Headers.ContentDisposition;
                string name = header == null ? "DDH裁剪结果.zip" : Path.GetFileName(header.FileNameStar ?? header.FileName);
                using (var dialog = new SaveFileDialog { Title = "另存压缩包副本", Filter = "ZIP 压缩包|*.zip", FileName = name })
                {
                    if (dialog.ShowDialog(owner) != DialogResult.OK) return "";
                    string temporary = dialog.FileName + "." + Guid.NewGuid().ToString("N") + ".tmp";
                    try
                    {
                        using (var input = await response.Content.ReadAsStreamAsync())
                        using (var output = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write))
                            await input.CopyToAsync(output);
                        if (File.Exists(dialog.FileName)) File.Replace(temporary, dialog.FileName, null);
                        else File.Move(temporary, dialog.FileName);
                        return dialog.FileName;
                    }
                    finally { if (File.Exists(temporary)) File.Delete(temporary); }
                }
            }
        }

        void Download(object sender, CoreWebView2DownloadStartingEventArgs args)
        {
            if (!IsLocal(args.DownloadOperation.Uri)) { args.Cancel = true; return; }
            args.Handled = true;
            using (var dialog = new SaveFileDialog { Title = "另存压缩包副本", Filter = "ZIP 压缩包|*.zip", FileName = Path.GetFileName(args.ResultFilePath) })
            {
                if (dialog.ShowDialog(owner) == DialogResult.OK) args.ResultFilePath = dialog.FileName;
                else args.Cancel = true;
            }
        }
    }
}
