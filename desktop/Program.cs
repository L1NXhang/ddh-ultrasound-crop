using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Windows.Forms;

namespace DDHDesktop
{
    static class Program
    {
        internal static string Root;
        internal static string RuntimePath {
            get { return Path.Combine(Root, "data", "studio_runtime"); }
        }
        [DllImport("user32.dll")] static extern bool SetForegroundWindow(IntPtr window);
        [DllImport("user32.dll")] static extern bool ShowWindow(IntPtr window, int state);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] static extern bool SetDllDirectory(string path);

        [STAThread]
        static void Main(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            try
            {
                Root = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ddh_multidevice_v2");
                if (!File.Exists(Path.Combine(Root, "tools", "crop_studio.py")) && !File.Exists(Path.Combine(Root, "backend", "DDHBackend.exe")))
                    throw new FileNotFoundException("请将软件与 ddh_multidevice_v2 项目文件夹放在同一目录。");
                string runtime = Path.Combine(Root, "desktop", "runtime");
                SetDllDirectory(runtime);
                AppDomain.CurrentDomain.AssemblyResolve += delegate(object sender, ResolveEventArgs request)
                {
                    string name = new AssemblyName(request.Name).Name;
                    if (!name.StartsWith("Microsoft.Web.WebView2.", StringComparison.Ordinal)) return null;
                    string path = Path.Combine(runtime, name + ".dll");
                    return File.Exists(path) ? Assembly.LoadFrom(path) : null;
                };
                bool created;
                using (var mutex = new Mutex(true, InstanceName(), out created))
                {
                    if (!created) { ActivateExisting(); return; }
                    Run(args);
                }
            }
            catch (Exception error)
            {
                DesktopLog.Write(error.ToString());
                MessageBox.Show(error.Message, "DDH 裁剪工具 · 启动失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        static string InstanceName()
        {
            using (var sha = SHA256.Create())
                return @"Local\DDHCropStudio_" + BitConverter.ToString(sha.ComputeHash(Encoding.UTF8.GetBytes(Root.ToLowerInvariant()))).Replace("-", "").Substring(0, 16);
        }

        static void ActivateExisting()
        {
            string executable = Process.GetCurrentProcess().MainModule.FileName;
            foreach (Process process in Process.GetProcessesByName(Process.GetCurrentProcess().ProcessName))
            {
                try
                {
                    if (process.MainModule.FileName != executable || process.MainWindowHandle == IntPtr.Zero) continue;
                    ShowWindow(process.MainWindowHandle, 9);
                    SetForegroundWindow(process.MainWindowHandle);
                    return;
                }
                catch (System.ComponentModel.Win32Exception error) { DesktopLog.Write(error.Message); }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        static void Run(string[] args) { Application.Run(new StudioWindow(args)); }
    }

    static class DesktopLog
    {
        static readonly object Gate = new object();
        internal static void Write(string value)
        {
            string root = Program.Root ?? Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ddh_multidevice_v2");
            string folder = Program.Root == null ? Path.Combine(root, "data", "studio_runtime") : Program.RuntimePath;
            try
            {
                Directory.CreateDirectory(folder);
                lock (Gate) File.AppendAllText(Path.Combine(folder, "desktop.log"), DateTime.Now.ToString("s") + " " + value + Environment.NewLine);
            }
            catch (IOException error) { Debug.WriteLine(error); }
            catch (UnauthorizedAccessException error) { Debug.WriteLine(error); }
        }
    }
}
