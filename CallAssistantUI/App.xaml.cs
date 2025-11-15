using System;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using CallAssistantUI.Services;
using CallAssistantUI.ViewModels;

namespace CallAssistantUI;

public partial class App : Application
{
    private MainViewModel? _mainViewModel;
    private Process? _backendProcess;
    private volatile bool _shuttingDown;

    protected override async void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        await EnsureBackendStartedAsync().ConfigureAwait(false);

        var backendBaseAddress = Environment.GetEnvironmentVariable("CALL_ASSISTANT_BACKEND_URL")
                                    ?? "http://localhost:8000/";
        var websocketBaseAddress = Environment.GetEnvironmentVariable("CALL_ASSISTANT_WS_URL")
                                       ?? "ws://localhost:8000";

        var restClient = new RestClient(backendBaseAddress);
        var webSocketManager = new WebSocketManager(new Uri(websocketBaseAddress));

        _mainViewModel = new MainViewModel(restClient, webSocketManager);

        var window = new MainWindow
        {
            DataContext = _mainViewModel
        };

        window.Show();
    }

    private async Task EnsureBackendStartedAsync()
    {
        try
        {
            if (await IsBackendHealthyAsync().ConfigureAwait(false))
            {
                return;
            }

            var repoRoot = AppContext.BaseDirectory;
            var dir = new DirectoryInfo(repoRoot);
            while (dir != null && !Directory.Exists(Path.Combine(dir.FullName, "backend_server")))
            {
                dir = dir.Parent;
            }
            var workingDir = dir?.FullName ?? repoRoot;

            Directory.CreateDirectory(Path.Combine(workingDir, "logs"));
            var logPath = Path.Combine(workingDir, "logs", "backend.log");

            var psi = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = "-u backend_server/main_server.py",
                WorkingDirectory = workingDir,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };

            // Help backend find the native .pyd if it exists
            var pyPath = (Environment.GetEnvironmentVariable("PYTHONPATH") ?? string.Empty);
            var candidate1 = Path.Combine(workingDir, "cpp_asr", "build", "Release");
            var candidate2 = Path.Combine(workingDir, "build", "Release");
            if (Directory.Exists(candidate1))
            {
                pyPath = string.IsNullOrEmpty(pyPath) ? candidate1 : candidate1 + Path.PathSeparator + pyPath;
            }
            if (Directory.Exists(candidate2))
            {
                pyPath = string.IsNullOrEmpty(pyPath) ? candidate2 : candidate2 + Path.PathSeparator + pyPath;
            }
            if (!string.IsNullOrEmpty(pyPath))
            {
                psi.Environment["PYTHONPATH"] = pyPath;
            }

            _backendProcess = new Process { StartInfo = psi, EnableRaisingEvents = true };

            var stdout = new StreamWriter(new FileStream(logPath, FileMode.Create, FileAccess.Write, FileShare.ReadWrite)) { AutoFlush = true };
            var stderr = stdout; // share same log

            _backendProcess.OutputDataReceived += (_, args) => { if (args.Data != null) stdout.WriteLine(args.Data); };
            _backendProcess.ErrorDataReceived += (_, args) => { if (args.Data != null) stderr.WriteLine(args.Data); };
            _backendProcess.Exited += async (_, _) =>
            {
                if (_shuttingDown) return;
                // brief delay to avoid tight restart loop
                await Task.Delay(1000).ConfigureAwait(false);
                await EnsureBackendStartedAsync().ConfigureAwait(false);
            };

            _backendProcess.Start();
            _backendProcess.BeginOutputReadLine();
            _backendProcess.BeginErrorReadLine();

            var cts = new CancellationTokenSource(TimeSpan.FromSeconds(20));
            while (!cts.IsCancellationRequested)
            {
                if (await IsBackendHealthyAsync().ConfigureAwait(false))
                {
                    break;
                }
                await Task.Delay(500, cts.Token).ConfigureAwait(false);
            }
        }
        catch
        {
            // swallow startup failures; UI will still load but won't connect
        }
    }

    private static async Task<bool> IsBackendHealthyAsync()
    {
        try
        {
            using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(2) };
            var response = await client.GetAsync("http://localhost:8000/health").ConfigureAwait(false);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    protected override void OnExit(ExitEventArgs e)
    {
        _shuttingDown = true;
        try
        {
            if (_backendProcess is { HasExited: false })
            {
                _backendProcess.Kill(entireProcessTree: true);
                _backendProcess.Dispose();
            }
        }
        catch
        {
            // ignore
        }
        base.OnExit(e);
    }
}
