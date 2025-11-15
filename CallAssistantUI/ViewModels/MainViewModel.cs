using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Threading;
using CallAssistantUI.Models;
using CallAssistantUI.Services;

namespace CallAssistantUI.ViewModels;

public class MainViewModel : ViewModelBase, IAsyncDisposable
{
    private readonly RestClient _restClient;
    private readonly WebSocketManager _webSocketManager;
    private readonly Dispatcher _dispatcher;
    private readonly Dictionary<string, QuestionViewModel> _questionLookup = new(StringComparer.OrdinalIgnoreCase);

    private readonly AsyncRelayCommand _startCallCommand;
    private readonly AsyncRelayCommand _endCallCommand;

    private CancellationTokenSource? _lifetimeCts;
    private bool _isInitialized;
    private bool _isBusy;
    private bool _isCallActive;
    private string _connectionStatusMessage = "Idle";
    private bool _disposed;

    public MainViewModel(RestClient restClient, WebSocketManager webSocketManager)
    {
        _restClient = restClient ?? throw new ArgumentNullException(nameof(restClient));
        _webSocketManager = webSocketManager ?? throw new ArgumentNullException(nameof(webSocketManager));

        _dispatcher = Application.Current.Dispatcher;

        Transcript = new TranscriptViewModel();
        Questions = new ObservableCollection<QuestionViewModel>();
        Status = new StatusViewModel();

        _startCallCommand = new AsyncRelayCommand(StartCallInternalAsync, CanStartCall);
        _endCallCommand = new AsyncRelayCommand(EndCallInternalAsync, CanEndCall);

        _webSocketManager.TranscriptReceived += OnTranscriptReceived;
        _webSocketManager.QuestionReceived += OnQuestionReceived;
        _webSocketManager.SuggestionReceived += OnSuggestionReceived;
        _webSocketManager.StatusReceived += OnStatusReceived;
        _webSocketManager.ConnectionStatusChanged += OnConnectionStateChanged;
    }

    public TranscriptViewModel Transcript { get; }

    public ObservableCollection<QuestionViewModel> Questions { get; }

    public StatusViewModel Status { get; }

    public ICommand StartCallCommand => _startCallCommand;

    public ICommand EndCallCommand => _endCallCommand;

    public string ConnectionStatusMessage
    {
        get => _connectionStatusMessage;
        set => SetProperty(ref _connectionStatusMessage, value);
    }

    public bool IsCallActive
    {
        get => _isCallActive;
        private set
        {
            if (SetProperty(ref _isCallActive, value))
            {
                UpdateCommandStates();
            }
        }
    }

    public async Task InitializeAsync()
    {
        if (_isInitialized)
        {
            return;
        }

        RunOnUi(() => ConnectionStatusMessage = "Connecting...");
        _lifetimeCts = new CancellationTokenSource();
        await _webSocketManager.StartAsync(_lifetimeCts.Token).ConfigureAwait(false);

        await LoadInitialTranscriptAsync().ConfigureAwait(false);

        _isInitialized = true;
    }

    private async Task LoadInitialTranscriptAsync()
    {
        try
        {
            RunOnUi(() => ConnectionStatusMessage = "Syncing transcript...");
            var transcript = await _restClient.GetCurrentTranscriptAsync().ConfigureAwait(false);
            RunOnUi(() => Transcript.SetText(transcript));
            RunOnUi(() => ConnectionStatusMessage = "Connected");
        }
        catch (Exception ex)
        {
            RunOnUi(() => ConnectionStatusMessage = $"Initial sync failed: {ex.Message}");
        }
    }

    private async Task StartCallInternalAsync()
    {
        if (_isBusy)
        {
            return;
        }

        _isBusy = true;
        UpdateCommandStates();

        try
        {
            RunOnUi(() => ConnectionStatusMessage = "Starting call...");
            var success = await _restClient.StartCallAsync().ConfigureAwait(false);
            if (success)
            {
                RunOnUi(() =>
                {
                    Transcript.Reset();
                    Status.Reset();
                    Questions.Clear();
                    _questionLookup.Clear();
                });

                RunOnUi(() => IsCallActive = true);
                RunOnUi(() => ConnectionStatusMessage = "Call live");
            }
            else
            {
                RunOnUi(() => ConnectionStatusMessage = "Start call failed");
            }
        }
        catch (Exception ex)
        {
            RunOnUi(() => ConnectionStatusMessage = $"Start call error: {ex.Message}");
        }
        finally
        {
            _isBusy = false;
            UpdateCommandStates();
        }
    }

    private async Task EndCallInternalAsync()
    {
        if (_isBusy)
        {
            return;
        }

        _isBusy = true;
        UpdateCommandStates();

        try
        {
            RunOnUi(() => ConnectionStatusMessage = "Ending call...");
            var success = await _restClient.EndCallAsync().ConfigureAwait(false);
            RunOnUi(() => ConnectionStatusMessage = success ? "Call ended" : "End call failed");
            RunOnUi(() => IsCallActive = false);
        }
        catch (Exception ex)
        {
            RunOnUi(() => ConnectionStatusMessage = $"End call error: {ex.Message}");
        }
        finally
        {
            _isBusy = false;
            UpdateCommandStates();
        }
    }

    private bool CanStartCall() => !_isBusy && !IsCallActive;

    private bool CanEndCall() => !_isBusy && IsCallActive;

    private void UpdateCommandStates() =>
        RunOnUi(() =>
        {
            _startCallCommand.RaiseCanExecuteChanged();
            _endCallCommand.RaiseCanExecuteChanged();
        });

    private void OnTranscriptReceived(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return;
        }

        RunOnUi(() => Transcript.AppendText(text));
    }

    private void OnQuestionReceived(Question question)
    {
        if (question is null || string.IsNullOrWhiteSpace(question.Text))
        {
            return;
        }

        RunOnUi(() =>
        {
            var viewModel = new QuestionViewModel
            {
                QuestionText = question.Text.Trim(),
                Timestamp = question.Timestamp
            };

            Questions.Insert(0, viewModel);
            _questionLookup[viewModel.QuestionText] = viewModel;
        });
    }

    private void OnSuggestionReceived(Suggestion suggestion)
    {
        if (suggestion is null || string.IsNullOrWhiteSpace(suggestion.Question))
        {
            return;
        }

        RunOnUi(() =>
        {
            if (!_questionLookup.TryGetValue(suggestion.Question.Trim(), out var viewModel))
            {
                viewModel = new QuestionViewModel
                {
                    QuestionText = suggestion.Question.Trim()
                };
                Questions.Insert(0, viewModel);
                _questionLookup[viewModel.QuestionText] = viewModel;
            }

            viewModel.SetSuggestions(suggestion.Items);
        });
    }

    private void OnStatusReceived(StatusInfo status)
    {
        if (status is null)
        {
            return;
        }

        RunOnUi(() => Status.Update(status));
    }

    private void OnConnectionStateChanged(string message)
    {
        RunOnUi(() => ConnectionStatusMessage = message);
    }

    private void RunOnUi(Action action)
    {
        if (action == null)
        {
            return;
        }

        if (_dispatcher.CheckAccess())
        {
            action();
        }
        else
        {
            _dispatcher.Invoke(action);
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        _webSocketManager.TranscriptReceived -= OnTranscriptReceived;
        _webSocketManager.QuestionReceived -= OnQuestionReceived;
        _webSocketManager.SuggestionReceived -= OnSuggestionReceived;
        _webSocketManager.StatusReceived -= OnStatusReceived;
        _webSocketManager.ConnectionStatusChanged -= OnConnectionStateChanged;

        if (_lifetimeCts is not null)
        {
            await _webSocketManager.StopAsync().ConfigureAwait(false);
            _lifetimeCts.Cancel();
            _lifetimeCts.Dispose();
        }

        _restClient.Dispose();
        await _webSocketManager.DisposeAsync().ConfigureAwait(false);
    }
}
