using CallAssistantUI.Models;

namespace CallAssistantUI.ViewModels;

public class StatusViewModel : ViewModelBase
{
    private int _asrLatencyMs;
    private int _queueDepth;
    private int _errors;
    private bool _processing;

    public int AsrLatencyMs
    {
        get => _asrLatencyMs;
        set => SetProperty(ref _asrLatencyMs, value);
    }

    public int QueueDepth
    {
        get => _queueDepth;
        set => SetProperty(ref _queueDepth, value);
    }

    public int Errors
    {
        get => _errors;
        set
        {
            if (SetProperty(ref _errors, value))
            {
                RaisePropertyChanged(nameof(HasError));
            }
        }
    }

    public bool Processing
    {
        get => _processing;
        set
        {
            if (SetProperty(ref _processing, value))
            {
                RaisePropertyChanged(nameof(ProcessingText));
            }
        }
    }

    public bool HasError => Errors > 0;

    public string ProcessingText => Processing ? "Yes" : "No";

    public void Update(StatusInfo info)
    {
        if (info is null)
        {
            return;
        }

        AsrLatencyMs = info.AsrLatencyMs;
        QueueDepth = info.QueueDepth;
        Errors = info.Errors;
        Processing = info.Processing;
    }

    public void Reset()
    {
        AsrLatencyMs = 0;
        QueueDepth = 0;
        Errors = 0;
        Processing = false;
    }
}
