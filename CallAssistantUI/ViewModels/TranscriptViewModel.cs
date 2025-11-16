namespace CallAssistantUI.ViewModels;

public class TranscriptViewModel : ViewModelBase
{
    private string _fullText = string.Empty;

    public string FullText
    {
        get => _fullText;
        set => SetProperty(ref _fullText, value);
    }

    public void Reset()
    {
        FullText = string.Empty;
    }

    public void SetText(string text)
    {
        FullText = text ?? string.Empty;
    }

    public void AppendText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return;
        }

        if (string.IsNullOrEmpty(FullText))
        {
            FullText = text.Trim();
        }
        else
        {
            FullText = $"{FullText}\n{text.Trim()}";
        }
    }
}
