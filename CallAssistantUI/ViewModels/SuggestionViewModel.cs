namespace CallAssistantUI.ViewModels;

public class SuggestionViewModel : ViewModelBase
{
    private string _text = string.Empty;

    public SuggestionViewModel()
    {
    }

    public SuggestionViewModel(string text)
    {
        _text = text;
    }

    public string Text
    {
        get => _text;
        set => SetProperty(ref _text, value);
    }
}
