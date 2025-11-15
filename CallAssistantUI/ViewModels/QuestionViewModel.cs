using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace CallAssistantUI.ViewModels;

public class QuestionViewModel : ViewModelBase
{
    private string _questionText = string.Empty;
    private DateTimeOffset _timestamp = DateTimeOffset.MinValue;

    public QuestionViewModel()
    {
        Suggestions = new ObservableCollection<SuggestionViewModel>();
    }

    public string QuestionText
    {
        get => _questionText;
        set
        {
            if (SetProperty(ref _questionText, value))
            {
                RaisePropertyChanged(nameof(HasQuestion));
            }
        }
    }

    public DateTimeOffset Timestamp
    {
        get => _timestamp;
        set
        {
            if (SetProperty(ref _timestamp, value))
            {
                RaisePropertyChanged(nameof(TimestampDisplay));
            }
        }
    }

    public string TimestampDisplay => Timestamp == DateTimeOffset.MinValue
        ? string.Empty
        : Timestamp.ToLocalTime().ToString("MMM dd, HH:mm:ss");

    public bool HasQuestion => !string.IsNullOrWhiteSpace(QuestionText);

    public ObservableCollection<SuggestionViewModel> Suggestions { get; }

    public void SetSuggestions(IEnumerable<string> suggestions)
    {
        var normalized = suggestions?.Where(s => !string.IsNullOrWhiteSpace(s)).ToList()
                         ?? new List<string>();

        Suggestions.Clear();
        foreach (var suggestion in normalized)
        {
            Suggestions.Add(new SuggestionViewModel(suggestion.Trim()));
        }

        RaisePropertyChanged(nameof(Suggestions));
    }
}
