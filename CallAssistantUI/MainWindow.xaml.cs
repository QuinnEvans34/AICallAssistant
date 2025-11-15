using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using CallAssistantUI.ViewModels;

namespace CallAssistantUI;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
    }

    private async void MainWindow_OnLoaded(object sender, RoutedEventArgs e)
    {
        if (DataContext is MainViewModel vm)
        {
            await vm.InitializeAsync();
        }
    }

    private void TranscriptTextBox_OnTextChanged(object sender, TextChangedEventArgs e)
    {
        if (sender is TextBox textBox)
        {
            textBox.ScrollToEnd();
        }
    }

    protected override async void OnClosed(System.EventArgs e)
    {
        if (DataContext is MainViewModel vm)
        {
            await vm.DisposeAsync();
        }

        base.OnClosed(e);
    }
}
