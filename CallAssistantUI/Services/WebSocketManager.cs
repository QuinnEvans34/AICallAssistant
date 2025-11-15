using System;
using System.Collections.Generic;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using CallAssistantUI.Models;

namespace CallAssistantUI.Services;

public class WebSocketManager : IAsyncDisposable, IDisposable
{
    private readonly Uri _baseUri;
    private readonly JsonSerializerOptions _serializerOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    private readonly TimeSpan _initialReconnectDelay = TimeSpan.FromSeconds(1);
    private readonly TimeSpan _maxReconnectDelay = TimeSpan.FromSeconds(30);

    private readonly List<Task> _runningTasks = new();
    private CancellationTokenSource? _cts;

    public event Action<string>? TranscriptReceived;
    public event Action<Question>? QuestionReceived;
    public event Action<Suggestion>? SuggestionReceived;
    public event Action<StatusInfo>? StatusReceived;
    public event Action<string>? ConnectionStatusChanged;

    public WebSocketManager(Uri baseUri)
    {
        _baseUri = baseUri ?? throw new ArgumentNullException(nameof(baseUri));
        if (!_baseUri.IsAbsoluteUri)
        {
            throw new ArgumentException("WebSocket base URI must be absolute", nameof(baseUri));
        }
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        if (_cts is not null)
        {
            return Task.CompletedTask;
        }

        _cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var token = _cts.Token;

        _runningTasks.Add(Task.Run(() => ListenLoopAsync("/ws/transcript", HandleTranscriptMessageAsync, token), token));
        _runningTasks.Add(Task.Run(() => ListenLoopAsync("/ws/questions", HandleQuestionMessageAsync, token), token));
        _runningTasks.Add(Task.Run(() => ListenLoopAsync("/ws/suggestions", HandleSuggestionMessageAsync, token), token));
        _runningTasks.Add(Task.Run(() => ListenLoopAsync("/ws/status", HandleStatusMessageAsync, token), token));

        return Task.CompletedTask;
    }

    public async Task StopAsync()
    {
        if (_cts is null)
        {
            return;
        }

        _cts.Cancel();

        try
        {
            await Task.WhenAll(_runningTasks).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            // expected on shutdown
        }

        _runningTasks.Clear();
        _cts.Dispose();
        _cts = null;
    }

    private async Task ListenLoopAsync(string relativePath, Func<string, Task> handler, CancellationToken token)
    {
        var delay = _initialReconnectDelay;
        while (!token.IsCancellationRequested)
        {
            using var client = new ClientWebSocket();
            var endpoint = new Uri(_baseUri, relativePath);
            try
            {
                await client.ConnectAsync(endpoint, token).ConfigureAwait(false);
                ConnectionStatusChanged?.Invoke($"{relativePath} connected");
                delay = _initialReconnectDelay;

                await ReceiveLoopAsync(client, handler, token).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (token.IsCancellationRequested)
            {
                break;
            }
            catch (Exception ex) when (!token.IsCancellationRequested)
            {
                ConnectionStatusChanged?.Invoke($"{relativePath} disconnected: {ex.Message}");
            }
            finally
            {
                if (client.State == WebSocketState.Open || client.State == WebSocketState.CloseReceived)
                {
                    try
                    {
                        await client.CloseAsync(WebSocketCloseStatus.NormalClosure, "shutdown", CancellationToken.None)
                                    .ConfigureAwait(false);
                    }
                    catch
                    {
                        // ignore shutdown exceptions
                    }
                }
            }

            try
            {
                await Task.Delay(delay, token).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }

            var nextDelayMilliseconds = Math.Min(delay.TotalMilliseconds * 2, _maxReconnectDelay.TotalMilliseconds);
            delay = TimeSpan.FromMilliseconds(nextDelayMilliseconds);
        }
    }

    private static async Task ReceiveLoopAsync(ClientWebSocket client, Func<string, Task> handler, CancellationToken token)
    {
        var buffer = new byte[4096];

        while (!token.IsCancellationRequested && client.State == WebSocketState.Open)
        {
            var builder = new StringBuilder();
            WebSocketReceiveResult? result;

            do
            {
                result = await client.ReceiveAsync(new ArraySegment<byte>(buffer), token).ConfigureAwait(false);
                if (result.MessageType == WebSocketMessageType.Close)
                {
                    await client.CloseAsync(WebSocketCloseStatus.NormalClosure, "closing", CancellationToken.None)
                                .ConfigureAwait(false);
                    return;
                }

                if (result.Count > 0)
                {
                    builder.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));
                }
            } while (!result.EndOfMessage);

            var payload = builder.ToString();
            if (!string.IsNullOrWhiteSpace(payload))
            {
                await handler(payload).ConfigureAwait(false);
            }
        }
    }

    private Task HandleTranscriptMessageAsync(string payload)
    {
        TranscriptReceived?.Invoke(payload.Trim());
        return Task.CompletedTask;
    }

    private Task HandleQuestionMessageAsync(string payload)
    {
        try
        {
            var question = JsonSerializer.Deserialize<Question>(payload, _serializerOptions);
            if (question is not null)
            {
                QuestionReceived?.Invoke(question);
            }
        }
        catch (JsonException)
        {
            ConnectionStatusChanged?.Invoke("Invalid question payload");
        }

        return Task.CompletedTask;
    }

    private Task HandleSuggestionMessageAsync(string payload)
    {
        try
        {
            var suggestion = JsonSerializer.Deserialize<Suggestion>(payload, _serializerOptions);
            if (suggestion is not null)
            {
                SuggestionReceived?.Invoke(suggestion);
            }
        }
        catch (JsonException)
        {
            ConnectionStatusChanged?.Invoke("Invalid suggestion payload");
        }

        return Task.CompletedTask;
    }

    private Task HandleStatusMessageAsync(string payload)
    {
        try
        {
            var status = JsonSerializer.Deserialize<StatusInfo>(payload, _serializerOptions);
            if (status is not null)
            {
                StatusReceived?.Invoke(status);
            }
        }
        catch (JsonException)
        {
            ConnectionStatusChanged?.Invoke("Invalid status payload");
        }

        return Task.CompletedTask;
    }

    public async ValueTask DisposeAsync()
    {
        await StopAsync().ConfigureAwait(false);
    }

    public void Dispose()
    {
        _ = StopAsync();
    }
}
