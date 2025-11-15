using System;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace CallAssistantUI.Services;

public class RestClient : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    public RestClient(string baseAddress)
    {
        if (string.IsNullOrWhiteSpace(baseAddress))
        {
            throw new ArgumentException("Base address must be provided", nameof(baseAddress));
        }

        if (!baseAddress.EndsWith("/"))
        {
            baseAddress += "/";
        }

        _httpClient = new HttpClient
        {
            BaseAddress = new Uri(baseAddress, UriKind.Absolute),
            Timeout = TimeSpan.FromSeconds(10)
        };
    }

    public async Task<bool> StartCallAsync(CancellationToken cancellationToken = default)
    {
        return await SendPostAsync("start_call", cancellationToken).ConfigureAwait(false);
    }

    public async Task<bool> EndCallAsync(CancellationToken cancellationToken = default)
    {
        return await SendPostAsync("end_call", cancellationToken).ConfigureAwait(false);
    }

    public async Task<string> GetCurrentTranscriptAsync(CancellationToken cancellationToken = default)
    {
        using var response = await _httpClient.GetAsync("current_transcript", cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        var json = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
        if (string.IsNullOrWhiteSpace(json))
        {
            return string.Empty;
        }

        using var document = JsonDocument.Parse(json);
        if (document.RootElement.TryGetProperty("transcript", out var transcriptElement))
        {
            return transcriptElement.GetString() ?? string.Empty;
        }

        return string.Empty;
    }

    private async Task<bool> SendPostAsync(string relativePath, CancellationToken cancellationToken)
    {
        using var response = await _httpClient.PostAsync(relativePath, content: null, cancellationToken).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
        {
            return false;
        }

        var json = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
        if (string.IsNullOrWhiteSpace(json))
        {
            return response.IsSuccessStatusCode;
        }

        try
        {
            using var document = JsonDocument.Parse(json);
            if (document.RootElement.TryGetProperty("status", out var statusElement))
            {
                return string.Equals(statusElement.GetString(), "ok", StringComparison.OrdinalIgnoreCase);
            }
        }
        catch (JsonException)
        {
            // Ignore malformed payloads, rely on HTTP status.
        }

        return response.IsSuccessStatusCode;
    }

    public void Dispose()
    {
        _httpClient.Dispose();
    }
}
