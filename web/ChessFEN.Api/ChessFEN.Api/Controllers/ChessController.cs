using Microsoft.AspNetCore.Mvc;
using ChessFEN.Api.Models;
using System.Text;
using System.Text.Json;

namespace ChessFEN.Api.Controllers;

[ApiController]
[Route("api/[controller]")]
public class ChessController : ControllerBase
{
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly ILogger<ChessController> _logger;
    private readonly IConfiguration _configuration;

    public ChessController(
        IHttpClientFactory httpClientFactory, 
        ILogger<ChessController> logger,
        IConfiguration configuration)
    {
        _httpClientFactory = httpClientFactory;
        _logger = logger;
        _configuration = configuration;
    }

    /// <summary>
    /// Predict FEN from uploaded chess board image
    /// </summary>
    /// <param name="image">Image file (JPEG/PNG)</param>
    /// <param name="debug">Include debug images in response</param>
    /// <param name="model">Model name to use (optional)</param>
    /// <param name="skipDetection">Skip board detection, treat entire image as board</param>
    [HttpPost("predict")]
    [Consumes("multipart/form-data")]
    public async Task<ActionResult<PredictionResponse>> Predict(
        IFormFile image,
        [FromQuery] bool debug = false,
        [FromQuery] string? model = null,
        [FromQuery] bool skipDetection = false)
    {
        if (image == null || image.Length == 0)
        {
            return BadRequest(new PredictionResponse 
            { 
                Success = false, 
                Error = "No image provided" 
            });
        }

        try
        {
            var client = _httpClientFactory.CreateClient("InferenceService");
            
            // Build query string
            var queryParams = new List<string>();
            if (debug) queryParams.Add("debug=true");
            if (!string.IsNullOrEmpty(model)) queryParams.Add($"model={model}");
            if (skipDetection) queryParams.Add("skip_detection=true");
            var query = queryParams.Count > 0 ? "?" + string.Join("&", queryParams) : "";

            // Send image to Python service
            using var content = new MultipartFormDataContent();
            using var stream = image.OpenReadStream();
            using var streamContent = new StreamContent(stream);
            content.Add(streamContent, "image", image.FileName);

            var response = await client.PostAsync($"/api/predict{query}", content);
            var json = await response.Content.ReadAsStringAsync();

            var result = JsonSerializer.Deserialize<PredictionResponse>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            return Ok(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calling inference service");
            return StatusCode(500, new PredictionResponse 
            { 
                Success = false, 
                Error = "Inference service error: " + ex.Message 
            });
        }
    }

    /// <summary>
    /// Predict FEN from base64 encoded image
    /// </summary>
    [HttpPost("predict/base64")]
    public async Task<ActionResult<PredictionResponse>> PredictBase64(
        [FromBody] Dictionary<string, string> body,
        [FromQuery] bool debug = false,
        [FromQuery] string? model = null,
        [FromQuery] bool skipDetection = false)
    {
        if (!body.TryGetValue("image", out var imageBase64) || string.IsNullOrEmpty(imageBase64))
        {
            return BadRequest(new PredictionResponse 
            { 
                Success = false, 
                Error = "No image provided" 
            });
        }

        try
        {
            var client = _httpClientFactory.CreateClient("InferenceService");
            
            var queryParams = new List<string>();
            if (debug) queryParams.Add("debug=true");
            if (!string.IsNullOrEmpty(model)) queryParams.Add($"model={model}");
            if (skipDetection) queryParams.Add("skip_detection=true");
            var query = queryParams.Count > 0 ? "?" + string.Join("&", queryParams) : "";

            var payload = JsonSerializer.Serialize(new { image = imageBase64 });
            var content = new StringContent(payload, Encoding.UTF8, "application/json");

            var response = await client.PostAsync($"/api/predict{query}", content);
            var json = await response.Content.ReadAsStringAsync();

            var result = JsonSerializer.Deserialize<PredictionResponse>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            return Ok(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calling inference service");
            return StatusCode(500, new PredictionResponse 
            { 
                Success = false, 
                Error = "Inference service error: " + ex.Message 
            });
        }
    }

    /// <summary>
    /// Get available model versions
    /// </summary>
    [HttpGet("models")]
    public async Task<ActionResult<List<ModelVersion>>> GetModels()
    {
        try
        {
            var client = _httpClientFactory.CreateClient("InferenceService");
            var response = await client.GetAsync("/api/models");
            var json = await response.Content.ReadAsStringAsync();

            var models = JsonSerializer.Deserialize<List<ModelVersion>>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            return Ok(models);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting models");
            return StatusCode(500, new { error = ex.Message });
        }
    }

    /// <summary>
    /// Select model to use for inference
    /// </summary>
    [HttpPost("models/select")]
    public async Task<ActionResult> SelectModel([FromBody] SelectModelRequest request)
    {
        try
        {
            var client = _httpClientFactory.CreateClient("InferenceService");
            var payload = JsonSerializer.Serialize(new { model_name = request.ModelName });
            var content = new StringContent(payload, Encoding.UTF8, "application/json");

            var response = await client.PostAsync("/api/models/select", content);
            var json = await response.Content.ReadAsStringAsync();

            return Ok(JsonSerializer.Deserialize<object>(json));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error selecting model");
            return StatusCode(500, new { error = ex.Message });
        }
    }

    /// <summary>
    /// Submit user correction feedback
    /// </summary>
    [HttpPost("feedback")]
    public async Task<ActionResult<FeedbackResponse>> SubmitFeedback([FromBody] FeedbackRequest request)
    {
        try
        {
            var client = _httpClientFactory.CreateClient("InferenceService");
            
            var payload = JsonSerializer.Serialize(new
            {
                original_fen = request.OriginalFen,
                corrected_fen = request.CorrectedFen,
                image = request.Image,
                corrected_squares = request.CorrectedSquares
            });
            var content = new StringContent(payload, Encoding.UTF8, "application/json");

            var response = await client.PostAsync("/api/feedback", content);
            var json = await response.Content.ReadAsStringAsync();

            var result = JsonSerializer.Deserialize<FeedbackResponse>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            return Ok(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error submitting feedback");
            return StatusCode(500, new FeedbackResponse 
            { 
                Success = false, 
                Error = ex.Message 
            });
        }
    }

    /// <summary>
    /// Get pending feedback items (admin endpoint)
    /// </summary>
    [HttpGet("feedback")]
    public async Task<ActionResult<List<FeedbackItem>>> GetFeedback()
    {
        try
        {
            var client = _httpClientFactory.CreateClient("InferenceService");
            var response = await client.GetAsync("/api/feedback");
            var json = await response.Content.ReadAsStringAsync();

            var items = JsonSerializer.Deserialize<List<FeedbackItem>>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            return Ok(items);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting feedback");
            return StatusCode(500, new { error = ex.Message });
        }
    }

    /// <summary>
    /// Check if games exist on Chess.com for a given FEN
    /// </summary>
    [HttpGet("chess-com-search")]
    public async Task<ActionResult<ChessComSearchResult>> SearchChessCom([FromQuery] string fen)
    {
        if (string.IsNullOrEmpty(fen))
        {
            return BadRequest(new ChessComSearchResult { Success = false, Error = "FEN is required" });
        }

        try
        {
            using var httpClient = new HttpClient();
            // Use a browser-like User-Agent to get the same response as a browser
            httpClient.DefaultRequestHeaders.Add("User-Agent", 
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36");
            httpClient.DefaultRequestHeaders.Add("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8");
            httpClient.DefaultRequestHeaders.Add("Accept-Language", "en-US,en;q=0.5");
            
            var encodedFen = Uri.EscapeDataString(fen);
            var url = $"https://www.chess.com/games/search?fen={encodedFen}";
            
            var response = await httpClient.GetAsync(url);
            var html = await response.Content.ReadAsStringAsync();
            
            // Primary check: explicit "no games" message - this is the most reliable indicator
            var noGamesIndicators = new[]
            {
                "Your search did not match any games",
                "did not match any games"
            };
            
            var hasNoGamesMessage = noGamesIndicators.Any(indicator => 
                html.Contains(indicator, StringComparison.OrdinalIgnoreCase));
            
            // If we have a clear "no games" message, trust it
            if (hasNoGamesMessage)
            {
                _logger.LogInformation("Chess.com search: No games found (explicit message) for FEN: {Fen}", 
                    fen.Length > 30 ? fen.Substring(0, 30) + "..." : fen);
                    
                return Ok(new ChessComSearchResult
                {
                    Success = true,
                    GamesFound = false,
                    SearchUrl = url,
                    Message = "No games found for this position."
                });
            }
            
            // Secondary check: look for actual game data
            // These are specific to actual game listings
            var gamesFoundIndicators = new[]
            {
                "master-games-master-game",   // Master games table row
                "master-games-username",      // Player name in master games
                "archived-games-game-row",    // Archived game row
                "/game/live/",               // Link to a live game
                "/game/daily/",              // Link to a daily game
            };
            
            var hasGamesIndicator = gamesFoundIndicators.Any(indicator => 
                html.Contains(indicator, StringComparison.OrdinalIgnoreCase));
            
            _logger.LogInformation("Chess.com search for FEN: {Fen}, GamesFound: {GamesInd}", 
                fen.Length > 30 ? fen.Substring(0, 30) + "..." : fen, hasGamesIndicator);
            
            return Ok(new ChessComSearchResult
            {
                Success = true,
                GamesFound = hasGamesIndicator,
                SearchUrl = url,
                Message = hasGamesIndicator ? "Games found on Chess.com!" : "No games found for this position."
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error searching Chess.com");
            return Ok(new ChessComSearchResult
            {
                Success = false,
                GamesFound = false,
                Error = ex.Message
            });
        }
    }

    /// <summary>
    /// Health check
    /// </summary>
    [HttpGet("health")]
    public async Task<ActionResult> Health()
    {
        try
        {
            var client = _httpClientFactory.CreateClient("InferenceService");
            var response = await client.GetAsync("/api/health");
            var json = await response.Content.ReadAsStringAsync();
            return Ok(JsonSerializer.Deserialize<object>(json));
        }
        catch (Exception ex)
        {
            return StatusCode(503, new { status = "unhealthy", error = ex.Message });
        }
    }

    /// <summary>
    /// Analyze a warped board image to detect piece positions and suggest grid adjustments.
    /// Used when automatic board detection produces misaligned grids.
    /// </summary>
    [HttpPost("align")]
    public async Task<ActionResult<GridAlignmentResult>> AlignGrid([FromBody] JsonElement body)
    {
        string? imageBase64 = null;
        bool skipDetection = false;
        
        if (body.TryGetProperty("image", out var imageElement))
        {
            imageBase64 = imageElement.GetString();
        }
        
        if (body.TryGetProperty("skip_detection", out var skipElement))
        {
            skipDetection = skipElement.GetBoolean();
        }
        
        if (string.IsNullOrEmpty(imageBase64))
        {
            return BadRequest(new GridAlignmentResult 
            { 
                Success = false, 
                Error = "No image provided" 
            });
        }

        try
        {
            var client = _httpClientFactory.CreateClient("InferenceService");
            
            var payload = JsonSerializer.Serialize(new { image = imageBase64, skip_detection = skipDetection });
            var content = new StringContent(payload, Encoding.UTF8, "application/json");

            var response = await client.PostAsync("/api/align", content);
            var json = await response.Content.ReadAsStringAsync();

            var result = JsonSerializer.Deserialize<GridAlignmentResult>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            return Ok(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calling alignment service");
            return StatusCode(500, new GridAlignmentResult 
            { 
                Success = false, 
                Error = "Alignment service error: " + ex.Message 
            });
        }
    }
}

/// <summary>
/// Result from Chess.com game search
/// </summary>
public class ChessComSearchResult
{
    public bool Success { get; set; }
    public bool GamesFound { get; set; }
    public string? SearchUrl { get; set; }
    public string? Message { get; set; }
    public string? Error { get; set; }
}

/// <summary>
/// Result from grid alignment analysis
/// </summary>
public class GridAlignmentResult
{
    public bool Success { get; set; }
    public bool BoardDetected { get; set; }
    public bool PieceFound { get; set; }
    public List<double>? PieceCenter { get; set; }
    public List<int>? PieceSquare { get; set; }
    public string? SquareName { get; set; }
    public List<double>? Offset { get; set; }
    public double Confidence { get; set; }
    public bool Aligned { get; set; }
    public string? Suggestion { get; set; }
    public string? Overlay { get; set; }
    public string? Error { get; set; }
}
