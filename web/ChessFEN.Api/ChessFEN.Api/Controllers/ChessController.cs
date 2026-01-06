using Microsoft.AspNetCore.Mvc;
using ChessFEN.Api.Models;
using ChessFEN.Api.Services;
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
    private readonly AuthorizationHelper _authHelper;

    public ChessController(
        IHttpClientFactory httpClientFactory,
        ILogger<ChessController> logger,
        IConfiguration configuration,
        AuthorizationHelper authHelper)
    {
        _httpClientFactory = httpClientFactory;
        _logger = logger;
        _configuration = configuration;
        _authHelper = authHelper;
    }

    /// <summary>
    /// Check if user is authorized to access this endpoint
    /// </summary>
    private ActionResult? CheckAuthorization(bool requireAdmin = false)
    {
        var userContext = _authHelper.GetUserContext(Request);

        if (!userContext.IsAuthenticated)
        {
            _logger.LogWarning("Unauthorized access attempt - not authenticated");
            return Unauthorized(new { error = "Authentication required", message = userContext.Message });
        }

        if (!userContext.HasAccess)
        {
            _logger.LogWarning("Forbidden access attempt - User: {Email}", userContext.Email);
            return StatusCode(403, new { error = "Access denied", message = userContext.Message });
        }

        if (requireAdmin && !userContext.IsAdmin)
        {
            _logger.LogWarning("Admin access denied - User: {Email}", userContext.Email);
            return StatusCode(403, new { error = "Admin access required" });
        }

        return null; // Authorized
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
        [FromQuery(Name = "skip_detection")] bool skipDetection = false)
    {
        // Check authorization
        var authResult = CheckAuthorization();
        if (authResult != null) return authResult;

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

            var response = await client.PostAsync($"/api/chess/predict{query}", content);
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
        [FromQuery(Name = "skip_detection")] bool skipDetection = false)
    {
        // Check authorization
        var authResult = CheckAuthorization();
        if (authResult != null) return authResult;

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

            var response = await client.PostAsync($"/api/chess/predict{query}", content);
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
    /// Deep analysis: Run all models and check for games on Chess.com
    /// </summary>
    [HttpPost("analyze/deep")]
    public async Task<ActionResult<DeepAnalysisResponse>> DeepAnalysis(
        [FromBody] Dictionary<string, string> body,
        [FromQuery(Name = "skip_detection")] bool skipDetection = false)
    {
        if (!body.TryGetValue("image", out var imageBase64) || string.IsNullOrEmpty(imageBase64))
        {
            return BadRequest(new DeepAnalysisResponse
            {
                Success = false,
                Error = "No image provided"
            });
        }

        try
        {
            var client = _httpClientFactory.CreateClient("InferenceService");
            
            var query = skipDetection ? "?skip_detection=true" : "";
            var payload = JsonSerializer.Serialize(new { image = imageBase64 });
            var content = new StringContent(payload, Encoding.UTF8, "application/json");

            // Call multi-model prediction
            var response = await client.PostAsync($"/api/chess/predict/multi{query}", content);
            var json = await response.Content.ReadAsStringAsync();

            var multiResult = JsonSerializer.Deserialize<JsonElement>(json);
            
            var result = new DeepAnalysisResponse
            {
                Success = multiResult.GetProperty("success").GetBoolean(),
                Consensus = multiResult.TryGetProperty("consensus", out var c) && c.GetBoolean(),
                ConsensusFen = multiResult.TryGetProperty("consensus_fen", out var cf) ? cf.GetString() : null,
                ModelResults = new List<ModelPrediction>(),
                Disagreements = new List<SquareDisagreement>()
            };

            // Parse model results
            if (multiResult.TryGetProperty("results", out var results))
            {
                foreach (var r in results.EnumerateArray())
                {
                    result.ModelResults.Add(new ModelPrediction
                    {
                        Model = r.TryGetProperty("model", out var m) ? m.GetString() : null,
                        Type = r.TryGetProperty("type", out var t) ? t.GetString() : null,
                        Accuracy = r.TryGetProperty("accuracy", out var a) ? a.GetString() : null,
                        Success = r.TryGetProperty("success", out var s) && s.GetBoolean(),
                        Fen = r.TryGetProperty("fen", out var f) ? f.GetString() : null,
                        Error = r.TryGetProperty("error", out var e) ? e.GetString() : null
                    });
                }
            }

            // Parse disagreements
            if (multiResult.TryGetProperty("disagreements", out var disagreements))
            {
                foreach (var d in disagreements.EnumerateArray())
                {
                    var dis = new SquareDisagreement
                    {
                        Square = d.TryGetProperty("square", out var sq) ? sq.GetString() : null,
                        Predictions = new Dictionary<string, string>()
                    };
                    if (d.TryGetProperty("predictions", out var preds))
                    {
                        foreach (var p in preds.EnumerateObject())
                        {
                            dis.Predictions[p.Name] = p.Value.GetString() ?? "";
                        }
                    }
                    result.Disagreements.Add(dis);
                }
            }

            // Get recommended FEN (consensus or best model)
            if (result.Consensus && result.ConsensusFen != null)
            {
                result.RecommendedFen = result.ConsensusFen;
            }
            else
            {
                // Use the FEN from the "best" model (one with highest accuracy or "best" in name)
                var bestModel = result.ModelResults?
                    .Where(r => r.Success && r.Fen != null)
                    .OrderByDescending(r => r.Model?.Contains("best") == true)
                    .ThenByDescending(r => r.Accuracy)
                    .FirstOrDefault();
                result.RecommendedFen = bestModel?.Fen;
            }

            // Check Chess.com for games (using recommended FEN)
            if (!string.IsNullOrEmpty(result.RecommendedFen))
            {
                var fenPart = result.RecommendedFen;
                var whiteFen = $"{fenPart} w KQkq - 0 1";
                var blackFen = $"{fenPart} b KQkq - 0 1";

                var whiteCheck = await CheckChessComGamesInternal(whiteFen);
                var blackCheck = await CheckChessComGamesInternal(blackFen);

                result.GamesFoundWhite = whiteCheck;
                result.GamesFoundBlack = blackCheck;

                if (whiteCheck && blackCheck)
                    result.GamesMessage = "✓ Games found for both sides!";
                else if (whiteCheck)
                    result.GamesMessage = "✓ Games found (White to move)";
                else if (blackCheck)
                    result.GamesMessage = "✓ Games found (Black to move)";
                else
                    result.GamesMessage = "No games found on Chess.com";
            }

            return Ok(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in deep analysis");
            return StatusCode(500, new DeepAnalysisResponse
            {
                Success = false,
                Error = "Deep analysis error: " + ex.Message
            });
        }
    }

    /// <summary>
    /// Internal helper to check Chess.com for games
    /// </summary>
    private async Task<bool> CheckChessComGamesInternal(string fen)
    {
        try
        {
            var encodedFen = Uri.EscapeDataString(fen);
            var url = $"https://www.chess.com/games/search?fen={encodedFen}";

            var httpClient = _httpClientFactory.CreateClient();
            httpClient.DefaultRequestHeaders.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36");
            httpClient.Timeout = TimeSpan.FromSeconds(10);

            var response = await httpClient.GetAsync(url);
            var html = await response.Content.ReadAsStringAsync();

            // Check for indicators of games found
            bool hasNoGamesMessage = html.Contains("No results found") || 
                                     html.Contains("no games found", StringComparison.OrdinalIgnoreCase);
            
            bool hasGamesIndicator = !hasNoGamesMessage && 
                                     (html.Contains("games-archive-table") || 
                                      html.Contains("game-row") ||
                                      html.Contains("master-games-component-row"));

            return hasGamesIndicator;
        }
        catch
        {
            return false;
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
            var response = await client.GetAsync("/api/chess/models");
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
    /// Select model to use for inference (admin only)
    /// </summary>
    [HttpPost("models/select")]
    public async Task<ActionResult> SelectModel([FromBody] SelectModelRequest request)
    {
        // Check admin authorization
        var authResult = CheckAuthorization(requireAdmin: true);
        if (authResult != null) return authResult;

        try
        {
            var client = _httpClientFactory.CreateClient("InferenceService");
            var payload = JsonSerializer.Serialize(new { model_name = request.ModelName });
            var content = new StringContent(payload, Encoding.UTF8, "application/json");

            var response = await client.PostAsync("/api/chess/models/select", content);
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
    public async Task<ActionResult<FeedbackResponse>> SubmitFeedback([FromBody] FeedbackRequest? request)
    {
        // Check authorization
        var authResult = CheckAuthorization();
        if (authResult != null) return authResult;

        if (request == null)
        {
            return BadRequest(new FeedbackResponse { Success = false, Error = "Request body is required" });
        }

        try
        {
            // Get user context to include in feedback
            var userContext = _authHelper.GetUserContext(Request);

            var client = _httpClientFactory.CreateClient("InferenceService");

            var payload = JsonSerializer.Serialize(new
            {
                original_fen = request.OriginalFen,
                corrected_fen = request.CorrectedFen,
                image = request.Image,
                corrected_squares = request.CorrectedSquares,
                user_id = userContext.UserId,
                user_email = userContext.Email
            });
            var content = new StringContent(payload, Encoding.UTF8, "application/json");

            var response = await client.PostAsync("/api/chess/feedback", content);
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
        // Check admin authorization
        var authResult = CheckAuthorization(requireAdmin: true);
        if (authResult != null) return authResult;

        try
        {
            var client = _httpClientFactory.CreateClient("InferenceService");
            var response = await client.GetAsync("/api/chess/feedback");
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
    /// Get user statistics (admin endpoint)
    /// </summary>
    [HttpGet("admin/statistics")]
    public async Task<ActionResult<AdminStatisticsResponse>> GetStatistics()
    {
        // Check admin authorization
        var authResult = CheckAuthorization(requireAdmin: true);
        if (authResult != null) return authResult;

        try
        {
            var client = _httpClientFactory.CreateClient("InferenceService");
            var response = await client.GetAsync("/api/chess/admin/statistics");
            var json = await response.Content.ReadAsStringAsync();

            var stats = JsonSerializer.Deserialize<AdminStatisticsResponse>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            _logger.LogInformation("Admin statistics requested. Total users: {UniqueUsers}, Total corrections: {TotalCorrections}",
                stats?.UniqueUsers, stats?.TotalCorrections);

            return Ok(stats);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting statistics");
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

            var response = await client.PostAsync("/api/chess/align", content);
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

    /// <summary>
    /// Validate multiple FEN variants against Chess.com to find which one matches real games.
    /// Used when piece colors are ambiguous (e.g., queen could be white or black).
    /// </summary>
    [HttpPost("validate-fen-variants")]
    public async Task<ActionResult<FenValidationResult>> ValidateFenVariants([FromBody] FenValidationRequest request)
    {
        if (request?.FenVariants == null || request.FenVariants.Count < 2)
        {
            return BadRequest(new FenValidationResult 
            { 
                Success = false, 
                Error = "At least 2 FEN variants are required" 
            });
        }

        try
        {
            using var httpClient = new HttpClient();
            httpClient.DefaultRequestHeaders.Add("User-Agent", 
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36");
            httpClient.DefaultRequestHeaders.Add("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8");
            httpClient.DefaultRequestHeaders.Add("Accept-Language", "en-US,en;q=0.5");

            var results = new List<FenVariantResult>();
            string? matchedFen = null;
            
            foreach (var variant in request.FenVariants)
            {
                var encodedFen = Uri.EscapeDataString(variant.Fen);
                var url = $"https://www.chess.com/games/search?fen={encodedFen}";
                
                var response = await httpClient.GetAsync(url);
                var html = await response.Content.ReadAsStringAsync();
                
                // Check for "no games" message
                var hasNoGamesMessage = html.Contains("Your search did not match any games", StringComparison.OrdinalIgnoreCase) ||
                                        html.Contains("did not match any games", StringComparison.OrdinalIgnoreCase);
                
                // Check for game indicators
                var gamesFound = !hasNoGamesMessage && (
                    html.Contains("master-games-master-game", StringComparison.OrdinalIgnoreCase) ||
                    html.Contains("master-games-username", StringComparison.OrdinalIgnoreCase) ||
                    html.Contains("archived-games-game-row", StringComparison.OrdinalIgnoreCase) ||
                    html.Contains("/game/live/", StringComparison.OrdinalIgnoreCase) ||
                    html.Contains("/game/daily/", StringComparison.OrdinalIgnoreCase));
                
                results.Add(new FenVariantResult
                {
                    Fen = variant.Fen,
                    Label = variant.Label,
                    GamesFound = gamesFound,
                    SearchUrl = url
                });
                
                _logger.LogInformation("FEN variant validation: {Label} -> GamesFound={GamesFound}", 
                    variant.Label, gamesFound);
                
                if (gamesFound && matchedFen == null)
                {
                    matchedFen = variant.Fen;
                }
                
                // Small delay between requests to be nice to Chess.com
                await Task.Delay(200);
            }
            
            // Determine recommendation
            var matchingVariants = results.Where(r => r.GamesFound).ToList();
            string recommendation;
            
            if (matchingVariants.Count == 1)
            {
                recommendation = $"Use '{matchingVariants[0].Label}' - it matches games on Chess.com";
            }
            else if (matchingVariants.Count > 1)
            {
                recommendation = "Multiple variants match games - position is ambiguous";
            }
            else
            {
                recommendation = "No variants match games on Chess.com - position may be from an ongoing game";
            }

            return Ok(new FenValidationResult
            {
                Success = true,
                Variants = results,
                MatchedFen = matchedFen,
                Recommendation = recommendation
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating FEN variants");
            return Ok(new FenValidationResult
            {
                Success = false,
                Error = ex.Message
            });
        }
    }
}

/// <summary>
/// Request to validate multiple FEN variants
/// </summary>
public class FenValidationRequest
{
    public List<FenVariant> FenVariants { get; set; } = new();
}

public class FenVariant
{
    public string Fen { get; set; } = "";
    public string Label { get; set; } = "";
}

/// <summary>
/// Result from FEN variant validation
/// </summary>
public class FenValidationResult
{
    public bool Success { get; set; }
    public List<FenVariantResult>? Variants { get; set; }
    public string? MatchedFen { get; set; }
    public string? Recommendation { get; set; }
    public string? Error { get; set; }
}

public class FenVariantResult
{
    public string Fen { get; set; } = "";
    public string Label { get; set; } = "";
    public bool GamesFound { get; set; }
    public string SearchUrl { get; set; } = "";
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
