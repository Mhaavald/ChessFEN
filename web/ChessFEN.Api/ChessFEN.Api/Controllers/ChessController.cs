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
    [HttpPost("predict")]
    [Consumes("multipart/form-data")]
    public async Task<ActionResult<PredictionResponse>> Predict(
        IFormFile image,
        [FromQuery] bool debug = false,
        [FromQuery] string? model = null)
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
        [FromQuery] string? model = null)
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
}
