using System.Text.Json.Serialization;

namespace ChessFEN.Api.Models;

/// <summary>
/// Response from FEN prediction
/// </summary>
public class PredictionResponse
{
    public bool Success { get; set; }
    public string? Fen { get; set; }
    public string[][]? Board { get; set; }
    public double[][]? Confidences { get; set; }  // 8x8 confidence values (0-1)
    public string? Error { get; set; }
    public string? Model { get; set; }
    
    [JsonPropertyName("debug_images")]  // Match Python API's snake_case
    public DebugImages? DebugImages { get; set; }
    
    [JsonPropertyName("questionable_squares")]
    public List<QuestionableSquare>? QuestionableSquares { get; set; }
}

/// <summary>
/// A square with a potentially incorrect piece color
/// </summary>
public class QuestionableSquare
{
    public string? Square { get; set; }   // e.g., "c3"
    public string? Piece { get; set; }    // e.g., "bQ"
    public string? Reason { get; set; }   // e.g., "Detected 2 black queens, 0 white"
}

/// <summary>
/// Debug images for development/troubleshooting
/// </summary>
public class DebugImages
{
    public string? Detection { get; set; }  // Base64 PNG
    public string? Warped { get; set; }     // Base64 PNG
    public string? Overlay { get; set; }    // Base64 PNG
}

/// <summary>
/// Request to submit user correction feedback
/// </summary>
public class FeedbackRequest
{
    public string OriginalFen { get; set; } = "";
    public string CorrectedFen { get; set; } = "";
    public string? Image { get; set; }  // Base64 encoded
    public Dictionary<string, string>? CorrectedSquares { get; set; }  // e.g., {"a1": "wR"}
}

/// <summary>
/// Response from feedback submission
/// </summary>
public class FeedbackResponse
{
    public bool Success { get; set; }
    public string? FeedbackId { get; set; }
    public string? Error { get; set; }
}

/// <summary>
/// Model version info
/// </summary>
public class ModelVersion
{
    public required string Name { get; set; }
    public required string File { get; set; }
    public required string Type { get; set; }
    public string? Accuracy { get; set; }
    public int Epoch { get; set; }
    public bool IsBest { get; set; }
}

/// <summary>
/// Request to select a model
/// </summary>
public class SelectModelRequest
{
    public required string ModelName { get; set; }
}

/// <summary>
/// Deep analysis response with multi-model comparison and game lookup
/// </summary>
public class DeepAnalysisResponse
{
    public bool Success { get; set; }
    public string? Error { get; set; }
    
    // Consensus results
    public bool Consensus { get; set; }
    public string? ConsensusFen { get; set; }
    public string? RecommendedFen { get; set; }  // Best model's FEN if no consensus
    
    // Per-model results
    public List<ModelPrediction>? ModelResults { get; set; }
    
    // Disagreements
    public List<SquareDisagreement>? Disagreements { get; set; }
    
    // Chess.com game lookup
    public bool GamesFoundWhite { get; set; }
    public bool GamesFoundBlack { get; set; }
    public string? GamesMessage { get; set; }
}

/// <summary>
/// Individual model prediction result
/// </summary>
public class ModelPrediction
{
    public string? Model { get; set; }
    public string? Type { get; set; }
    public string? Accuracy { get; set; }
    public bool Success { get; set; }
    public string? Fen { get; set; }
    public string[][]? Board { get; set; }
    public string? Error { get; set; }
}

/// <summary>
/// Square where models disagree
/// </summary>
public class SquareDisagreement
{
    public string? Square { get; set; }
    public Dictionary<string, string>? Predictions { get; set; }
}

/// <summary>
/// Feedback item for admin review
/// </summary>
public class FeedbackItem
{
    public string? Id { get; set; }
    public string? Timestamp { get; set; }

    [JsonPropertyName("model_name")]
    public string? ModelName { get; set; }

    [JsonPropertyName("original_fen")]
    public string? OriginalFen { get; set; }

    [JsonPropertyName("corrected_fen")]
    public string? CorrectedFen { get; set; }

    [JsonPropertyName("corrected_squares")]
    public Dictionary<string, string>? CorrectedSquares { get; set; }

    [JsonPropertyName("user_id")]
    public string? UserId { get; set; }

    [JsonPropertyName("user_email")]
    public string? UserEmail { get; set; }
}

/// <summary>
/// Azure Container Apps authentication principal from X-MS-CLIENT-PRINCIPAL header
/// </summary>
public class ClientPrincipal
{
    public string? IdentityProvider { get; set; }
    public string? UserId { get; set; }
    public string? UserDetails { get; set; }  // Email address
    public IEnumerable<ClientPrincipalClaim>? Claims { get; set; }
}

/// <summary>
/// Claim from Azure Container Apps authentication
/// </summary>
public class ClientPrincipalClaim
{
    public string? Typ { get; set; }  // Claim type (e.g., "groups", "name", "email")
    public string? Val { get; set; }  // Claim value
}
