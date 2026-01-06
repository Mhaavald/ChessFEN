using System.Text.Json;
using ChessFEN.Api.Models;

namespace ChessFEN.Api.Services;

public class AuthorizationHelper
{
    private readonly IConfiguration _configuration;
    private readonly ILogger<AuthorizationHelper> _logger;

    public AuthorizationHelper(IConfiguration configuration, ILogger<AuthorizationHelper> logger)
    {
        _configuration = configuration;
        _logger = logger;
    }

    /// <summary>
    /// Get user authentication context from Azure Container Apps headers
    /// </summary>
    public UserAuthContext GetUserContext(HttpRequest request)
    {
        var principalHeader = request.Headers["X-MS-CLIENT-PRINCIPAL"].FirstOrDefault();

        if (string.IsNullOrEmpty(principalHeader))
        {
            return new UserAuthContext
            {
                IsAuthenticated = false,
                Message = "Not authenticated - no X-MS-CLIENT-PRINCIPAL header"
            };
        }

        try
        {
            // Decode base64 header
            var decoded = Convert.FromBase64String(principalHeader);
            var json = System.Text.Encoding.UTF8.GetString(decoded);
            var principal = JsonSerializer.Deserialize<ClientPrincipal>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            if (principal == null)
            {
                return new UserAuthContext { IsAuthenticated = false, Message = "Invalid principal" };
            }

            // Extract user info
            var userId = principal.UserId;
            var email = principal.UserDetails;

            // If UserDetails is null or doesn't contain @, try to get email from claims
            if (string.IsNullOrEmpty(email) || !email.Contains("@"))
            {
                email = principal.Claims?.FirstOrDefault(c =>
                    c.Typ?.Equals("email", StringComparison.OrdinalIgnoreCase) == true ||
                    c.Typ?.Equals("emails", StringComparison.OrdinalIgnoreCase) == true ||
                    c.Typ?.Equals("preferred_username", StringComparison.OrdinalIgnoreCase) == true
                )?.Val;
            }

            // Check if email domain is allowed
            var hasAccess = IsEmailDomainAllowed(email, out bool isAdmin);

            if (!hasAccess)
            {
                _logger.LogWarning("Access denied for user: {Email}", email);
                return new UserAuthContext
                {
                    IsAuthenticated = true,
                    HasAccess = false,
                    Email = email,
                    UserId = userId,
                    Message = $"Email domain not allowed. Contact administrator to request access."
                };
            }

            _logger.LogInformation("User authenticated: Email={Email}, IsAdmin={IsAdmin}", email, isAdmin);

            return new UserAuthContext
            {
                IsAuthenticated = true,
                HasAccess = true,
                IsAdmin = isAdmin,
                UserId = userId,
                Email = email,
                Message = "Authorized"
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error parsing principal header");
            return new UserAuthContext
            {
                IsAuthenticated = false,
                Message = $"Error parsing auth: {ex.Message}"
            };
        }
    }

    /// <summary>
    /// Check if email domain is in the allowed list
    /// </summary>
    private bool IsEmailDomainAllowed(string? email, out bool isAdmin)
    {
        isAdmin = false;

        if (string.IsNullOrEmpty(email))
            return false;

        // Get allowed domains from configuration
        var allowedDomains = _configuration.GetSection("Authorization:AllowedDomains")
            .Get<string[]>() ?? Array.Empty<string>();

        // Get admin emails from configuration
        var adminEmails = _configuration.GetSection("Authorization:AdminEmails")
            .Get<string[]>() ?? Array.Empty<string>();

        // Check if user is admin by exact email match
        isAdmin = adminEmails.Any(adminEmail =>
            email.Equals(adminEmail, StringComparison.OrdinalIgnoreCase));

        if (isAdmin)
            return true; // Admins always have access

        // Check regular allowed domains
        return allowedDomains.Any(domain =>
            email.EndsWith($"@{domain}", StringComparison.OrdinalIgnoreCase));
    }
}

/// <summary>
/// User authentication and authorization context
/// </summary>
public class UserAuthContext
{
    public bool IsAuthenticated { get; set; }
    public bool HasAccess { get; set; }
    public bool IsAdmin { get; set; }
    public string? UserId { get; set; }
    public string? Email { get; set; }
    public string? Message { get; set; }
}
