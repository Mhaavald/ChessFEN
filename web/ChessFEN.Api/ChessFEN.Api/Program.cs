using System.Text.Json;
using System.Text.Json.Serialization;
using ChessFEN.Api.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services
builder.Services.AddControllers()
    .AddJsonOptions(options =>
    {
        // Accept camelCase input from JavaScript clients
        options.JsonSerializerOptions.PropertyNameCaseInsensitive = true;
        // Output camelCase to match JavaScript conventions
        options.JsonSerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.CamelCase;
        options.JsonSerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
    });

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Register authorization helper
builder.Services.AddSingleton<AuthorizationHelper>();
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyMethod()
              .AllowAnyHeader();
    });
});

// Register HttpClient for Python inference service
builder.Services.AddHttpClient("InferenceService", client =>
{
    var baseUrl = builder.Configuration["InferenceService:BaseUrl"] ?? "http://localhost:5000";
    client.BaseAddress = new Uri(baseUrl);
    client.Timeout = TimeSpan.FromSeconds(60);
});

var app = builder.Build();

// Configure the HTTP request pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();

    // Mock authentication for local development
    app.Use(async (context, next) =>
    {
        if (!context.Request.Headers.ContainsKey("X-MS-CLIENT-PRINCIPAL"))
        {
            var mockPrincipal = new
            {
                identityProvider = "aad",
                userId = "dev-user-12345",
                userDetails = "developer@localhost.dev",
                claims = new[]
                {
                    new { typ = "name", val = "Local Developer" },
                    new { typ = "email", val = "developer@localhost.dev" }
                }
            };

            var json = JsonSerializer.Serialize(mockPrincipal);
            var base64 = Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes(json));
            context.Request.Headers["X-MS-CLIENT-PRINCIPAL"] = base64;

            Console.WriteLine($"[DEV AUTH] Mocked user: {mockPrincipal.userDetails}");
        }

        await next();
    });
}

app.UseCors();
app.MapControllers();

app.Run();

