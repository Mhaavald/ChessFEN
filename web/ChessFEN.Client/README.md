# Chess FEN Scanner - Client

A lightweight, static web application for scanning chess board images and finding games on Chess.com.

## Features

- 📸 **Camera Capture** - Capture board images with overlay guide
- 📁 **File Upload** - Upload images from device storage
- 🔍 **Board Analysis** - ML-powered FEN generation
- ✅ **FEN Validation** - Automatic validation with clear feedback
- 📋 **Copy FEN** - One-click copy to clipboard
- ♟️ **Chess.com Search** - Find games by position (White/Black to move)

## Quick Start

### Development (with full stack)

```powershell
# From repository root
.\web\ChessFEN.Client\start-client.ps1
```

This starts:
- Python inference service (port 5000)
- .NET API (port 5001)
- Client app (port 5004)

Then open: http://localhost:5004

### HTTPS for Mobile Camera

Mobile browsers require HTTPS for camera access:

```powershell
# Start HTTPS server (port 5005)
python web\ChessFEN.Client\serve_https.py
```

Then open: https://localhost:5005 (accept certificate warning)

### Static Hosting

The client is a static site that can be hosted anywhere:

1. Copy these files to your web server:
   - `index.html`
   - `styles.css`
   - `app.js`
   - `config.js`

2. Edit `config.js` to set your API endpoint:
   ```javascript
   window.APP_CONFIG = {
       API_BASE: 'https://your-api-server.com/api/chess',
       CHESS_COM_SEARCH_URL: 'https://www.chess.com/games/search'
   };
   ```

## Architecture

```
┌─────────────┐     ┌──────────┐     ┌─────────────────┐
│   Browser   │────►│ .NET API │────►│ Python Inference│
│  (Client)   │     │ (5001)   │     │    (5000)       │
└─────────────┘     └──────────┘     └─────────────────┘
```

- **Client**: Pure HTML/CSS/JS, no framework dependencies
- **.NET API**: Handles routing and Chess.com integration
- **Python**: ML inference with PyTorch models

## User Flow

1. **Capture or Upload** - Take a photo or select an image
2. **Analyze** - Click to run ML analysis
3. **Review** - See the detected position on a visual board
4. **Copy/Search** - Copy FEN or search Chess.com for games

## Browser Support

- Chrome/Edge 80+
- Safari 14+
- Firefox 75+

Camera capture requires HTTPS on mobile browsers.

## Configuration

Edit `config.js`:

| Setting | Description |
|---------|-------------|
| `API_BASE` | Backend API URL |
| `CHESS_COM_SEARCH_URL` | Chess.com search endpoint |

## Files

| File | Description |
|------|-------------|
| `index.html` | Main HTML structure |
| `styles.css` | All styling |
| `app.js` | Application logic |
| `config.js` | Configuration |
| `serve.py` | HTTP dev server |
| `serve_https.py` | HTTPS dev server |
| `start-client.ps1` | Full stack startup |
