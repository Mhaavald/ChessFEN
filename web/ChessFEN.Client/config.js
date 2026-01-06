/**
 * Chess FEN Scanner - Configuration
 * 
 * When using serve_proxy.py (HTTPS), API calls go through the proxy
 * When using serve.py (HTTP), API calls go directly to backend
 */

// If on HTTPS, use proxy (same origin) - avoids all CORS/cert issues
// If on HTTP, call API directly
const useProxy = window.location.protocol === 'https:';

const apiHost = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? 'localhost'
    : window.location.hostname;

// Proxy mode: /api/chess (same origin, proxied to backend)
// Direct mode: http://host:5000/api/chess (Python backend)
const API_BASE = useProxy
    ? '/api/chess'  // Proxied through serve_proxy.py
    : `http://${apiHost}:5000/api/chess`;

// .NET API for chess-com-search (not available in Python backend)
const DOTNET_API_BASE = useProxy
    ? '/api/chess'  // Proxied through serve_proxy.py
    : `http://${apiHost}:5001/api/chess`;

window.APP_CONFIG = {
    API_BASE: API_BASE,
    DOTNET_API_BASE: DOTNET_API_BASE,
    CHESS_COM_SEARCH_URL: 'https://www.chess.com/games/search'
};

console.log('=== Chess FEN Client Config ===');
console.log('Page URL:', window.location.href);
console.log('Using proxy:', useProxy);
console.log('API configured:', window.APP_CONFIG.API_BASE);
console.log('================================');
