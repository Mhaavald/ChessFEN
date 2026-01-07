# Chess FEN Scanner - HTTPS Server with API Proxy
# Proxies /api/* requests to the backend, avoiding CORS/cert issues on mobile

import http.server
import socketserver
import ssl
import os
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path
import threading

PORT = 5006
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CERT_DIR = Path(DIRECTORY) / "certs"
PYTHON_BACKEND = "http://localhost:5000"  # Python Flask inference service
DOTNET_BACKEND = "http://localhost:5001"  # .NET API for admin, chess-com-search


# Use ThreadingMixIn to handle multiple concurrent requests
class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()
    
    def do_GET(self):
        if self.path.startswith('/api/'):
            self.proxy_request('GET')
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path.startswith('/api/'):
            self.proxy_request('POST')
        else:
            self.send_error(405, "Method not allowed")
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def proxy_request(self, method):
        """Proxy request to backend API"""
        # Route to correct backend based on path (matching nginx.conf logic)
        if self.path.startswith('/api/chess/admin/') or self.path.startswith('/api/chess/chess-com-search'):
            backend = DOTNET_BACKEND
        elif self.path.startswith('/api/chess/'):
            backend = PYTHON_BACKEND
        else:
            backend = DOTNET_BACKEND  # Default to .NET for other /api/ routes

        url = f"{backend}{self.path}"

        print(f"[PROXY] {method} {self.path} -> {url}")

        try:
            # Read request body for POST
            body = None
            if method == 'POST':
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                print(f"[PROXY] Request body length: {len(body)} bytes")

            # Create request
            req = urllib.request.Request(url, data=body, method=method)
            req.add_header('Content-Type', 'application/json')
            
            # Make request to backend
            with urllib.request.urlopen(req, timeout=120) as response:
                response_body = response.read()
                
                print(f"[PROXY] Response: {response.status}, {len(response_body)} bytes")
                
                self.send_response(response.status)
                self.send_header('Content-Type', response.headers.get('Content-Type', 'application/json'))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(response_body)
                
        except urllib.error.HTTPError as e:
            error_body = e.read() if e.fp else b'{"error": "API error"}'
            print(f"[PROXY] HTTP Error {e.code}: {error_body.decode()[:200]}")
            self.send_response(e.code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(error_body)
            
        except Exception as e:
            print(f"[PROXY] Exception: {e}")
            self.send_response(502)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Proxy error: {str(e)}"}).encode())


def generate_self_signed_cert():
    """Generate self-signed certificate for HTTPS."""
    CERT_DIR.mkdir(exist_ok=True)
    cert_file = CERT_DIR / "cert.pem"
    key_file = CERT_DIR / "key.pem"
    
    if cert_file.exists() and key_file.exists():
        print("Using existing certificate")
        return str(cert_file), str(key_file)
    
    print("Generating self-signed certificate...")
    
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        import datetime
        
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        ).sign(key, hashes.SHA256())
        
        with open(key_file, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        print("Certificate generated")
        return str(cert_file), str(key_file)
        
    except ImportError:
        print("ERROR: pip install cryptography")
        sys.exit(1)


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    
    cert_file, key_file = generate_self_signed_cert()
    
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_file, key_file)
    
    with ThreadedHTTPServer(("0.0.0.0", port), ProxyHandler) as httpd:
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        
        print(f"Chess FEN Scanner Client (HTTPS + Proxy)")
        print(f"=========================================")
        print(f"Serving at: https://localhost:{port}")
        print(f"API proxy:  /api/chess/* -> {PYTHON_BACKEND}")
        print(f"            /api/chess/admin/*, chess-com-search -> {DOTNET_BACKEND}")
        print()
        print("⚠️  Accept the certificate warning in your browser")
        print()
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
