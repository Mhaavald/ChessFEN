# Chess FEN Scanner - Client Development Server
# Simple Python HTTP server with CORS support for development

import http.server
import socketserver
import os
import sys

PORT = 5004
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    
    with socketserver.TCPServer(("0.0.0.0", port), CORSRequestHandler) as httpd:
        print(f"Chess FEN Scanner Client")
        print(f"========================")
        print(f"Serving at: http://localhost:{port}")
        print(f"Directory:  {DIRECTORY}")
        print()
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
