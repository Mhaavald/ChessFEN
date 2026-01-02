# Chess FEN Scanner - HTTPS Client Server
# Uses Python's ssl module with self-signed certificate for mobile camera access

import http.server
import socketserver
import ssl
import os
import sys
import subprocess
from pathlib import Path

PORT = 5005
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CERT_DIR = Path(DIRECTORY) / "certs"

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


def generate_self_signed_cert():
    """Generate self-signed certificate for HTTPS."""
    CERT_DIR.mkdir(exist_ok=True)
    cert_file = CERT_DIR / "cert.pem"
    key_file = CERT_DIR / "key.pem"
    
    if cert_file.exists() and key_file.exists():
        print("Using existing certificate")
        return str(cert_file), str(key_file)
    
    print("Generating self-signed certificate...")
    
    # Try using openssl if available
    try:
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key_file), "-out", str(cert_file),
            "-days", "365", "-nodes",
            "-subj", "/CN=localhost"
        ], check=True, capture_output=True)
        print("Certificate generated with openssl")
        return str(cert_file), str(key_file)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Fallback: use Python's cryptography library
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        import datetime
        
        # Generate key
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        
        # Generate certificate
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
        
        # Write key
        with open(key_file, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        # Write cert
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        print("Certificate generated with cryptography library")
        return str(cert_file), str(key_file)
        
    except ImportError:
        print("ERROR: Could not generate certificate.")
        print("Install cryptography: pip install cryptography")
        print("Or install openssl and add to PATH")
        sys.exit(1)


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    
    cert_file, key_file = generate_self_signed_cert()
    
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_file, key_file)
    
    with socketserver.TCPServer(("0.0.0.0", port), CORSRequestHandler) as httpd:
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        
        print(f"Chess FEN Scanner Client (HTTPS)")
        print(f"=================================")
        print(f"Serving at: https://localhost:{port}")
        print(f"Directory:  {DIRECTORY}")
        print()
        print("⚠️  Accept the certificate warning in your browser")
        print()
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
