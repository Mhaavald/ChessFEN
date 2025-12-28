"""
Chess FEN Inference Service - Flask API for model inference.

Provides endpoints for:
- Board detection and FEN generation
- Debug mode with overlay images
- Model version management
- User feedback storage for retraining
"""

import os
import sys

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import json
import uuid
import base64
from pathlib import Path
from datetime import datetime
from io import BytesIO
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image

# Add project paths
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from detect_board import detect_board_quad_grid_aware, warp_quad
from piece_detect import suggest_grid_adjustment, draw_piece_detection_overlay

app = Flask(__name__)
CORS(app)

# ============================
# Configuration
# ============================

MODELS_DIR = REPO_ROOT / "models"
FEEDBACK_DIR = REPO_ROOT / "data" / "user_feedback"
MODEL_VERSIONS_FILE = MODELS_DIR / "versions.json"

IMG_SIZE = 64
CLASSES = [
    "empty",
    "wP", "wN", "wB", "wR", "wQ", "wK",
    "bP", "bN", "bB", "bR", "bQ", "bK",
]

PIECE_TO_FEN = {
    "empty": "",
    "wP": "P", "wN": "N", "wB": "B", "wR": "R", "wQ": "Q", "wK": "K",
    "bP": "p", "bN": "n", "bB": "b", "bR": "r", "bQ": "q", "bK": "k",
}

# Unicode chess pieces for display
PIECE_UNICODE = {
    "empty": " ",
    "wP": "♙", "wN": "♘", "wB": "♗", "wR": "♖", "wQ": "♕", "wK": "♔",
    "bP": "♟", "bN": "♞", "bB": "♝", "bR": "♜", "bQ": "♛", "bK": "♚",
}

# ============================
# Model Loading
# ============================

class ChessCNN(nn.Module):
    """Simple CNN for chess piece classification."""
    
    def __init__(self, num_classes=13):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_resnet18(num_classes=13):
    """Create ResNet-18 model matching training script structure."""
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


# Global model cache
_models = {}
_current_model = None
_current_model_name = None


def get_model_versions():
    """Get available model versions."""
    versions = []
    
    for pth_file in MODELS_DIR.glob("*.pth"):
        checkpoint = torch.load(pth_file, map_location='cpu', weights_only=False)
        model_type = checkpoint.get('model_type', 'cnn')
        val_acc = checkpoint.get('val_acc', 0)
        epoch = checkpoint.get('epoch', 0)
        
        versions.append({
            "name": pth_file.stem,
            "file": pth_file.name,
            "type": model_type,
            "accuracy": f"{val_acc:.2%}" if val_acc else "unknown",
            "epoch": epoch,
            "is_best": "best" in pth_file.stem,
        })
    
    return sorted(versions, key=lambda x: x['name'])


def load_model(model_name: str = None):
    """Load a model by name. Returns cached model if available."""
    global _models, _current_model, _current_model_name
    
    if model_name is None:
        # Find best ResNet model, fallback to best CNN
        if (MODELS_DIR / "chess_resnet18_best.pth").exists():
            model_name = "chess_resnet18_best"
        elif (MODELS_DIR / "chess_cnn_best.pth").exists():
            model_name = "chess_cnn_best"
        else:
            raise FileNotFoundError("No model found")
    
    if model_name in _models:
        _current_model = _models[model_name]
        _current_model_name = model_name
        return _current_model
    
    model_path = MODELS_DIR / f"{model_name}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    device = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_type = checkpoint.get('model_type', 'cnn')
    
    if model_type == 'resnet18':
        model = create_resnet18(num_classes=len(CLASSES))
    else:
        model = ChessCNN(num_classes=len(CLASSES))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    _models[model_name] = model
    _current_model = model
    _current_model_name = model_name
    
    print(f"Loaded model: {model_name} (type: {model_type})")
    return model


# Image transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ============================
# Inference Logic
# ============================

def predict_board(img_bgr, model, debug=False, skip_detection=False):
    """
    Detect board, classify pieces, return FEN and optional debug images.
    
    Args:
        img_bgr: Input image in BGR format
        model: PyTorch model for classification
        debug: Include debug images in result
        skip_detection: If True, treat the entire image as the board (skip detection/warp)
    
    Returns:
        dict with keys: fen, board (2D list), success, error, debug_images (if debug=True)
    """
    result = {
        "success": False,
        "fen": None,
        "board": None,
        "error": None,
    }
    
    if debug:
        result["debug_images"] = {}
    
    if skip_detection:
        # Treat entire image as the board - just resize to 512x512
        warped = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_AREA)
        tile_size = 512 // 8
        
        if debug:
            # No detection overlay when skipping detection
            result["debug_images"]["detection"] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            result["debug_images"]["warped"] = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    else:
        # Detect board
        quad = detect_board_quad_grid_aware(img_bgr, crop_ratio=0.80)
        if quad is None:
            result["error"] = "Could not detect chess board"
            return result
        
        # Warp board
        warped = warp_quad(img_bgr, quad, out_size=512)
        tile_size = 512 // 8
        
        if debug:
            # Board detection overlay
            overlay = img_bgr.copy()
            pts = quad.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(overlay, [pts], True, (0, 255, 0), 3)
            result["debug_images"]["detection"] = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            result["debug_images"]["warped"] = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    
    # Classify each tile
    board = [[None for _ in range(8)] for _ in range(8)]
    
    with torch.no_grad():
        for row in range(8):
            for col in range(8):
                y1, y2 = row * tile_size, (row + 1) * tile_size
                x1, x2 = col * tile_size, (col + 1) * tile_size
                tile = warped[y1:y2, x1:x2]
                
                # Convert to PIL for transform
                tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(tile_rgb)
                img_tensor = transform(pil_img).unsqueeze(0)
                
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_idx = probs.argmax().item()
                
                board[row][col] = CLASSES[pred_idx]
    
    # Generate FEN
    fen_rows = []
    for row in range(8):
        fen_row = ""
        empty_count = 0
        for col in range(8):
            piece = board[row][col]
            if piece == "empty":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += PIECE_TO_FEN[piece]
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    
    fen = "/".join(fen_rows)
    
    if debug:
        # Create overlay with predicted pieces
        overlay_warped = warped.copy()
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece != "empty":
                    cx = col * tile_size + tile_size // 2
                    cy = row * tile_size + tile_size // 2
                    cv2.putText(overlay_warped, PIECE_UNICODE[piece], 
                               (cx - 20, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 
                               1.5, (255, 0, 255), 3)
        result["debug_images"]["overlay"] = cv2.cvtColor(overlay_warped, cv2.COLOR_BGR2RGB)
    
    result["success"] = True
    result["fen"] = fen
    result["board"] = board
    
    return result


def image_to_base64(img_rgb):
    """Convert RGB numpy array to base64 string."""
    pil_img = Image.fromarray(img_rgb)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# ============================
# Feedback Storage
# ============================

def save_feedback(original_fen: str, corrected_fen: str, image_base64: str, 
                  corrected_squares: dict, model_name: str):
    """Save user correction feedback for later retraining."""
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    
    feedback_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()
    
    feedback = {
        "id": feedback_id,
        "timestamp": timestamp,
        "model_name": model_name,
        "original_fen": original_fen,
        "corrected_fen": corrected_fen,
        "corrected_squares": corrected_squares,  # e.g., {"a1": "wR", "b2": "empty"}
    }
    
    # Save feedback JSON
    feedback_file = FEEDBACK_DIR / f"{feedback_id}.json"
    with open(feedback_file, 'w') as f:
        json.dump(feedback, f, indent=2)
    
    # Save original image
    if image_base64:
        img_data = base64.b64decode(image_base64)
        img_file = FEEDBACK_DIR / f"{feedback_id}.png"
        with open(img_file, 'wb') as f:
            f.write(img_data)
    
    return feedback_id


def get_pending_feedback():
    """Get list of pending feedback items for admin review."""
    if not FEEDBACK_DIR.exists():
        return []
    
    items = []
    for json_file in FEEDBACK_DIR.glob("*.json"):
        with open(json_file) as f:
            items.append(json.load(f))
    
    return sorted(items, key=lambda x: x['timestamp'], reverse=True)


# ============================
# API Endpoints
# ============================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model": _current_model_name})


@app.route('/api/models', methods=['GET'])
def list_models():
    """List available model versions."""
    return jsonify(get_model_versions())


@app.route('/api/models/select', methods=['POST'])
def select_model():
    """Select a model to use for inference."""
    data = request.json
    model_name = data.get('model_name')
    
    try:
        load_model(model_name)
        return jsonify({"success": True, "model": model_name})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict FEN from chess board image.
    
    Request:
        - image: base64 encoded image OR file upload
        - debug: boolean (optional) - include debug images
        - model: string (optional) - model name to use
        - skip_detection: boolean (optional) - skip board detection, treat image as board
    
    Response:
        - success: boolean
        - fen: string
        - board: 2D array of piece codes
        - debug_images: dict of base64 images (if debug=True)
    """
    debug = request.args.get('debug', 'false').lower() == 'true'
    model_name = request.args.get('model', None)
    skip_detection = request.args.get('skip_detection', 'false').lower() == 'true'
    
    print(f"[PREDICT] Received request, debug={debug}, model={model_name}, skip_detection={skip_detection}")
    
    # Load model
    try:
        model = load_model(model_name)
    except Exception as e:
        print(f"[PREDICT] Model error: {e}")
        return jsonify({"success": False, "error": f"Model error: {e}"}), 400
    
    # Get image
    img_bgr = None
    if 'image' in request.files:
        print("[PREDICT] Image from files")
        file = request.files['image']
        img_bytes = file.read()
        print(f"[PREDICT] File bytes: {len(img_bytes)}")
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif request.json and 'image' in request.json:
        print("[PREDICT] Image from JSON body")
        img_base64 = request.json['image']
        print(f"[PREDICT] Base64 length: {len(img_base64)}, first 50 chars: {img_base64[:50]}")
        
        # Handle data URL format (e.g., "data:image/jpeg;base64,...")
        if ',' in img_base64 and img_base64.startswith('data:'):
            img_base64 = img_base64.split(',', 1)[1]
            print(f"[PREDICT] Stripped data URL prefix, new length: {len(img_base64)}")
        
        try:
            img_data = base64.b64decode(img_base64)
            print(f"[PREDICT] Decoded bytes: {len(img_data)}")
            
            # Try OpenCV first
            nparr = np.frombuffer(img_data, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # If OpenCV fails, try PIL
            if img_bgr is None:
                print("[PREDICT] cv2.imdecode returned None, trying PIL...")
                try:
                    pil_img = Image.open(BytesIO(img_data))
                    # Convert to RGB if needed
                    if pil_img.mode != 'RGB':
                        print(f"[PREDICT] Converting from {pil_img.mode} to RGB")
                        pil_img = pil_img.convert('RGB')
                    # Convert PIL to OpenCV format (BGR)
                    img_rgb = np.array(pil_img)
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    print(f"[PREDICT] PIL decoded image shape: {img_bgr.shape}")
                except Exception as pil_err:
                    print(f"[PREDICT] PIL decode also failed: {pil_err}")
            else:
                print(f"[PREDICT] Decoded image shape: {img_bgr.shape}")
        except Exception as e:
            print(f"[PREDICT] Decode error: {e}")
            return jsonify({"success": False, "error": f"Decode error: {e}"}), 400
    else:
        print("[PREDICT] No image in request")
        return jsonify({"success": False, "error": "No image provided"}), 400
    
    if img_bgr is None:
        print("[PREDICT] Invalid image - img_bgr is None")
        return jsonify({"success": False, "error": "Invalid image"}), 400
    
    print(f"[PREDICT] Running prediction on image {img_bgr.shape}, skip_detection={skip_detection}")
    # Run prediction
    result = predict_board(img_bgr, model, debug=debug, skip_detection=skip_detection)
    
    # Convert debug images to base64
    if debug and result.get("debug_images"):
        for key, img in result["debug_images"].items():
            result["debug_images"][key] = image_to_base64(img)
    
    result["model"] = _current_model_name
    
    return jsonify(result)


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit user correction feedback.
    
    Request:
        - original_fen: string
        - corrected_fen: string
        - image: base64 encoded image
        - corrected_squares: dict of {square: piece} corrections
    """
    data = request.json
    
    feedback_id = save_feedback(
        original_fen=data.get('original_fen'),
        corrected_fen=data.get('corrected_fen'),
        image_base64=data.get('image'),
        corrected_squares=data.get('corrected_squares', {}),
        model_name=_current_model_name
    )
    
    return jsonify({"success": True, "feedback_id": feedback_id})


@app.route('/api/feedback', methods=['GET'])
def list_feedback():
    """List pending feedback items (admin endpoint)."""
    return jsonify(get_pending_feedback())


@app.route('/api/align', methods=['POST'])
def align_grid():
    """
    Analyze an image to detect the board, warp it, and show piece positions for grid alignment.
    
    This endpoint helps when automatic board detection produces misaligned grids.
    It detects the board, warps it to a square, finds chess pieces, and shows 
    how far off the grid alignment is.
    
    Request:
        - image: base64 encoded original image (not warped)
    
    Response:
        - success: boolean
        - board_detected: boolean
        - piece_found: boolean
        - piece_center: [x, y] if found
        - piece_square: [row, col] detected square
        - square_name: string (e.g., "e4")
        - offset: [dx, dy] suggested offset in pixels
        - confidence: float 0-1
        - overlay: base64 image with warped board, grid, and piece detection
    """
    print("[ALIGN] Received grid alignment request")
    
    # Check skip_detection flag
    skip_detection = False
    if request.json and 'skip_detection' in request.json:
        skip_detection = request.json.get('skip_detection', False)
    
    # Get image
    img_bgr = None
    if request.json and 'image' in request.json:
        img_base64 = request.json['image']
        
        # Handle data URL format
        if ',' in img_base64 and img_base64.startswith('data:'):
            img_base64 = img_base64.split(',', 1)[1]
        
        try:
            img_data = base64.b64decode(img_base64)
            nparr = np.frombuffer(img_data, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img_bgr is None:
                # Try PIL
                pil_img = Image.open(BytesIO(img_data))
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                img_rgb = np.array(pil_img)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[ALIGN] Decode error: {e}")
            return jsonify({"success": False, "error": f"Decode error: {e}"}), 400
    else:
        return jsonify({"success": False, "error": "No image provided"}), 400
    
    if img_bgr is None:
        return jsonify({"success": False, "error": "Invalid image"}), 400
    
    print(f"[ALIGN] Analyzing image {img_bgr.shape}, skip_detection={skip_detection}")
    
    # Step 1: Detect the board or use image as-is
    if skip_detection:
        # Treat entire image as the board - resize to 640x640
        warped = cv2.resize(img_bgr, (640, 640), interpolation=cv2.INTER_AREA)
        print(f"[ALIGN] Skip detection - resized to {warped.shape}")
    else:
        quad = detect_board_quad_grid_aware(img_bgr)
        if quad is None:
            print("[ALIGN] No board detected")
            return jsonify({
                "success": True,
                "board_detected": False,
                "piece_found": False,
                "suggestion": "Could not detect a chess board in the image. Try a clearer photo with better lighting.",
                "overlay": None
            })
        
        # Step 2: Warp to square
        warped = warp_quad(img_bgr, quad, out_size=640)
        print(f"[ALIGN] Warped to {warped.shape}")
    
    # Step 3: Run piece detection on warped board
    detection = suggest_grid_adjustment(warped)
    
    # Step 4: Create detailed overlay
    h, w = warped.shape[:2]
    sq_size = w // 8
    overlay = warped.copy()
    
    # Draw grid lines
    for i in range(9):
        p = i * sq_size
        cv2.line(overlay, (0, p), (w, p), (0, 255, 0), 2)
        cv2.line(overlay, (p, 0), (p, h), (0, 255, 0), 2)
    
    # Draw square labels
    files = "abcdefgh"
    for row in range(8):
        for col in range(8):
            label = f"{files[col]}{8-row}"
            x = col * sq_size + 5
            y = row * sq_size + 18
            cv2.putText(overlay, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Build response
    result = {
        "success": True,
        "board_detected": True,
        "piece_found": detection["piece_found"],
    }
    
    if detection["piece_found"]:
        cx, cy = detection["piece_center"]
        row, col = detection["piece_square"]
        dx, dy = detection["offset"]
        
        # Draw expected center (green circle)
        expected_cx = int((col + 0.5) * sq_size)
        expected_cy = int((row + 0.5) * sq_size)
        cv2.circle(overlay, (expected_cx, expected_cy), 12, (0, 255, 0), 3)
        
        # Draw actual center (red filled circle)
        cv2.circle(overlay, (int(cx), int(cy)), 8, (0, 0, 255), -1)
        
        # Draw offset arrow
        cv2.arrowedLine(overlay, (expected_cx, expected_cy), (int(cx), int(cy)), 
                        (255, 0, 255), 3, tipLength=0.3)
        
        # Add text info
        square_name = f"{files[col]}{8-row}"
        info_text = f"Piece in {square_name}, offset: ({dx:.0f}, {dy:.0f})px"
        cv2.putText(overlay, info_text, (10, h - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if abs(dx) > 5 or abs(dy) > 5:
            adjust_text = f"Adjust grid by ({-dx:.0f}, {-dy:.0f})px"
            cv2.putText(overlay, adjust_text, (10, h - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            result["aligned"] = False
            result["suggestion"] = f"Grid misaligned. The piece center is {abs(dx):.0f}px {'left' if dx < 0 else 'right'} and {abs(dy):.0f}px {'up' if dy < 0 else 'down'} from expected."
        else:
            cv2.putText(overlay, "Grid aligned OK", (10, h - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            result["aligned"] = True
            result["suggestion"] = "Grid alignment looks good. Pieces are centered in their squares."
        
        result["piece_center"] = [round(cx, 1), round(cy, 1)]
        result["piece_square"] = [row, col]
        result["square_name"] = square_name
        result["offset"] = [round(dx, 1), round(dy, 1)]
        result["confidence"] = round(detection["confidence"], 2)
    else:
        cv2.putText(overlay, "No piece detected", (10, h - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        result["suggestion"] = "No piece detected. The board may be empty or pieces have low contrast."
    
    # Convert overlay to base64
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    overlay_b64 = image_to_base64(overlay_rgb)
    result["overlay"] = overlay_b64
    
    print(f"[ALIGN] Result: board_detected=True, piece_found={detection['piece_found']}")
    return jsonify(result)


# ============================
# Main
# ============================

if __name__ == '__main__':
    # Load default model on startup
    try:
        load_model()
        print(f"Default model loaded: {_current_model_name}")
    except Exception as e:
        print(f"Warning: Could not load default model: {e}")
    
    # Run Flask app
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    print(f"Starting inference service on port {port} (debug={debug_mode})")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
