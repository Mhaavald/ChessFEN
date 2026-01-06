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
# In container: /app/inference/inference_service.py -> parents[1] = /app
# In dev: src/inference/inference_service.py -> parents[2] = repo root
SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR.parent / "scripts").exists():
    REPO_ROOT = SCRIPT_DIR.parent  # Container: /app
else:
    REPO_ROOT = SCRIPT_DIR.parents[1]  # Dev: repo root
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
PREDICTIONS_DIR = REPO_ROOT / "data" / "predictions"
MODEL_VERSIONS_FILE = MODELS_DIR / "versions.json"

# Warped board resolution - use 640 for 80x80 tiles to match training data
WARP_SIZE = 640
TILE_SIZE = WARP_SIZE // 8  # 80px tiles

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


def preprocess_image(img_bgr, apply_grayscale=True):
    """
    Preprocess input image for consistent results across all clients.
    
    Args:
        img_bgr: Input image in BGR format
        apply_grayscale: If True, convert to grayscale (enabled by default as
                        training data is grayscale)
    
    Applies:
    1. Grayscale conversion (required - training data is grayscale)
    2. Auto-levels (histogram stretch)
    3. Contrast enhancement (1.3x)
    
    This centralizes preprocessing so all UIs get the same results.
    """
    if not apply_grayscale:
        # Return original color image - NOT RECOMMENDED, training data is grayscale
        return img_bgr
    
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Auto-levels (histogram stretch)
    min_val = gray.min()
    max_val = gray.max()
    if max_val > min_val:
        gray = ((gray.astype(np.float32) - min_val) / (max_val - min_val) * 255.0)
    
    # Step 3: Contrast enhancement (1.3x around midpoint)
    contrast_factor = 1.3
    gray = 128 + (gray - 128) * contrast_factor
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    
    # Convert back to 3-channel BGR for model compatibility
    img_out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    return img_out


# ============================
# Inference Logic
# ============================

# Offset search configuration
OFFSET_SEARCH_CONFIDENCE_BOOST = 0.15  # If offset gives 15%+ higher confidence, prefer it
OFFSET_PERCENTAGES = [-0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20]


def predict_tile_with_offset_search(warped, row, col, tile_size, model):
    """
    Predict a single tile, always trying offset positions to find best alignment.
    
    This compensates for lens distortion and perspective errors that cause
    misalignment between the detected grid and actual square boundaries.
    
    The function first gets the baseline prediction, then only searches offsets
    if the baseline confidence is below a threshold OR if the top-2 predictions
    are close (indicating confusion).
    
    Returns:
        piece: predicted piece class
        confidence: confidence score
        used_offset: (dx_pct, dy_pct) offset used, or (0, 0) if baseline was used
    """
    h, w = warped.shape[:2]
    base_y = row * tile_size
    base_x = col * tile_size
    
    # First, get baseline prediction (no offset)
    y1, y2 = base_y, base_y + tile_size
    x1, x2 = base_x, base_x + tile_size
    
    tile = warped[y1:y2, x1:x2]
    tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(tile_rgb)
    img_tensor = transform(pil_img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        
        # Get top-2 predictions to check for confusion
        top2_probs, top2_indices = torch.topk(probs[0], 2)
        baseline_conf = top2_probs[0].item()
        second_conf = top2_probs[1].item()
        baseline_piece = CLASSES[top2_indices[0].item()]
    
    # Decide if we need offset search:
    # 1. Low confidence (< 85%)
    # 2. Top-2 are close (within 20% of each other) - indicates confusion
    confidence_gap = baseline_conf - second_conf
    needs_offset_search = baseline_conf < 0.85 or confidence_gap < 0.20
    
    if not needs_offset_search:
        # High confidence, clear winner - use baseline
        return baseline_piece, baseline_conf, (0, 0)
    
    # Search offsets for better prediction
    best_piece = baseline_piece
    best_conf = baseline_conf
    best_offset = (0, 0)
    
    for dy_pct in OFFSET_PERCENTAGES:
        for dx_pct in OFFSET_PERCENTAGES:
            if dy_pct == 0 and dx_pct == 0:
                continue  # Already have baseline
            
            dy = int(dy_pct * tile_size)
            dx = int(dx_pct * tile_size)
            
            oy1 = max(0, base_y + dy)
            oy2 = min(h, base_y + tile_size + dy)
            ox1 = max(0, base_x + dx)
            ox2 = min(w, base_x + tile_size + dx)
            
            # Skip if tile would be too small
            if oy2 - oy1 < tile_size * 0.7 or ox2 - ox1 < tile_size * 0.7:
                continue
            
            tile = warped[oy1:oy2, ox1:ox2]
            tile_resized = cv2.resize(tile, (tile_size, tile_size))
            tile_rgb = cv2.cvtColor(tile_resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(tile_rgb)
            img_tensor = transform(pil_img).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_idx = probs.argmax().item()
                conf = probs[0, pred_idx].item()
            
            candidate_piece = CLASSES[pred_idx]
            
            # Apply penalty for switching from a piece to "empty"
            # This prevents the common failure mode where offset causes piece to be missed
            effective_conf = conf
            if baseline_piece != "empty" and candidate_piece == "empty":
                # Strong penalty - require much higher confidence to switch to empty
                effective_conf = conf * 0.5  # Halve the effective confidence
            
            if effective_conf > best_conf:
                best_piece = candidate_piece
                best_conf = conf  # Store actual confidence, not penalized one
                best_offset = (dx_pct, dy_pct)
    
    # Log improvement if offset helped
    if best_offset != (0, 0) and best_conf > baseline_conf + 0.05:
        print(f"[OFFSET] Improved: baseline {baseline_piece} {baseline_conf:.1%} -> {best_piece} {best_conf:.1%} at offset {best_offset}")
    
    return best_piece, best_conf, best_offset


def predict_tiles_with_confidence(warped, model, tile_size=64, use_offset_search=True):
    """
    Classify all 64 tiles and return predictions with confidence scores.
    
    Args:
        warped: Warped board image
        model: Classification model
        tile_size: Size of each tile in pixels
        use_offset_search: If True, try offset positions for low-confidence tiles
    
    Returns:
        board: 8x8 list of piece names
        confidences: 8x8 list of confidence values (0-1)
        avg_confidence: average confidence across all tiles
        low_conf_count: number of tiles with confidence < 0.5
    """
    board = [[None for _ in range(8)] for _ in range(8)]
    confidences = [[0.0 for _ in range(8)] for _ in range(8)]
    offsets_used = [[None for _ in range(8)] for _ in range(8)]
    
    for row in range(8):
        for col in range(8):
            if use_offset_search:
                # Use offset search for potentially misaligned tiles
                piece, confidence, offset = predict_tile_with_offset_search(
                    warped, row, col, tile_size, model
                )
                board[row][col] = piece
                confidences[row][col] = confidence
                offsets_used[row][col] = offset
            else:
                # Original logic - no offset search
                y1, y2 = row * tile_size, (row + 1) * tile_size
                x1, x2 = col * tile_size, (col + 1) * tile_size
                tile = warped[y1:y2, x1:x2]
                
                tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(tile_rgb)
                img_tensor = transform(pil_img).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_idx = probs.argmax().item()
                    confidence = probs[0, pred_idx].item()
                
                board[row][col] = CLASSES[pred_idx]
                confidences[row][col] = confidence
    
    # Calculate quality metrics
    all_confs = [confidences[r][c] for r in range(8) for c in range(8)]
    avg_confidence = sum(all_confs) / len(all_confs)
    low_conf_count = sum(1 for c in all_confs if c < 0.5)
    
    # Log if offset search improved results
    if use_offset_search:
        offset_count = sum(1 for r in range(8) for c in range(8) 
                          if offsets_used[r][c] and offsets_used[r][c] != (0, 0))
        if offset_count > 0:
            print(f"[OFFSET] Used offset search for {offset_count} tiles")
    
    return board, confidences, avg_confidence, low_conf_count


def validate_board_detection(board, confidences, avg_confidence, low_conf_count):
    """
    Validate whether the board detection appears correct based on predictions.
    
    Returns:
        is_valid: True if detection seems good
        reason: explanation if invalid
    """
    # Check 1: Average confidence should be reasonable (> 0.6)
    if avg_confidence < 0.6:
        return False, f"Low average confidence: {avg_confidence:.2f}"
    
    # Check 2: Too many low-confidence tiles (> 20 out of 64)
    if low_conf_count > 20:
        return False, f"Too many low-confidence tiles: {low_conf_count}/64"
    
    # Check 3: Valid piece distribution (basic sanity)
    piece_counts = {}
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            piece_counts[piece] = piece_counts.get(piece, 0) + 1
    
    # A real chess position should have at most 2 kings, 1 per side
    if piece_counts.get("wK", 0) > 1 or piece_counts.get("bK", 0) > 1:
        return False, "Too many kings detected"
    
    # Should have reasonable number of empty squares (not all empty, not all pieces)
    empty_count = piece_counts.get("empty", 0)
    if empty_count > 58:  # Almost all empty
        return False, "Almost all squares detected as empty"
    if empty_count < 10:  # Too few empty
        return False, "Too few empty squares detected"
    
    return True, "OK"


def find_missing_king(warped, board, confidences, missing_king, model, tile_size):
    """
    Search for a missing king among empty and low-confidence squares.
    
    Args:
        warped: Warped board image
        board: 8x8 list of current predictions
        confidences: 8x8 list of confidence values
        missing_king: 'wK' or 'bK'
        model: Classification model
        tile_size: Size of each tile
    
    Returns:
        (row, col, confidence) of best candidate, or None if not found
    """
    king_idx = CLASSES.index(missing_king)
    candidates = []
    
    # Collect candidate squares: empty predictions, low confidence, and edge squares
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            conf = confidences[row][col]
            
            # Priority 1: Empty squares (most likely to have missed a piece)
            # Priority 2: Low confidence squares
            # Priority 3: Edge/corner squares (alignment issues)
            is_edge = row == 0 or row == 7 or col == 0 or col == 7
            is_empty = piece == "empty"
            is_low_conf = conf < 0.8
            
            if is_empty or is_low_conf or is_edge:
                candidates.append((row, col, is_empty, is_low_conf, is_edge))
    
    print(f"[KING SEARCH] Looking for {missing_king} among {len(candidates)} candidates")
    
    # For each candidate, use offset search to find best king probability
    best_result = None
    best_king_prob = 0.0
    
    for row, col, is_empty, is_low_conf, is_edge in candidates:
        # Get predictions with offset search
        piece, conf, offset = predict_tile_with_offset_search(
            warped, row, col, tile_size, model
        )
        
        # Get the raw probabilities for this tile with the best offset
        h, w = warped.shape[:2]
        base_y, base_x = row * tile_size, col * tile_size
        
        if offset != (0, 0):
            dx_pct, dy_pct = offset
            dy = int(dy_pct * tile_size)
            dx = int(dx_pct * tile_size)
            oy1 = max(0, base_y + dy)
            oy2 = min(h, base_y + tile_size + dy)
            ox1 = max(0, base_x + dx)
            ox2 = min(w, base_x + tile_size + dx)
            tile = warped[oy1:oy2, ox1:ox2]
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                tile = cv2.resize(tile, (tile_size, tile_size))
        else:
            y1, y2 = base_y, base_y + tile_size
            x1, x2 = base_x, base_x + tile_size
            tile = warped[y1:y2, x1:x2]
        
        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(tile_rgb)
        img_tensor = transform(pil_img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            king_prob = probs[0, king_idx].item()
        
        if king_prob > best_king_prob:
            best_king_prob = king_prob
            best_result = (row, col, king_prob, offset)
            
        # Debug output for promising candidates
        if king_prob > 0.1:
            square = f"{'abcdefgh'[col]}{8-row}"
            print(f"  {square}: {missing_king} prob = {king_prob*100:.1f}%")
    
    if best_result and best_king_prob > 0.05:  # At least 5% confidence
        row, col, prob, offset = best_result
        square = f"{'abcdefgh'[col]}{8-row}"
        print(f"[KING SEARCH] Found {missing_king} at {square} with {prob*100:.1f}% confidence")
        return best_result
    
    print(f"[KING SEARCH] Could not find {missing_king}")
    return None


def validate_and_fix_kings(warped, board, confidences, model, tile_size):
    """
    Check for missing kings and attempt to find them.
    
    Args:
        warped: Warped board image
        board: 8x8 list of predictions (will be modified in place)
        confidences: 8x8 list of confidences (will be modified in place)
        model: Classification model
        tile_size: Tile size in pixels
    
    Returns:
        fixes_made: List of (square, piece) fixes applied
    """
    # Count kings
    wK_count = sum(1 for r in range(8) for c in range(8) if board[r][c] == "wK")
    bK_count = sum(1 for r in range(8) for c in range(8) if board[r][c] == "bK")
    
    fixes_made = []
    
    # Check for missing white king
    if wK_count == 0:
        print("[VALIDATION] Missing white king - searching...")
        result = find_missing_king(warped, board, confidences, "wK", model, tile_size)
        if result:
            row, col, prob, offset = result
            old_piece = board[row][col]
            board[row][col] = "wK"
            confidences[row][col] = prob
            square = f"{'abcdefgh'[col]}{8-row}"
            fixes_made.append((square, "wK", old_piece))
            print(f"[VALIDATION] Fixed: {square} {old_piece} -> wK")
    
    # Check for missing black king
    if bK_count == 0:
        print("[VALIDATION] Missing black king - searching...")
        result = find_missing_king(warped, board, confidences, "bK", model, tile_size)
        if result:
            row, col, prob, offset = result
            old_piece = board[row][col]
            board[row][col] = "bK"
            confidences[row][col] = prob
            square = f"{'abcdefgh'[col]}{8-row}"
            fixes_made.append((square, "bK", old_piece))
            print(f"[VALIDATION] Fixed: {square} {old_piece} -> bK")
    
    return fixes_made


def fix_duplicate_queens(warped, board, confidences, model, tile_size):
    """
    If there are 2+ queens of the same color, try to fix by checking if one should be the other color.
    In standard chess, each side has at most 1 queen (promotions are rare).
    
    Key insight: Having one white queen and one black queen is FAR more common than having
    two queens of the same color. So when uncertain, prefer keeping opposite colors.
    
    Args:
        warped: Warped board image
        board: 8x8 list of predictions (will be modified in place)
        confidences: 8x8 list of confidences (will be modified in place)
        model: Classification model
        tile_size: Tile size in pixels
    
    Returns:
        tuple: (fixes_made, questionable_squares)
            - fixes_made: List of fixes applied
            - questionable_squares: List of squares that might have color issues
    """
    fixes_made = []
    questionable_squares = []
    
    # Find all queens
    wQ_positions = [(r, c, confidences[r][c]) for r in range(8) for c in range(8) if board[r][c] == "wQ"]
    bQ_positions = [(r, c, confidences[r][c]) for r in range(8) for c in range(8) if board[r][c] == "bQ"]
    
    total_queens = len(wQ_positions) + len(bQ_positions)
    
    # Special case: Exactly 2 queens detected, both same color
    # In this case, it's much more likely that one should be the opposite color
    # (having 1 wQ + 1 bQ is far more common than 2 wQ or 2 bQ)
    if total_queens == 2:
        if len(wQ_positions) == 2 and len(bQ_positions) == 0:
            print(f"[QUEEN FIX] Found 2 white queens, 0 black - likely one is black")
            # Sort by confidence - the lower confidence one is more likely misidentified
            wQ_positions.sort(key=lambda x: x[2])
            lowest_conf_row, lowest_conf_col, lowest_conf = wQ_positions[0]
            highest_conf_row, highest_conf_col, highest_conf = wQ_positions[1]
            
            square_low = f"{'abcdefgh'[lowest_conf_col]}{8-lowest_conf_row}"
            square_high = f"{'abcdefgh'[highest_conf_col]}{8-highest_conf_row}"
            
            # Position-based heuristic: queen on opponent's back ranks more likely to be that color
            is_on_black_back_ranks = lowest_conf_row <= 1  # rows 0-1 are ranks 8-7
            
            print(f"[QUEEN FIX] Lower conf: {square_low} ({lowest_conf*100:.1f}%), Higher: {square_high} ({highest_conf*100:.1f}%)")
            
            # If confidence difference is large OR position strongly suggests black queen, swap
            conf_diff = highest_conf - lowest_conf
            if conf_diff > 0.15 or is_on_black_back_ranks:
                print(f"[QUEEN FIX] Strong evidence - changing {square_low} from wQ to bQ")
                board[lowest_conf_row][lowest_conf_col] = "bQ"
                fixes_made.append((square_low, "bQ", "wQ"))
            else:
                # Not confident enough to auto-fix, but prefer opposite colors
                # Change the lower confidence one and mark as questionable
                print(f"[QUEEN FIX] Weak evidence - changing {square_low} to bQ but marking questionable")
                board[lowest_conf_row][lowest_conf_col] = "bQ"
                fixes_made.append((square_low, "bQ", "wQ"))
                questionable_squares.append({
                    "square": square_low,
                    "piece": "bQ",
                    "reason": "Changed from wQ to bQ (2 same-color queens unlikely) - verify color"
                })
            return fixes_made, questionable_squares
        
        if len(bQ_positions) == 2 and len(wQ_positions) == 0:
            print(f"[QUEEN FIX] Found 2 black queens, 0 white - likely one is white")
            # Sort by confidence - the lower confidence one is more likely misidentified
            bQ_positions.sort(key=lambda x: x[2])
            lowest_conf_row, lowest_conf_col, lowest_conf = bQ_positions[0]
            highest_conf_row, highest_conf_col, highest_conf = bQ_positions[1]
            
            square_low = f"{'abcdefgh'[lowest_conf_col]}{8-lowest_conf_row}"
            square_high = f"{'abcdefgh'[highest_conf_col]}{8-highest_conf_row}"
            
            # Position-based heuristic: queen on opponent's back ranks more likely to be that color
            is_on_white_back_ranks = lowest_conf_row >= 6  # rows 6-7 are ranks 2-1
            
            print(f"[QUEEN FIX] Lower conf: {square_low} ({lowest_conf*100:.1f}%), Higher: {square_high} ({highest_conf*100:.1f}%)")
            
            # If confidence difference is large OR position strongly suggests white queen, swap
            conf_diff = highest_conf - lowest_conf
            if conf_diff > 0.15 or is_on_white_back_ranks:
                print(f"[QUEEN FIX] Strong evidence - changing {square_low} from bQ to wQ")
                board[lowest_conf_row][lowest_conf_col] = "wQ"
                fixes_made.append((square_low, "wQ", "bQ"))
            else:
                # Not confident enough to auto-fix, but prefer opposite colors
                # Change the lower confidence one and mark as questionable
                print(f"[QUEEN FIX] Weak evidence - changing {square_low} to wQ but marking questionable")
                board[lowest_conf_row][lowest_conf_col] = "wQ"
                fixes_made.append((square_low, "wQ", "bQ"))
                questionable_squares.append({
                    "square": square_low,
                    "piece": "wQ",
                    "reason": "Changed from bQ to wQ (2 same-color queens unlikely) - verify color"
                })
            return fixes_made, questionable_squares
    
    # Handle cases with 3+ queens of same color (original logic)
    # Check for duplicate white queens (more than one, when there's already a black queen)
    if len(wQ_positions) > 1:
        print(f"[QUEEN FIX] Found {len(wQ_positions)} white queens, {len(bQ_positions)} black queens")
        # The one with lowest confidence is most likely to be wrong
        wQ_positions.sort(key=lambda x: x[2])  # Sort by confidence ascending
        lowest_conf_row, lowest_conf_col, lowest_conf = wQ_positions[0]
        highest_conf = wQ_positions[-1][2]
        
        # Calculate confidence difference
        conf_diff = highest_conf - lowest_conf
        
        # Position-based heuristic: if a "white" queen is on ranks 7-8 (black's back ranks),
        # it's very likely to be a misidentified black queen
        is_on_black_back_ranks = lowest_conf_row <= 1  # rows 0-1 are ranks 8-7 (black's back ranks)
        
        square = f"{'abcdefgh'[lowest_conf_col]}{8-lowest_conf_row}"
        print(f"[QUEEN FIX] Checking {square}: conf_diff={conf_diff*100:.1f}%, on_black_ranks={is_on_black_back_ranks}")
        
        # Only swap if there's no black queen yet
        if len(bQ_positions) == 0:
            if conf_diff > 0.03 or is_on_black_back_ranks:
                print(f"[QUEEN FIX] Changing {square} from wQ to bQ")
                board[lowest_conf_row][lowest_conf_col] = "bQ"
                confidences[lowest_conf_row][lowest_conf_col] = lowest_conf
                fixes_made.append((square, "bQ", "wQ"))
            else:
                # Mark as questionable
                for r, c, _ in wQ_positions:
                    sq = f"{'abcdefgh'[c]}{8-r}"
                    questionable_squares.append({
                        "square": sq,
                        "piece": "wQ",
                        "reason": f"Multiple white queens detected ({len(wQ_positions)})"
                    })
    
    # Check for duplicate black queens
    if len(bQ_positions) > 1:
        print(f"[QUEEN FIX] Found {len(bQ_positions)} black queens, {len(wQ_positions)} white queens")
        bQ_positions.sort(key=lambda x: x[2])  # Sort by confidence ascending
        lowest_conf_row, lowest_conf_col, lowest_conf = bQ_positions[0]
        highest_conf = bQ_positions[-1][2]
        
        conf_diff = highest_conf - lowest_conf
        is_on_white_back_ranks = lowest_conf_row >= 6  # rows 6-7 are ranks 2-1
        
        square = f"{'abcdefgh'[lowest_conf_col]}{8-lowest_conf_row}"
        print(f"[QUEEN FIX] Checking {square}: conf_diff={conf_diff*100:.1f}%, on_white_ranks={is_on_white_back_ranks}")
        
        # Only swap if there's no white queen yet
        if len(wQ_positions) == 0:
            if conf_diff > 0.03 or is_on_white_back_ranks:
                print(f"[QUEEN FIX] Changing {square} from bQ to wQ")
                board[lowest_conf_row][lowest_conf_col] = "wQ"
                confidences[lowest_conf_row][lowest_conf_col] = lowest_conf
                fixes_made.append((square, "wQ", "bQ"))
            else:
                for r, c, _ in bQ_positions:
                    sq = f"{'abcdefgh'[c]}{8-r}"
                    questionable_squares.append({
                        "square": sq,
                        "piece": "bQ",
                        "reason": f"Multiple black queens detected ({len(bQ_positions)})"
                    })
    
    return fixes_made, questionable_squares


def classify_tile_at(img_bgr, x, y, tile_size, model):
    """
    Classify a tile at given position and return (class, probability, top3).
    Returns None if position is out of bounds.
    """
    h, w = img_bgr.shape[:2]
    if x < 0 or y < 0 or x + tile_size > w or y + tile_size > h:
        return None
    
    tile = img_bgr[y:y+tile_size, x:x+tile_size]
    tile_resized = cv2.resize(tile, (TILE_SIZE, TILE_SIZE))
    tile_rgb = cv2.cvtColor(tile_resized, cv2.COLOR_BGR2RGB)
    
    tensor = torch.from_numpy(tile_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    
    top_prob, top_idx = probs.max(0)
    pred_class = CLASSES[top_idx.item()]
    
    # Get top 3 for debugging
    top3_probs, top3_indices = probs.topk(3)
    top3 = [(CLASSES[i.item()], p.item()) for i, p in zip(top3_indices, top3_probs)]
    
    return pred_class, top_prob.item(), top3


def is_valid_piece(pred_class, prob, allow_empty=True):
    """Check if prediction looks like a valid chess piece (or empty square)."""
    if pred_class == 'empty':
        return allow_empty and prob > 0.3
    # Pieces should have reasonable confidence
    return prob > 0.4


def find_best_tile_at(img_bgr, x, y, tile_size, model, max_shift=None):
    """
    Find the best tile classification near position (x,y) by trying small offsets.
    
    Returns:
        (best_x, best_y, pred_class, prob) or None if nothing found
    """
    if max_shift is None:
        max_shift = tile_size // 3
    
    step = max(2, tile_size // 10)
    best = None
    best_prob = 0
    
    for dy in range(-max_shift, max_shift + 1, step):
        for dx in range(-max_shift, max_shift + 1, step):
            nx, ny = x + dx, y + dy
            result = classify_tile_at(img_bgr, nx, ny, tile_size, model)
            if result is None:
                continue
            
            pred_class, prob, _ = result
            
            # We want high confidence predictions (piece or empty)
            if prob > best_prob:
                best = (nx, ny, pred_class, prob)
                best_prob = prob
    
    return best


def find_complete_line(img_bgr, anchor_x, anchor_y, tile_size, horizontal, model, debug=False):
    """
    From anchor, walk in both directions along a line to find all 8 squares.
    
    This walks left+right (horizontal=True) or up+down (horizontal=False),
    refining position at each step until we find 8 squares total.
    
    Returns:
        list of 8 (x, y, pred_class, prob) tuples if found, else None
        The list is ordered from left-to-right or top-to-bottom
    """
    h, w = img_bgr.shape[:2]
    
    if horizontal:
        dx1, dy1 = -1, 0  # left
        dx2, dy2 = 1, 0   # right
        dir_name = "horizontal"
    else:
        dx1, dy1 = 0, -1  # up
        dx2, dy2 = 0, 1   # down
        dir_name = "vertical"
    
    if debug:
        print(f"  Finding {dir_name} line from ({anchor_x},{anchor_y})...")
    
    # Start with anchor
    tiles = [(anchor_x, anchor_y, None, 0)]
    
    # Get anchor classification
    result = find_best_tile_at(img_bgr, anchor_x, anchor_y, tile_size, model)
    if result:
        tiles[0] = result
    
    # Walk in negative direction (left or up)
    curr_x, curr_y = anchor_x, anchor_y
    for step in range(1, 8):
        next_x = curr_x + dx1 * tile_size
        next_y = curr_y + dy1 * tile_size
        
        result = find_best_tile_at(img_bgr, next_x, next_y, tile_size, model)
        if result is None:
            break
        
        best_x, best_y, pred_class, prob = result
        if prob < 0.35:
            break
        
        tiles.insert(0, result)  # Prepend
        curr_x, curr_y = best_x, best_y
        
        if debug:
            print(f"    <- Step {step}: {pred_class} ({prob*100:.0f}%) at ({best_x},{best_y})")
    
    # Walk in positive direction (right or down)
    curr_x, curr_y = anchor_x, anchor_y
    for step in range(1, 8):
        next_x = curr_x + dx2 * tile_size
        next_y = curr_y + dy2 * tile_size
        
        result = find_best_tile_at(img_bgr, next_x, next_y, tile_size, model)
        if result is None:
            break
        
        best_x, best_y, pred_class, prob = result
        if prob < 0.35:
            break
        
        tiles.append(result)
        curr_x, curr_y = best_x, best_y
        
        if debug:
            print(f"    -> Step {step}: {pred_class} ({prob*100:.0f}%) at ({best_x},{best_y})")
    
    if debug:
        print(f"    Found {len(tiles)} tiles")
    
    # Check if we have exactly 8 (or close to it)
    if len(tiles) >= 7:
        return tiles
    
    return None


def expand_grid_from_anchor(img_bgr, anchor_x, anchor_y, tile_size, model, debug=False):
    """
    From an anchor piece, find a complete row or column, then derive the full board.
    
    Strategy:
    1. Try to find a complete horizontal row (8 tiles)
    2. If found, we know the left edge and tile positions
    3. Then find where the vertical edges are
    4. If horizontal fails, try vertical column instead
    
    Returns:
        grid_info: dict with {left, right, top, bottom, tile_size} or None
    """
    h, w = img_bgr.shape[:2]
    
    if debug:
        print(f"[LINE] Starting from anchor ({anchor_x},{anchor_y}), tile={tile_size}")
    
    # Try horizontal first
    h_tiles = find_complete_line(img_bgr, anchor_x, anchor_y, tile_size, 
                                  horizontal=True, model=model, debug=debug)
    
    if h_tiles and len(h_tiles) >= 7:
        # Found horizontal row! 
        # The leftmost tile gives us grid_left
        # Now find vertical extent using the leftmost tile's x position
        grid_left = h_tiles[0][0]
        row_y = h_tiles[0][1]
        
        if debug:
            print(f"[LINE] Found horizontal row of {len(h_tiles)}, left={grid_left}")
        
        # Find vertical extent from the leftmost column
        v_tiles = find_complete_line(img_bgr, grid_left, row_y, tile_size,
                                      horizontal=False, model=model, debug=debug)
        
        if v_tiles and len(v_tiles) >= 7:
            grid_top = v_tiles[0][1]
            
            # We have the board!
            grid_right = grid_left + 8 * tile_size
            grid_bottom = grid_top + 8 * tile_size
            
            # Clamp to image
            if grid_right > w:
                grid_left = max(0, w - 8 * tile_size)
                grid_right = w
            if grid_bottom > h:
                grid_top = max(0, h - 8 * tile_size)
                grid_bottom = h
            
            if debug:
                print(f"[LINE] SUCCESS! Grid: ({grid_left},{grid_top}) to ({grid_right},{grid_bottom})")
            
            return {
                'left': grid_left,
                'top': grid_top,
                'right': grid_right,
                'bottom': grid_bottom,
                'tile_size': tile_size,
                'h_span': len(h_tiles),
                'v_span': len(v_tiles)
            }
    
    # Try vertical first if horizontal didn't work
    v_tiles = find_complete_line(img_bgr, anchor_x, anchor_y, tile_size,
                                  horizontal=False, model=model, debug=debug)
    
    if v_tiles and len(v_tiles) >= 7:
        # Found vertical column!
        grid_top = v_tiles[0][1]
        col_x = v_tiles[0][0]
        
        if debug:
            print(f"[LINE] Found vertical column of {len(v_tiles)}, top={grid_top}")
        
        # Find horizontal extent from the top row
        h_tiles = find_complete_line(img_bgr, col_x, grid_top, tile_size,
                                      horizontal=True, model=model, debug=debug)
        
        if h_tiles and len(h_tiles) >= 7:
            grid_left = h_tiles[0][0]
            
            grid_right = grid_left + 8 * tile_size
            grid_bottom = grid_top + 8 * tile_size
            
            if grid_right > w:
                grid_left = max(0, w - 8 * tile_size)
                grid_right = w
            if grid_bottom > h:
                grid_top = max(0, h - 8 * tile_size)
                grid_bottom = h
            
            if debug:
                print(f"[LINE] SUCCESS! Grid: ({grid_left},{grid_top}) to ({grid_right},{grid_bottom})")
            
            return {
                'left': grid_left,
                'top': grid_top,
                'right': grid_right,
                'bottom': grid_bottom,
                'tile_size': tile_size,
                'h_span': len(h_tiles),
                'v_span': len(v_tiles)
            }
    
    if debug:
        print(f"[LINE] Could not find complete row or column")
    
    return None


def score_grid_alignment(img_bgr, grid_left, grid_top, tile_size, model):
    """
    Score how well an 8x8 grid at this position aligns with chess pieces.
    Returns (score, piece_count) where score is sum of confidences and
    piece_count is number of non-empty squares with good confidence.
    """
    total_score = 0
    piece_count = 0
    empty_count = 0
    
    for row in range(8):
        for col in range(8):
            x = grid_left + col * tile_size
            y = grid_top + row * tile_size
            
            result = classify_tile_at(img_bgr, x, y, tile_size, model)
            if result is None:
                return 0, 0  # Grid extends outside image
            
            pred_class, prob, _ = result
            
            if pred_class == 'empty':
                if prob > 0.4:
                    total_score += prob * 0.5  # Empty squares count less
                    empty_count += 1
            else:
                if prob > 0.3:
                    total_score += prob
                    piece_count += 1
    
    # Penalize if too many or too few pieces (typical game has 16-32 pieces)
    if piece_count < 4 or piece_count > 40:
        total_score *= 0.5
    
    return total_score, piece_count


def predict_board_at_position(img_bgr, grid_left, grid_top, tile_size, model):
    """
    Predict full 8x8 board at given position.
    Returns (board_2d, avg_confidence, piece_count) or None if invalid.
    """
    h, w = img_bgr.shape[:2]
    
    board = []
    total_conf = 0
    piece_count = 0
    
    for row in range(8):
        board_row = []
        for col in range(8):
            x = grid_left + col * tile_size
            y = grid_top + row * tile_size
            
            result = classify_tile_at(img_bgr, x, y, tile_size, model)
            if result is None:
                return None  # Off image
            
            pred_class, prob, _ = result
            board_row.append(pred_class)
            total_conf += prob
            if pred_class != 'empty':
                piece_count += 1
        
        board.append(board_row)
    
    avg_conf = total_conf / 64
    return board, avg_conf, piece_count


def is_valid_chess_position(board, piece_count):
    """
    Check if a board position is plausibly valid chess.
    Returns (is_valid, reason).
    """
    # Count pieces
    counts = {}
    for row in board:
        for piece in row:
            counts[piece] = counts.get(piece, 0) + 1
    
    # Must have exactly one white king and one black king
    if counts.get('wK', 0) != 1:
        return False, f"white kings: {counts.get('wK', 0)}"
    if counts.get('bK', 0) != 1:
        return False, f"black kings: {counts.get('bK', 0)}"
    
    # Reasonable piece counts (max in standard chess)
    if counts.get('wQ', 0) > 9:  # 1 original + 8 promoted
        return False, "too many white queens"
    if counts.get('bQ', 0) > 9:
        return False, "too many black queens"
    if counts.get('wR', 0) > 10:
        return False, "too many white rooks"
    if counts.get('bR', 0) > 10:
        return False, "too many black rooks"
    
    # Total pieces should be reasonable (2 to 32)
    if piece_count < 2:
        return False, "too few pieces"
    if piece_count > 32:
        return False, "too many pieces"
    
    return True, "valid"


def find_board_from_high_confidence_pieces(img_bgr, model, debug=False):
    """
    Find board by locating high-confidence pieces and deriving board candidates.
    
    Strategy:
    1. Scan image for pieces with very high confidence (>90%)
    2. For each high-confidence piece found, it constrains the board position
    3. Generate board candidates based on where the piece could be (any of 64 squares)
    4. But only try positions where piece is on valid square given its location
    5. Score each candidate by predicting full board and checking validity
    
    Args:
        img_bgr: Input image in BGR format  
        model: PyTorch model for classification
        debug: Include debug info
    
    Returns:
        quad: 4 corner points of detected board (or None if not found)
        debug_info: dict with search details if debug=True
    """
    import time
    start_time = time.time()
    
    h, w = img_bgr.shape[:2]
    debug_info = {"method": "high_confidence_pieces"} if debug else None
    
    print(f"[HC DETECT] Image size: {w}x{h}")
    print(f"[HC DETECT] Looking for high-confidence pieces...")
    
    # Try different tile sizes
    min_dim = min(h, w)
    tile_sizes = []
    for divisor in range(8, 14):
        tile_size = min_dim // divisor
        if 40 <= tile_size <= 120:
            tile_sizes.append(tile_size)
    
    tile_sizes = sorted(set(tile_sizes), reverse=True)
    
    for tile_size in tile_sizes:
        print(f"[HC DETECT] Trying tile_size={tile_size}...")
        
        high_conf_pieces = []
        
        # Scan with 50% overlap for speed
        step = tile_size // 2
        
        for y in range(0, h - tile_size, step):
            for x in range(0, w - tile_size, step):
                result = classify_tile_at(img_bgr, x, y, tile_size, model)
                if result is None:
                    continue
                
                pred_class, prob, _ = result
                
                # Only high confidence pieces (not empty)
                if pred_class != 'empty' and prob > 0.85:
                    # Refine position
                    best = find_best_tile_at(img_bgr, x, y, tile_size, model, 
                                              max_shift=tile_size // 4)
                    if best and best[3] > 0.85:
                        high_conf_pieces.append({
                            'x': best[0],
                            'y': best[1],
                            'class': best[2],
                            'prob': best[3]
                        })
                        if debug:
                            print(f"  Found {best[2]} ({best[3]*100:.0f}%) at ({best[0]},{best[1]})")
        
        if len(high_conf_pieces) < 2:
            print(f"  Only {len(high_conf_pieces)} high-confidence pieces, need at least 2")
            continue
        
        print(f"  Found {len(high_conf_pieces)} high-confidence pieces")
        
        # Use the highest confidence piece as anchor
        high_conf_pieces.sort(key=lambda p: -p['prob'])
        anchor = high_conf_pieces[0]
        
        # The anchor piece could be on any of the 64 squares
        # But we can limit based on where it is in the image
        # If anchor is at (ax, ay), and board is at (bx, by), then:
        # ax = bx + col * tile_size, ay = by + row * tile_size
        # So: bx = ax - col * tile_size, by = ay - row * tile_size
        
        board_candidates = []
        
        for row in range(8):
            for col in range(8):
                grid_left = anchor['x'] - col * tile_size
                grid_top = anchor['y'] - row * tile_size
                
                # Check bounds
                grid_right = grid_left + 8 * tile_size
                grid_bottom = grid_top + 8 * tile_size
                
                if grid_left < -tile_size//2 or grid_top < -tile_size//2:
                    continue
                if grid_right > w + tile_size//2 or grid_bottom > h + tile_size//2:
                    continue
                
                # Clamp to image
                grid_left = max(0, grid_left)
                grid_top = max(0, grid_top)
                
                board_candidates.append((grid_left, grid_top, row, col))
        
        print(f"  Generated {len(board_candidates)} board candidates")
        
        # Score each candidate
        best_candidate = None
        best_score = 0
        
        for grid_left, grid_top, anchor_row, anchor_col in board_candidates:
            result = predict_board_at_position(img_bgr, grid_left, grid_top, tile_size, model)
            if result is None:
                continue
            
            board, avg_conf, piece_count = result
            
            # Check if valid position
            is_valid, reason = is_valid_chess_position(board, piece_count)
            
            if not is_valid:
                continue
            
            # Score: average confidence + bonus for piece count in reasonable range
            score = avg_conf
            if 8 <= piece_count <= 32:
                score += 0.1
            
            if score > best_score:
                best_score = score
                best_candidate = (grid_left, grid_top, board, avg_conf, piece_count)
        
        if best_candidate:
            grid_left, grid_top, board, avg_conf, piece_count = best_candidate
            grid_right = grid_left + 8 * tile_size
            grid_bottom = grid_top + 8 * tile_size
            
            quad = np.array([
                [grid_left, grid_top],
                [grid_right, grid_top],
                [grid_right, grid_bottom],
                [grid_left, grid_bottom],
            ], dtype=np.float32)
            
            elapsed = time.time() - start_time
            print(f"[HC DETECT] SUCCESS in {elapsed:.2f}s!")
            print(f"  Grid: ({grid_left},{grid_top}) to ({grid_right},{grid_bottom})")
            print(f"  Avg confidence: {avg_conf*100:.1f}%, Pieces: {piece_count}")
            
            if debug:
                debug_info["success"] = True
                debug_info["elapsed_seconds"] = elapsed
                debug_info["grid_left"] = grid_left
                debug_info["grid_top"] = grid_top
                debug_info["tile_size"] = tile_size
                debug_info["avg_confidence"] = avg_conf
                debug_info["piece_count"] = piece_count
            
            return quad, debug_info
        
        print(f"  No valid board found at this tile size")
        
        # Timeout
        if time.time() - start_time > 10:
            print(f"[HC DETECT] Timeout")
            break
    
    elapsed = time.time() - start_time
    print(f"[HC DETECT] FAILED after {elapsed:.2f}s")
    
    if debug:
        debug_info["success"] = False
        debug_info["elapsed_seconds"] = elapsed
    
    return None, debug_info


def refine_tile_position(img_bgr, x, y, tile_size, model, piece_class):
    """
    Fine-tune tile position by shifting small amounts to maximize confidence.
    
    When we find a piece at moderate confidence (e.g. 60%), shifting by a few
    pixels might get us to 90%+ - indicating we found the true tile alignment.
    
    Returns:
        (best_x, best_y, best_prob) or None if can't improve significantly
    """
    best_x, best_y = x, y
    best_prob = 0
    
    # Initial classification
    result = classify_tile_at(img_bgr, x, y, tile_size, model)
    if result:
        _, best_prob, _ = result
    
    # Try shifting by small amounts (up to 1/4 tile size)
    max_shift = tile_size // 4
    step = max(2, tile_size // 16)  # Smaller steps for finer tuning
    
    for dy in range(-max_shift, max_shift + 1, step):
        for dx in range(-max_shift, max_shift + 1, step):
            nx, ny = x + dx, y + dy
            result = classify_tile_at(img_bgr, nx, ny, tile_size, model)
            if result is None:
                continue
            
            pred_class, prob, _ = result
            
            # Only consider if it's the same piece type
            if pred_class == piece_class and prob > best_prob:
                best_x, best_y = nx, ny
                best_prob = prob
    
    return best_x, best_y, best_prob


def find_board_from_center(img_bgr, model, debug=False):
    """
    Board detection by starting from image center and finding pieces.
    
    Strategy:
    1. Assume center of image is somewhere on the board
    2. Try different tile sizes, starting from center
    3. When we find a piece with moderate confidence, fine-tune position
    4. Once we have high confidence (>85%), we know the tile size is right
    5. Expand from that anchor to find the full 8x8 grid
    
    Args:
        img_bgr: Input image in BGR format
        model: PyTorch model for classification
        debug: Include debug info
    
    Returns:
        quad: 4 corner points of detected board (or None if not found)
        debug_info: dict with search details if debug=True
    """
    import time
    start_time = time.time()
    
    h, w = img_bgr.shape[:2]
    debug_info = {"method": "center_piece_expansion"} if debug else None
    
    print(f"[PIECE DETECT] Image size: {w}x{h}")
    print(f"[PIECE DETECT] Starting from center, looking for pieces...")
    
    # Center of image
    cx, cy = w // 2, h // 2
    
    # Try tile sizes from large to small
    min_dim = min(h, w)
    tile_sizes = []
    for divisor in range(8, 16):
        tile_size = min_dim // divisor
        if 30 <= tile_size <= 150:
            tile_sizes.append(tile_size)
    
    tile_sizes = sorted(set(tile_sizes), reverse=True)
    print(f"[PIECE DETECT] Tile sizes to try: {tile_sizes}")
    
    classifications = 0
    
    for tile_size in tile_sizes:
        print(f"[PIECE DETECT] Trying tile_size={tile_size}...")
        
        # Sample positions around center - spiral outward
        # Start at center, then check positions in expanding squares
        positions = [(cx - tile_size//2, cy - tile_size//2)]  # Center tile
        
        # Add positions in expanding rings around center
        for ring in range(1, 6):  # Up to 5 rings out
            offset = ring * tile_size
            # Top and bottom edges of ring
            for dx in range(-ring, ring + 1):
                positions.append((cx - tile_size//2 + dx * tile_size, cy - tile_size//2 - offset))
                positions.append((cx - tile_size//2 + dx * tile_size, cy - tile_size//2 + offset))
            # Left and right edges of ring (excluding corners)
            for dy in range(-ring + 1, ring):
                positions.append((cx - tile_size//2 - offset, cy - tile_size//2 + dy * tile_size))
                positions.append((cx - tile_size//2 + offset, cy - tile_size//2 + dy * tile_size))
        
        for x, y in positions:
            result = classify_tile_at(img_bgr, x, y, tile_size, model)
            classifications += 1
            
            if result is None:
                continue
            
            pred_class, prob, top3 = result
            
            # Look for any piece (not empty) with at least 40% confidence
            if pred_class != 'empty' and prob > 0.4:
                print(f"  Found candidate: {pred_class} ({prob*100:.0f}%) at ({x},{y})")
                
                # Fine-tune the position to maximize confidence
                ref_x, ref_y, ref_prob = refine_tile_position(
                    img_bgr, x, y, tile_size, model, pred_class
                )
                classifications += (tile_size // max(2, tile_size // 16)) ** 2  # Approximate
                
                print(f"  Refined: {pred_class} ({ref_prob*100:.0f}%) at ({ref_x},{ref_y})")
                
                # If we got high confidence, we found the right tile size and alignment
                if ref_prob > 0.75:
                    print(f"  HIGH CONFIDENCE! Expanding grid from this anchor...")
                    
                    # Expand from this anchor
                    grid_info = expand_grid_from_anchor(
                        img_bgr, ref_x, ref_y, tile_size, model, debug=True
                    )
                    
                    if grid_info is not None:
                        # Validate
                        score, piece_count = score_grid_alignment(
                            img_bgr, grid_info['left'], grid_info['top'],
                            tile_size, model
                        )
                        
                        print(f"  Grid: score={score:.1f}, pieces={piece_count}")
                        
                        if score > 12 and piece_count >= 2:
                            x1, y1 = grid_info['left'], grid_info['top']
                            x2, y2 = grid_info['right'], grid_info['bottom']
                            
                            quad = np.array([
                                [x1, y1],
                                [x2, y1],
                                [x2, y2],
                                [x1, y2],
                            ], dtype=np.float32)
                            
                            elapsed = time.time() - start_time
                            print(f"[PIECE DETECT] SUCCESS in {elapsed:.2f}s!")
                            print(f"  Anchor: {pred_class} at ({ref_x},{ref_y}) conf={ref_prob*100:.0f}%")
                            print(f"  Grid: ({x1},{y1}) to ({x2},{y2}), tile={tile_size}")
                            print(f"  Classifications: ~{classifications}")
                            
                            if debug:
                                debug_info["success"] = True
                                debug_info["elapsed_seconds"] = elapsed
                                debug_info["anchor_piece"] = pred_class
                                debug_info["anchor_pos"] = (ref_x, ref_y)
                                debug_info["anchor_confidence"] = ref_prob
                                debug_info["tile_size"] = tile_size
                                debug_info["grid"] = grid_info
                                debug_info["score"] = score
                            
                            return quad, debug_info
            
            # Timeout check
            if time.time() - start_time > 15:
                print(f"[PIECE DETECT] Timeout!")
                break
        
        # Check if we should continue to next tile size
        if time.time() - start_time > 15:
            break
    
    elapsed = time.time() - start_time
    print(f"[PIECE DETECT] FAILED after {elapsed:.2f}s, ~{classifications} classifications")
    
    if debug:
        debug_info["success"] = False
        debug_info["elapsed_seconds"] = elapsed
    
    return None, debug_info


def find_board_from_kings(img_bgr, model, debug=False):
    """
    Try multiple fallback methods to detect the board.
    
    Methods tried in order:
    1. High-confidence piece detection - find pieces with >85% confidence,
       generate board candidates, validate with FEN
    2. Center-based expansion - start from center, find pieces, expand to grid
    """
    print("[FALLBACK] Trying high-confidence piece method...")
    quad, debug_info = find_board_from_high_confidence_pieces(img_bgr, model, debug)
    if quad is not None:
        return quad, debug_info
    
    print("[FALLBACK] Trying center-based expansion method...")
    quad, debug_info = find_board_from_center(img_bgr, model, debug)
    if quad is not None:
        return quad, debug_info
    
    print("[FALLBACK] All methods failed")
    return None, debug_info if debug_info else {"method": "all_failed"}


def get_piece_probability(warped, row, col, tile_size, model, piece_class):
    """Get the probability for a specific piece class at a position."""
    h, w = warped.shape[:2]
    y1, y2 = row * tile_size, (row + 1) * tile_size
    x1, x2 = col * tile_size, (col + 1) * tile_size
    tile = warped[y1:y2, x1:x2]
    
    tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(tile_rgb)
    img_tensor = transform(pil_img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        piece_idx = CLASSES.index(piece_class)
        return probs[0, piece_idx].item()


def detect_board_with_validation(img_bgr, model, debug=False):
    """
    Detect board using multiple candidates and validate with model predictions.
    Returns the best detection that passes validation.
    
    Args:
        img_bgr: Input image in BGR format
        model: PyTorch model for classification
        debug: Include debug info
    
    Returns:
        warped: Warped board image (or None if all candidates fail)
        board: 8x8 piece predictions
        confidences: 8x8 confidence values
        avg_confidence: average confidence
        debug_info: dict with detection details if debug=True
    """
    from detect_board import _candidate_quads_from_edges, combined_grid_score
    
    debug_info = {} if debug else None
    
    # Get all candidate quads
    candidates = _candidate_quads_from_edges(img_bgr)
    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    
    # Filter and score candidates
    scored_candidates = []
    for quad in candidates:
        area = cv2.contourArea(quad)
        rel_area = area / img_area
        if rel_area < 0.20:
            continue
        
        rect = cv2.minAreaRect(quad)
        (_, _), (bw, bh), _ = rect
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect > 1.15:
            continue
        
        warped = warp_quad(img_bgr, quad, out_size=256)
        grid_score = combined_grid_score(warped)
        
        if grid_score < 0.15:
            continue
        
        scored_candidates.append((grid_score, quad, rel_area))
    
    # Sort by grid score (highest first)
    scored_candidates.sort(key=lambda x: -x[0])
    
    if debug:
        debug_info["num_candidates"] = len(scored_candidates)
        debug_info["tried_candidates"] = []
    
    # Try candidates in order of grid score, validate with model
    for i, (grid_score, quad, rel_area) in enumerate(scored_candidates[:5]):  # Try up to 5
        warped = warp_quad(img_bgr, quad, out_size=WARP_SIZE)
        tile_size = TILE_SIZE
        
        board, confidences, avg_confidence, low_conf_count = predict_tiles_with_confidence(
            warped, model, tile_size
        )
        
        is_valid, reason = validate_board_detection(board, confidences, avg_confidence, low_conf_count)
        
        if debug:
            debug_info["tried_candidates"].append({
                "index": i,
                "grid_score": grid_score,
                "rel_area": rel_area,
                "avg_confidence": avg_confidence,
                "low_conf_count": low_conf_count,
                "is_valid": is_valid,
                "reason": reason
            })
        
        if is_valid:
            if debug:
                debug_info["selected_index"] = i
            return warped, quad, board, confidences, avg_confidence, debug_info
    
    # No valid candidate found, return best one anyway with warning
    if scored_candidates:
        best_grid_score, best_quad, _ = scored_candidates[0]
        warped = warp_quad(img_bgr, best_quad, out_size=WARP_SIZE)
        tile_size = TILE_SIZE
        board, confidences, avg_confidence, low_conf_count = predict_tiles_with_confidence(
            warped, model, tile_size
        )
        if debug:
            debug_info["selected_index"] = 0
            debug_info["warning"] = "No candidate passed validation, using best grid score"
        return warped, best_quad, board, confidences, avg_confidence, debug_info
    
    # FALLBACK: No edge-based candidates found - try king-based detection
    print("[DETECTION] No edge candidates found, trying king-based fallback...")
    king_quad, king_debug = find_board_from_kings(img_bgr, model, debug=debug)
    
    if king_quad is not None:
        warped = warp_quad(img_bgr, king_quad, out_size=WARP_SIZE)
        tile_size = TILE_SIZE
        board, confidences, avg_confidence, low_conf_count = predict_tiles_with_confidence(
            warped, model, tile_size
        )
        if debug:
            debug_info["king_fallback"] = king_debug
            debug_info["warning"] = "Used king-based fallback detection"
        return warped, king_quad, board, confidences, avg_confidence, debug_info
    
    return None, None, None, None, None, debug_info


def validate_fen(fen: str) -> bool:
    """
    Validate a FEN string (board position only, not full FEN).

    Returns True if the FEN is valid (8 rows, 8 squares each, exactly 1 king per side).
    """
    try:
        rows = fen.split('/')

        if len(rows) != 8:
            return False

        white_kings = 0
        black_kings = 0

        for row in rows:
            count = 0
            for char in row:
                if char.isdigit():
                    count += int(char)
                elif char in 'pnbrqkPNBRQK':
                    count += 1
                    if char == 'K':
                        white_kings += 1
                    elif char == 'k':
                        black_kings += 1
                else:
                    return False

            if count != 8:
                return False

        # Must have exactly 1 king per side
        return white_kings == 1 and black_kings == 1
    except Exception:
        return False

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
    
    # Apply consistent preprocessing to all images
    img_bgr = preprocess_image(img_bgr)
    
    if skip_detection:
        # Treat entire image as the board - just resize to WARP_SIZE
        warped = cv2.resize(img_bgr, (WARP_SIZE, WARP_SIZE), interpolation=cv2.INTER_AREA)
        tile_size = TILE_SIZE
        
        if debug:
            # No detection overlay when skipping detection
            result["debug_images"]["detection"] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            result["debug_images"]["warped"] = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        
        # Classify tiles
        board, confidences, avg_confidence, low_conf_count = predict_tiles_with_confidence(
            warped, model, tile_size
        )
    else:
        # Use validated detection (tries multiple candidates)
        warped, quad, board, confidences, avg_confidence, detection_debug = detect_board_with_validation(
            img_bgr, model, debug=debug
        )
        
        if warped is None:
            result["error"] = "Could not detect chess board"
            return result
        
        tile_size = TILE_SIZE
        
        if debug:
            # Board detection overlay
            overlay = img_bgr.copy()
            pts = quad.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(overlay, [pts], True, (0, 255, 0), 3)
            result["debug_images"]["detection"] = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            result["debug_images"]["warped"] = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            
            # Add detection validation info
            if detection_debug:
                result["detection_info"] = detection_debug
    
    # Add confidence info to result
    result["avg_confidence"] = avg_confidence
    
    # Validate and fix missing kings
    fixes = validate_and_fix_kings(warped, board, confidences, model, tile_size)
    if fixes:
        result["king_fixes"] = fixes
        # Recalculate average confidence after fixes
        all_confs = [confidences[r][c] for r in range(8) for c in range(8)]
        result["avg_confidence"] = sum(all_confs) / len(all_confs)
    
    # Fix duplicate queens (e.g., 2 black queens when there should be 1 of each)
    queen_fixes, questionable_squares = fix_duplicate_queens(warped, board, confidences, model, tile_size)
    if queen_fixes:
        result["queen_fixes"] = queen_fixes
        # Recalculate average confidence after fixes
        all_confs = [confidences[r][c] for r in range(8) for c in range(8)]
        result["avg_confidence"] = sum(all_confs) / len(all_confs)
    
    if questionable_squares:
        result["questionable_squares"] = questionable_squares
    
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
    result["confidences"] = confidences  # 8x8 array of confidence values
    
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

def get_user_info_from_headers():
    """Extract user information from Azure AD authentication headers."""
    try:
        # Azure Container Apps Easy Auth provides user info in X-MS-CLIENT-PRINCIPAL header
        principal_header = request.headers.get('X-MS-CLIENT-PRINCIPAL')
        if principal_header:
            # Decode base64 encoded JSON
            principal_json = base64.b64decode(principal_header).decode('utf-8')
            principal = json.loads(principal_json)

            user_id = principal.get('userId') or principal.get('sub')
            user_email = None

            # Extract email from claims
            claims = principal.get('claims', [])
            for claim in claims:
                if claim.get('typ') in ['emails', 'email', 'preferred_username']:
                    user_email = claim.get('val')
                    break

            # Fallback: use name if no email found
            if not user_email:
                user_email = principal.get('userDetails') or user_id

            print(f"[AUTH] Extracted user: {user_email} (ID: {user_id})")
            return user_id, user_email
    except Exception as e:
        print(f"[AUTH] Error extracting user info: {e}")

    return None, None

def log_prediction(user_id: str = None, user_email: str = None, fen: str = None, valid_fen: bool = False):
    """Log a prediction event for statistics tracking."""
    try:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().isoformat()
        prediction_log = {
            "timestamp": timestamp,
            "user_id": user_id,
            "user_email": user_email,
            "valid_fen": valid_fen,
            "fen": fen
        }

        # Append to daily log file
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = PREDICTIONS_DIR / f"{date_str}.jsonl"

        with open(log_file, 'a') as f:
            f.write(json.dumps(prediction_log) + '\n')

        print(f"[PREDICTION] Logged prediction for user: {user_email} ({user_id}), valid_fen: {valid_fen}")
    except Exception as e:
        print(f"[PREDICTION] Error logging prediction: {e}")

def save_feedback(original_fen: str, corrected_fen: str, image_base64: str,
                  corrected_squares: dict, model_name: str,
                  user_id: str = None, user_email: str = None):
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
        "user_id": user_id,
        "user_email": user_email,
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

    print(f"[FEEDBACK] Saved feedback {feedback_id} from user: {user_email} ({user_id})")

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

    # Log prediction for statistics (only if successful)
    if result.get("success"):
        user_id, user_email = get_user_info_from_headers()
        fen = result.get("fen")
        valid_fen = validate_fen(fen) if fen else False
        log_prediction(user_id, user_email, fen, valid_fen)

    return jsonify(result)


@app.route('/api/predict/multi', methods=['POST'])
def predict_multi():
    """
    Run prediction with ALL available models and compare results.
    
    Request:
        - image: base64 encoded image
        - skip_detection: boolean (optional) - skip board detection
    
    Response:
        - success: boolean
        - results: list of {model, fen, board, success}
        - consensus: bool - true if all models agree on FEN
        - consensus_fen: string - FEN if all agree, null otherwise
        - disagreements: list of squares where models disagree
    """
    skip_detection = request.args.get('skip_detection', 'false').lower() == 'true'
    
    print(f"[PREDICT_MULTI] Received request, skip_detection={skip_detection}")
    
    # Get image from request
    img_bgr = None
    if request.json and 'image' in request.json:
        img_base64 = request.json['image']
        if ',' in img_base64 and img_base64.startswith('data:'):
            img_base64 = img_base64.split(',', 1)[1]
        try:
            img_data = base64.b64decode(img_base64)
            nparr = np.frombuffer(img_data, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                pil_img = Image.open(BytesIO(img_data))
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                img_rgb = np.array(pil_img)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            return jsonify({"success": False, "error": f"Decode error: {e}"}), 400
    else:
        return jsonify({"success": False, "error": "No image provided"}), 400
    
    if img_bgr is None:
        return jsonify({"success": False, "error": "Invalid image"}), 400
    
    # Get all available models
    versions = get_model_versions()
    results = []
    
    for ver in versions:
        model_name = ver['name']
        try:
            model = load_model(model_name)
            result = predict_board(img_bgr.copy(), model, debug=False, skip_detection=skip_detection)
            results.append({
                "model": model_name,
                "type": ver['type'],
                "accuracy": ver['accuracy'],
                "fen": result.get('fen'),
                "board": result.get('board'),
                "success": result.get('success', False)
            })
        except Exception as e:
            results.append({
                "model": model_name,
                "type": ver['type'],
                "accuracy": ver['accuracy'],
                "fen": None,
                "board": None,
                "success": False,
                "error": str(e)
            })
    
    # Check for consensus
    successful_fens = [r['fen'] for r in results if r['success'] and r['fen']]
    unique_fens = list(set(successful_fens))
    
    consensus = len(unique_fens) == 1 and len(successful_fens) == len(versions)
    consensus_fen = unique_fens[0] if consensus else None
    
    # Find disagreements (if any)
    disagreements = []
    if len(unique_fens) > 1:
        # Compare boards square by square
        successful_boards = [r['board'] for r in results if r['success'] and r['board']]
        if len(successful_boards) > 1:
            files = 'abcdefgh'
            for row in range(8):
                for col in range(8):
                    pieces = set(b[row][col] for b in successful_boards)
                    if len(pieces) > 1:
                        square = f"{files[col]}{8-row}"
                        disagreements.append({
                            "square": square,
                            "predictions": {r['model']: r['board'][row][col] for r in results if r['success'] and r['board']}
                        })
    
    return jsonify({
        "success": True,
        "results": results,
        "consensus": consensus,
        "consensus_fen": consensus_fen,
        "disagreements": disagreements,
        "model_count": len(versions),
        "successful_count": len(successful_fens)
    })


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit user correction feedback.

    Request:
        - original_fen: string
        - corrected_fen: string
        - image: base64 encoded image
        - corrected_squares: dict of {square: piece} corrections
        - user_id: string (from Azure AD)
        - user_email: string (from Azure AD)
    """
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data received"}), 400

    # Extract user info from Azure AD authentication headers
    user_id, user_email = get_user_info_from_headers()

    feedback_id = save_feedback(
        original_fen=data.get('original_fen'),
        corrected_fen=data.get('corrected_fen'),
        image_base64=data.get('image'),
        corrected_squares=data.get('corrected_squares', {}),
        model_name=_current_model_name,
        user_id=user_id,
        user_email=user_email
    )
    
    return jsonify({"success": True, "feedback_id": feedback_id})


@app.route('/api/feedback', methods=['GET'])
def list_feedback():
    """List pending feedback items (admin endpoint)."""
    return jsonify(get_pending_feedback())


def get_user_statistics():
    """Calculate user statistics from feedback data and prediction logs."""
    # Count predictions and valid FENs from log files
    total_predictions = 0
    valid_fen_count = 0
    if PREDICTIONS_DIR.exists():
        for log_file in PREDICTIONS_DIR.glob("*.jsonl"):
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        total_predictions += 1
                        try:
                            log_entry = json.loads(line)
                            if log_entry.get('valid_fen', False):
                                valid_fen_count += 1
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"[STATS] Error counting predictions in {log_file}: {e}")

    if not FEEDBACK_DIR.exists():
        return {
            "total_predictions": total_predictions,
            "valid_fen_count": valid_fen_count,
            "total_corrections": 0,
            "unique_users": 0,
            "users": []
        }

    # Aggregate statistics by user
    user_stats = {}

    for json_file in FEEDBACK_DIR.glob("*.json"):
        try:
            with open(json_file) as f:
                feedback = json.load(f)

            user_email = feedback.get('user_email', 'Unknown')
            user_id = feedback.get('user_id', 'unknown')

            if user_email not in user_stats:
                user_stats[user_email] = {
                    "email": user_email,
                    "user_id": user_id,
                    "corrections_count": 0,
                    "first_activity": feedback.get('timestamp'),
                    "last_activity": feedback.get('timestamp')
                }

            user_stats[user_email]["corrections_count"] += 1

            # Update activity timestamps
            timestamp = feedback.get('timestamp')
            if timestamp:
                if timestamp < user_stats[user_email]["first_activity"]:
                    user_stats[user_email]["first_activity"] = timestamp
                if timestamp > user_stats[user_email]["last_activity"]:
                    user_stats[user_email]["last_activity"] = timestamp

        except Exception as e:
            print(f"[STATS] Error processing {json_file}: {e}")
            continue

    # Convert to list and sort by corrections count
    users_list = sorted(user_stats.values(),
                       key=lambda x: x['corrections_count'],
                       reverse=True)

    return {
        "total_predictions": total_predictions,
        "valid_fen_count": valid_fen_count,
        "total_corrections": sum(u['corrections_count'] for u in users_list),
        "unique_users": len(users_list),
        "users": users_list
    }


@app.route('/api/admin/statistics', methods=['GET'])
def admin_statistics():
    """Get user statistics (admin endpoint)."""
    return jsonify(get_user_statistics())


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
