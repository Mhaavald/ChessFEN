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
    
    # Check for duplicate white queens
    if len(wQ_positions) > 1 and len(bQ_positions) == 0:
        print(f"[QUEEN FIX] Found {len(wQ_positions)} white queens, 0 black queens")
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
        rank = 8 - lowest_conf_row
        print(f"[QUEEN FIX] Confidence difference: {conf_diff*100:.1f}% (highest: {highest_conf*100:.1f}%, lowest: {lowest_conf*100:.1f}%)")
        print(f"[QUEEN FIX] Candidate {square} (rank {rank}) is {'on black back ranks' if is_on_black_back_ranks else 'NOT on black back ranks'}")
        
        # Apply fix if: significant confidence diff OR queen is on black's back ranks (7-8)
        if conf_diff > 0.03 or is_on_black_back_ranks:
            print(f"[QUEEN FIX] Changing {square} from wQ to bQ")
            board[lowest_conf_row][lowest_conf_col] = "bQ"
            confidences[lowest_conf_row][lowest_conf_col] = lowest_conf
            fixes_made.append((square, "bQ", "wQ"))
        else:
            print(f"[QUEEN FIX] Not changing - marking as questionable")
            # Mark ALL queens as questionable since we don't know which one is wrong
            for r, c, _ in wQ_positions:
                sq = f"{'abcdefgh'[c]}{8-r}"
                questionable_squares.append({
                    "square": sq,
                    "piece": "wQ",
                    "reason": "Detected 2 white queens, 0 black - one might be black queen"
                })
    
    # Check for duplicate black queens
    if len(bQ_positions) > 1 and len(wQ_positions) == 0:
        print(f"[QUEEN FIX] Found {len(bQ_positions)} black queens, 0 white queens")
        # The one with lowest confidence is most likely to be wrong
        bQ_positions.sort(key=lambda x: x[2])  # Sort by confidence ascending
        lowest_conf_row, lowest_conf_col, lowest_conf = bQ_positions[0]
        highest_conf = bQ_positions[-1][2]
        
        # Calculate confidence difference
        conf_diff = highest_conf - lowest_conf
        
        # Position-based heuristic: if a "black" queen is on ranks 1-2 (white's back ranks),
        # it's very likely to be a misidentified white queen (white rarely loses queen on back rank)
        is_on_white_back_ranks = lowest_conf_row >= 6  # rows 6-7 are ranks 2-1 (white's back ranks)
        
        square = f"{'abcdefgh'[lowest_conf_col]}{8-lowest_conf_row}"
        rank = 8 - lowest_conf_row
        print(f"[QUEEN FIX] Confidence difference: {conf_diff*100:.1f}% (highest: {highest_conf*100:.1f}%, lowest: {lowest_conf*100:.1f}%)")
        print(f"[QUEEN FIX] Candidate {square} (rank {rank}) is {'on white back ranks' if is_on_white_back_ranks else 'NOT on white back ranks'}")
        
        # Apply fix if: significant confidence diff OR queen is on white's back ranks (1-2)
        if conf_diff > 0.03 or is_on_white_back_ranks:
            print(f"[QUEEN FIX] Changing {square} from bQ to wQ")
            board[lowest_conf_row][lowest_conf_col] = "wQ"
            confidences[lowest_conf_row][lowest_conf_col] = lowest_conf
            fixes_made.append((square, "wQ", "bQ"))
        else:
            print(f"[QUEEN FIX] Not changing - marking as questionable")
            # Mark ALL queens as questionable since we don't know which one is wrong
            for r, c, _ in bQ_positions:
                sq = f"{'abcdefgh'[c]}{8-r}"
                questionable_squares.append({
                    "square": sq,
                    "piece": "bQ",
                    "reason": "Detected 2 black queens, 0 white - one might be white queen"
                })
    
    return fixes_made, questionable_squares


def find_board_from_kings(img_bgr, model, debug=False):
    """
    Fallback board detection: scan image for kings and estimate board location.
    
    Kings are always present and have distinctive shapes. This function:
    1. Scans the image center area with a sliding window
    2. Uses the model to detect king probabilities
    3. If kings found, estimates the board quadrilateral
    
    Args:
        img_bgr: Input image in BGR format
        model: PyTorch model for classification
        debug: Include debug info
    
    Returns:
        quad: 4 corner points of estimated board (or None if not found)
        debug_info: dict with search details if debug=True
    """
    h, w = img_bgr.shape[:2]
    debug_info = {} if debug else None
    
    # Estimate tile size based on image - assume board is 50-90% of image
    min_tile = min(h, w) // 16  # Board = 8 tiles, so min 50% -> 8 tiles
    max_tile = min(h, w) // 8   # Board = 8 tiles, so max 100% -> 8 tiles
    
    # Try a few tile sizes
    tile_sizes = [
        min(h, w) // 10,  # ~80% of image is board
        min(h, w) // 12,  # ~67% of image is board  
        min(h, w) // 14,  # ~57% of image is board
    ]
    
    best_king_score = 0
    best_king_pos = None
    best_tile_size = None
    best_king_type = None
    
    # Scan center region of image (kings are usually somewhere on the board)
    # Focus on center 80% of image
    margin_y = int(h * 0.1)
    margin_x = int(w * 0.1)
    
    king_candidates = []
    
    for tile_size in tile_sizes:
        if tile_size < 20:
            continue
            
        # Slide window across image
        step = tile_size // 2  # 50% overlap
        
        for y in range(margin_y, h - margin_y - tile_size, step):
            for x in range(margin_x, w - margin_x - tile_size, step):
                tile = img_bgr[y:y+tile_size, x:x+tile_size]
                
                # Resize to model input size
                tile_resized = cv2.resize(tile, (TILE_SIZE, TILE_SIZE))
                tile_rgb = cv2.cvtColor(tile_resized, cv2.COLOR_BGR2RGB)
                
                tensor = torch.from_numpy(tile_rgb).permute(2, 0, 1).float() / 255.0
                tensor = tensor.unsqueeze(0)
                
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                
                wK_prob = probs[CLASSES.index('wK')].item()
                bK_prob = probs[CLASSES.index('bK')].item()
                
                # Check if this looks like a king
                if wK_prob > 0.5:
                    king_candidates.append((wK_prob, x, y, tile_size, 'wK'))
                if bK_prob > 0.5:
                    king_candidates.append((bK_prob, x, y, tile_size, 'bK'))
    
    if not king_candidates:
        if debug:
            debug_info["message"] = "No kings found in image scan"
        return None, debug_info
    
    # Sort by probability
    king_candidates.sort(key=lambda x: -x[0])
    
    if debug:
        debug_info["king_candidates"] = len(king_candidates)
        debug_info["best_candidates"] = [
            {"prob": p, "x": x, "y": y, "tile": t, "type": k}
            for p, x, y, t, k in king_candidates[:5]
        ]
    
    # Use best king to estimate board
    prob, kx, ky, tile_size, king_type = king_candidates[0]
    
    # King center
    king_cx = kx + tile_size // 2
    king_cy = ky + tile_size // 2
    
    # Estimate which square the king is on (assuming standard initial position as hint)
    # White king usually on e1 or nearby, black king on e8 or nearby
    # But in practice, king could be anywhere - we'll estimate board covers most of image
    
    # Assume board is centered roughly around image center
    # And tile_size tells us the scale
    board_size = tile_size * 8
    
    # Estimate board center - use image center weighted toward king position
    img_cx, img_cy = w // 2, h // 2
    board_cx = (img_cx + king_cx) // 2  # Average of image center and king
    board_cy = (img_cy + king_cy) // 2
    
    # Calculate board corners
    half_board = board_size // 2
    x1 = max(0, board_cx - half_board)
    y1 = max(0, board_cy - half_board)
    x2 = min(w, board_cx + half_board)
    y2 = min(h, board_cy + half_board)
    
    # Adjust to maintain square aspect
    actual_w = x2 - x1
    actual_h = y2 - y1
    if actual_w != actual_h:
        size = min(actual_w, actual_h)
        x2 = x1 + size
        y2 = y1 + size
    
    # Create quad (clockwise from top-left)
    quad = np.array([
        [x1, y1],  # top-left
        [x2, y1],  # top-right
        [x2, y2],  # bottom-right
        [x1, y2],  # bottom-left
    ], dtype=np.float32)
    
    if debug:
        debug_info["estimated_board"] = {
            "from_king": king_type,
            "king_pos": (kx, ky),
            "tile_size": tile_size,
            "board_corners": quad.tolist()
        }
    
    print(f"[KING FALLBACK] Found {king_type} at ({kx},{ky}) with {prob*100:.1f}% confidence")
    print(f"[KING FALLBACK] Estimated board: {x1},{y1} to {x2},{y2} (tile_size={tile_size})")
    
    return quad, debug_info


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
    """
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data received"}), 400
    
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
