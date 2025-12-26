"""
Piece detection for grid alignment validation and correction.

When automatic board detection produces misaligned grids, this module
can locate chess pieces in the warped image and use their center positions
to calculate the correct grid offset.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

# Piece detection uses contrast and shape analysis, not the trained model
# This makes it independent of model accuracy


def detect_piece_blobs(
    warped_bgr: np.ndarray,
    min_area_ratio: float = 0.002,
    max_area_ratio: float = 0.02,
) -> List[Tuple[float, float, float]]:
    """
    Detect blobs that are likely chess pieces using contrast analysis.
    
    Returns list of (center_x, center_y, area) for detected pieces.
    Pieces should appear as high-contrast regions against the square background.
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    total_area = h * w
    
    # Apply adaptive threshold to find dark/light regions
    # This will catch both dark pieces on light squares and light pieces on dark squares
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    pieces = []
    
    # Find dark blobs (dark pieces or shadows)
    _, thresh_dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find light blobs (light pieces)
    _, thresh_light = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    for thresh in [thresh_dark, thresh_light]:
        # Clean up with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            area = cv2.contourArea(c)
            area_ratio = area / total_area
            
            # Filter by size - pieces should be a reasonable portion of a square
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            
            # Get bounding rect and check aspect ratio (pieces are roughly circular/tall)
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            if aspect > 2.5:  # Too elongated to be a piece
                continue
            
            # Calculate center using moments for more accuracy
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                pieces.append((cx, cy, area))
    
    # Deduplicate overlapping detections
    deduped = []
    for p in pieces:
        is_dup = False
        for existing in deduped:
            dist = np.sqrt((p[0] - existing[0])**2 + (p[1] - existing[1])**2)
            if dist < w / 16:  # Within half a square
                is_dup = True
                break
        if not is_dup:
            deduped.append(p)
    
    return deduped


def find_strongest_piece(
    warped_bgr: np.ndarray,
    margin_ratio: float = 0.1,
) -> Optional[Tuple[float, float, float]]:
    """
    Find the single most prominent piece in the warped image.
    
    Uses edge density and contrast to identify the most "piece-like" region.
    Avoids the margins where grid lines might interfere.
    
    Returns (center_x, center_y, confidence) or None.
    """
    h, w = warped_bgr.shape[:2]
    sq_size = w // 8
    
    # Exclude margin areas
    margin = int(sq_size * margin_ratio)
    
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    
    # Calculate local contrast using Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    contrast = np.abs(laplacian)
    
    # Find edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Analyze each square
    best_square = None
    best_score = 0
    
    for row in range(8):
        for col in range(8):
            y1 = row * sq_size + margin
            y2 = (row + 1) * sq_size - margin
            x1 = col * sq_size + margin
            x2 = (col + 1) * sq_size - margin
            
            if y2 <= y1 or x2 <= x1:
                continue
            
            # Get region
            region_contrast = contrast[y1:y2, x1:x2]
            region_edges = edges[y1:y2, x1:x2]
            region_gray = gray[y1:y2, x1:x2]
            
            # Calculate scores
            edge_density = np.mean(region_edges) / 255.0
            contrast_score = np.std(region_gray)  # Higher std = more contrast = likely piece
            
            # Combined score
            score = edge_density * 0.4 + (contrast_score / 128.0) * 0.6
            
            if score > best_score:
                best_score = score
                best_square = (row, col, score)
    
    if best_square is None or best_score < 0.15:
        return None
    
    row, col, _ = best_square
    
    # Now find the precise center of the piece within this square
    y1 = row * sq_size
    y2 = (row + 1) * sq_size
    x1 = col * sq_size
    x2 = (col + 1) * sq_size
    
    square_img = gray[y1:y2, x1:x2]
    
    # Use adaptive threshold to find the piece
    thresh = cv2.adaptiveThreshold(
        square_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 5
    )
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback: return center of square
        return (x1 + sq_size/2, y1 + sq_size/2, best_score)
    
    # Find the largest contour (likely the piece)
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    
    if M["m00"] > 0:
        cx = x1 + M["m10"] / M["m00"]
        cy = y1 + M["m01"] / M["m00"]
    else:
        cx = x1 + sq_size / 2
        cy = y1 + sq_size / 2
    
    return (cx, cy, best_score)


def calculate_grid_offset(
    piece_center: Tuple[float, float],
    board_size: int,
) -> Tuple[float, float]:
    """
    Calculate how far off the grid is based on a piece's center position.
    
    A perfectly centered piece should be at (n + 0.5) * sq_size for some integer n.
    This returns the offset needed to align the grid.
    
    Returns (dx, dy) offset in pixels.
    """
    sq_size = board_size / 8
    cx, cy = piece_center
    
    # Find which square this piece is in
    expected_col = int(cx / sq_size)
    expected_row = int(cy / sq_size)
    
    # Expected center of that square
    expected_cx = (expected_col + 0.5) * sq_size
    expected_cy = (expected_row + 0.5) * sq_size
    
    # Offset is how far the actual center is from expected
    dx = cx - expected_cx
    dy = cy - expected_cy
    
    return (dx, dy)


def suggest_grid_adjustment(
    warped_bgr: np.ndarray,
) -> Dict:
    """
    Analyze the warped board and suggest grid adjustments.
    
    Returns a dict with:
    - piece_found: bool
    - piece_center: (x, y) if found
    - piece_square: (row, col) detected square
    - offset: (dx, dy) suggested offset
    - confidence: float 0-1
    """
    h, w = warped_bgr.shape[:2]
    
    result = {
        "piece_found": False,
        "piece_center": None,
        "piece_square": None,
        "offset": (0, 0),
        "confidence": 0.0,
    }
    
    piece = find_strongest_piece(warped_bgr)
    
    if piece is None:
        return result
    
    cx, cy, confidence = piece
    sq_size = w / 8
    
    # Determine which square
    col = int(cx / sq_size)
    row = int(cy / sq_size)
    
    # Clamp to valid range
    col = max(0, min(7, col))
    row = max(0, min(7, row))
    
    # Calculate offset
    dx, dy = calculate_grid_offset((cx, cy), w)
    
    result["piece_found"] = True
    result["piece_center"] = (cx, cy)
    result["piece_square"] = (row, col)
    result["offset"] = (dx, dy)
    result["confidence"] = confidence
    
    return result


def draw_piece_detection_overlay(
    warped_bgr: np.ndarray,
    detection_result: Dict,
    sq_size: int = 80,
) -> np.ndarray:
    """
    Draw the piece detection result on the warped board.
    Shows:
    - 8x8 grid
    - Detected piece center (red dot)
    - Expected center (green dot)
    - Offset arrow
    """
    # Resize to standard size
    target_size = sq_size * 8
    img = cv2.resize(warped_bgr.copy(), (target_size, target_size))
    
    # Draw grid
    for i in range(9):
        p = i * sq_size
        cv2.line(img, (0, p), (target_size, p), (0, 255, 0), 1)
        cv2.line(img, (p, 0), (p, target_size), (0, 255, 0), 1)
    
    if not detection_result.get("piece_found"):
        cv2.putText(img, "No piece detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img
    
    # Scale coordinates to new size
    orig_size = warped_bgr.shape[0]
    scale = target_size / orig_size
    
    cx, cy = detection_result["piece_center"]
    cx = int(cx * scale)
    cy = int(cy * scale)
    
    row, col = detection_result["piece_square"]
    expected_cx = int((col + 0.5) * sq_size)
    expected_cy = int((row + 0.5) * sq_size)
    
    # Draw expected center (green)
    cv2.circle(img, (expected_cx, expected_cy), 8, (0, 255, 0), 2)
    
    # Draw actual center (red)
    cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)
    
    # Draw offset arrow
    cv2.arrowedLine(img, (expected_cx, expected_cy), (cx, cy), (255, 0, 255), 2)
    
    # Draw info
    dx, dy = detection_result["offset"]
    dx_scaled = dx * scale
    dy_scaled = dy * scale
    conf = detection_result["confidence"]
    
    text = f"Offset: ({dx_scaled:.1f}, {dy_scaled:.1f}) conf={conf:.2f}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Square label
    files = "abcdefgh"
    square_name = f"{files[col]}{8-row}"
    cv2.putText(img, f"Piece in: {square_name}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return img


# CLI for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python piece_detect.py <warped_board.png> [output_overlay.png]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Failed to load: {input_path}")
        sys.exit(1)
    
    print(f"Analyzing: {input_path}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    
    result = suggest_grid_adjustment(img)
    
    print()
    if result["piece_found"]:
        cx, cy = result["piece_center"]
        row, col = result["piece_square"]
        dx, dy = result["offset"]
        files = "abcdefgh"
        square_name = f"{files[col]}{8-row}"
        
        print(f"Piece detected in square: {square_name}")
        print(f"  Center position: ({cx:.1f}, {cy:.1f})")
        print(f"  Grid offset: dx={dx:.1f}px, dy={dy:.1f}px")
        print(f"  Confidence: {result['confidence']:.2f}")
        
        if abs(dx) > 5 or abs(dy) > 5:
            print()
            print(f"⚠ Grid may be misaligned! Suggest adjusting by ({-dx:.0f}, {-dy:.0f}) pixels")
    else:
        print("No piece detected - board may be empty or detection failed")
    
    if output_path:
        overlay = draw_piece_detection_overlay(img, result)
        cv2.imwrite(output_path, overlay)
        print(f"\nOverlay saved to: {output_path}")
