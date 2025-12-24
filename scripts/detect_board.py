import cv2
import numpy as np
import math
from typing import Optional, Tuple, List


# ============================
# Geometry helpers
# ============================

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_quad(img: np.ndarray, quad: np.ndarray, out_size: int = 640) -> np.ndarray:
    """
    Perspective warp a detected quadrilateral to a square.
    """
    rect = order_points(quad)
    dst = np.array(
        [[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (out_size, out_size))


def center_crop(img: np.ndarray, crop_ratio: float = 0.80) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Crop the center region to avoid selecting page borders / text blocks.
    Returns cropped_image, (x0, y0) offset.
    """
    h, w = img.shape[:2]
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    return img[y0:y0 + ch, x0:x0 + cw], (x0, y0)


def offset_quad(quad: np.ndarray, dx: int, dy: int) -> np.ndarray:
    q = quad.copy()
    q[:, 0, 0] += dx
    q[:, 0, 1] += dy
    return q


# ============================
# Grid scoring (old + new)
# ============================

def _smooth_1d(x: np.ndarray, k: int = 17) -> np.ndarray:
    k = max(3, k | 1)
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def _count_peaks(x: np.ndarray, min_dist: int = 18, rel_thresh: float = 0.28):
    """
    Simple 1D peak picking.
    Returns (peaks, normalized_signal)
    """
    x = x.astype(np.float32)
    x = (x - x.min()) / (np.ptp(x) + 1e-6)

    peaks = []
    last = -10**9
    for i in range(1, len(x) - 1):
        if x[i] > rel_thresh and x[i] >= x[i - 1] and x[i] >= x[i + 1]:
            if i - last >= min_dist:
                peaks.append(i)
                last = i
            else:
                if peaks and x[i] > x[peaks[-1]]:
                    peaks[-1] = i
                    last = i
    return peaks, x


def grid_peak_score(warped_bgr: np.ndarray) -> float:
    """
    Peak-based grid score (works well, but text can sometimes fool it).
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    proj_x = _smooth_1d(np.abs(gx).sum(axis=0))
    proj_y = _smooth_1d(np.abs(gy).sum(axis=1))

    # Damp border influence
    for proj in (proj_x, proj_y):
        cut = int(0.03 * len(proj))
        proj[:cut] *= 0.3
        proj[-cut:] *= 0.3

    peaks_x, nx = _count_peaks(proj_x)
    peaks_y, ny = _count_peaks(proj_y)

    target = 9
    cx = math.exp(-abs(len(peaks_x) - target) / 2.0)
    cy = math.exp(-abs(len(peaks_y) - target) / 2.0)

    strength = 0.0
    if peaks_x:
        strength += float(np.sum(nx[peaks_x]))
    if peaks_y:
        strength += float(np.sum(ny[peaks_y]))

    return (cx * cy) * strength


def grid_line_score(warped_bgr: np.ndarray) -> float:
    """
    Morphological line score: rewards long horizontal+vertical lines
    spread across the image (very board-specific; rejects text blocks).
    """
    g = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)

    bw = cv2.adaptiveThreshold(
        g, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )

    h, w = bw.shape[:2]
    kx = max(12, w // 20)
    ky = max(12, h // 20)

    horiz = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
    )
    vert = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky))
    )

    # fraction of pixels that are part of extracted long lines
    line_ink = (horiz > 0).mean() + (vert > 0).mean()

    # Encourage lines in the *interior*, not just a page border
    border = int(0.06 * min(h, w))
    if border * 2 < min(h, w):
        inner = bw[border:-border, border:-border]
        inner_ink = (inner > 0).mean()
    else:
        inner_ink = (bw > 0).mean()

    return float(line_ink * (0.5 + inner_ink))


def combined_grid_score(warped_bgr: np.ndarray) -> float:
    """
    Blend the two scores. Line score is usually the stronger discriminator.
    """
    return 0.35 * grid_peak_score(warped_bgr) + 0.65 * grid_line_score(warped_bgr)


def validate_board_squareness(warped_bgr: np.ndarray) -> float:
    """
    Validate that the warped board is actually square by checking the grid pattern.
    Returns a score 0-1, where higher means more square-like.
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Warped image should already be square (from warp_quad)
    # So we just need to check if the content looks like a proper chess board
    
    # Use the combined_grid_score as the main indicator
    # If it has a good grid, it's square
    grid = combined_grid_score(warped_bgr)
    
    # Simple validation: if grid score is reasonable, it's likely square
    # The grid score already validates the pattern quality
    return min(1.0, grid * 2.0)  # Scale grid score (0-0.5 typical) to 0-1


# ============================
# Candidate generation
# ============================

def _candidate_quads_from_color_threshold(img_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Find board by thresholding to detect non-white regions.
    Works well when board is surrounded by white background.
    Handles both colored and grayscale boards.
    """
    h, w = img_bgr.shape[:2]
    img_area = float(h * w)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Calculate adaptive threshold based on image brightness
    # For boards with gray/white squares, we need appropriate threshold
    mean_brightness = gray.mean()
    
    # For very bright images (lots of white background), use moderate threshold
    # We want to capture the board (even if light gray) vs pure white background
    if mean_brightness > 200:
        # Very bright image - use fixed threshold to catch board vs white
        threshold_val = 180
    elif mean_brightness > 180:
        # Bright image - slightly lower threshold
        threshold_val = 175
    else:
        # Normal or dark image
        threshold_val = 190
    
    _, thresh = cv2.threshold(gray, int(threshold_val), 255, cv2.THRESH_BINARY_INV)
    
    # Clean up noise with morphological operations
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours of dark regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    quads = []
    for c in contours:
        area = cv2.contourArea(c)
        # Board should be significant portion of image
        # Color-based can use lower threshold since it's more precise
        if area < 0.10 * img_area or area > 0.98 * img_area:
            continue
        
        rect = cv2.minAreaRect(c)
        (_, _), (bw, bh), _ = rect
        if bw < 100 or bh < 100:
            continue
        
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        # Must be very nearly square (tighter than edge-based)
        if aspect > 1.15:
            continue
        
        box = cv2.boxPoints(rect).astype(np.int32).reshape(-1, 1, 2)
        quads.append((area, box, aspect))
    
    return quads

def _candidate_quads_from_edges(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    h, w = gray.shape[:2]
    img_area = float(h * w)
    
    all_quads = []
    
    # Try multiple Canny threshold combinations to find different edge features
    canny_params = [(20, 100), (30, 120), (40, 140)]
    
    for low_thresh, high_thresh in canny_params:
        edges = cv2.Canny(gray, low_thresh, high_thresh)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1
        )

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            # Adjust area range to find boards better
            if area < 0.002 * img_area or area > 0.99 * img_area:
                continue

            rect = cv2.minAreaRect(c)
            (_, _), (bw, bh), _ = rect
            # Lower minimum to catch boards in small images
            if bw < 100 or bh < 100:
                continue

            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            # Chess boards must be nearly square - very strict (1.25 instead of 1.4)
            if aspect > 1.25:
                continue

            box = cv2.boxPoints(rect).astype(np.int32).reshape(-1, 1, 2)
            all_quads.append((area, box, aspect))
    
    # Add color-based candidates (for boards surrounded by white)
    color_quads = _candidate_quads_from_color_threshold(img_bgr)
    all_quads.extend(color_quads)

    # Sort by aspect ratio first (prefer square), then by area
    all_quads.sort(key=lambda t: (abs(t[2] - 1.0), -t[0]))

    # Deduplicate similar rectangles
    out = []
    seen = set()
    for _, q, _ in all_quads:
        x, y, bw, bh = cv2.boundingRect(q)
        key = (int(x / 30), int(y / 30), int(bw / 30), int(bh / 30))
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= 30:
            break

    return out


# ============================
# Public API
# ============================

def _detect_board_quad_grid_aware_impl(
    img_bgr: np.ndarray,
    score_warp_size: int = 256,  # Changed from 512 - peak detection is tuned for 256
    min_relative_area: float = 0.20,  # Lowered from 0.35 to handle grayish backgrounds
    debug: bool = False,
) -> Optional[np.ndarray]:
    candidates = _candidate_quads_from_edges(img_bgr)
    
    # Calculate image dimensions for relative size checking
    img_h, img_w = img_bgr.shape[:2]
    img_area = img_h * img_w
    min_abs_area = 250 * 250  # Minimum 250x250 pixels in absolute terms

    if debug:
        print(f"Image size: {img_w}x{img_h}, area: {img_area}")
        print(f"Found {len(candidates)} candidates")

    best_quad = None
    best_score = -1.0
    best_index = -1

    for i, quad in enumerate(candidates):
        # Calculate quad area relative to image
        quad_area = cv2.contourArea(quad)
        relative_area = quad_area / img_area
        
        if debug:
            print(f"  Cand {i}: area={quad_area:.0f} ({relative_area*100:.1f}%)", end="")
        
        # Board should be at least 35% of the cropped image area
        # Also enforce absolute minimum size
        if relative_area < min_relative_area or quad_area < min_abs_area:
            if debug:
                print(" - SKIPPED (too small)")
            continue
        
        warped = warp_quad(img_bgr, quad, out_size=score_warp_size)
        grid_score = combined_grid_score(warped)
        
        # Reject candidates with very low grid scores (likely not a chess board)
        # Real chess boards should have grid_score > 0.15 at minimum
        if grid_score < 0.15:
            if debug:
                print(f" - grid={grid_score:.3f} - SKIPPED (low grid score)")
            continue
        
        # Cap unrealistic grid scores (numerical artifacts)
        grid_score = min(grid_score, 1.0)
        
        # Get aspect ratio of the detected quad in original image
        rect = cv2.minAreaRect(quad)
        (_, _), (bw, bh), _ = rect
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        
        # Balance grid quality, size, and shape
        # Chess boards should be nearly square (aspect close to 1.0)
        
        # Strong grid score + high area + square shape = excellent detection
        if relative_area > 0.85 and aspect < 1.15 and grid_score > 0.50:
            # Board fills image perfectly - highly favor this
            adjusted_score = grid_score + 0.3
        elif relative_area > 0.80:
            # Large detection: penalize unless grid is excellent
            # This prevents wrapping too much extra content
            if grid_score > 0.70:
                # Excellent grid score overrides size concern
                adjusted_score = grid_score + 0.1
            else:
                size_penalty = (relative_area - 0.80) * 2.0
                adjusted_score = grid_score - size_penalty
        elif relative_area < 0.45:
            # Small detections: need excellent grid score
            size_penalty = (0.45 - relative_area) * 2.0
            adjusted_score = grid_score - size_penalty
        else:
            # Good size range: moderate size bonus
            size_bonus = (relative_area - 0.35) * 0.5
            adjusted_score = grid_score + size_bonus
        
        if debug:
            penalty_or_bonus = adjusted_score - grid_score
            print(f" - grid={grid_score:.3f}, aspect={aspect:.2f}, adj={penalty_or_bonus:+.3f}, final={adjusted_score:.3f}")

        if adjusted_score > best_score:
            best_score = adjusted_score
            best_quad = quad
            best_index = i
            if debug:
                print(f"    ^ NEW BEST")

    if debug:
        print(f"Best: candidate {best_index}, score={best_score:.3f}")
    
    return best_quad


def _crop_region(img_bgr: np.ndarray, crop_ratio: float, vertical_position: str = "center") -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Crop image with specified ratio and vertical position.
    vertical_position: "center", "top", "bottom"
    Returns (cropped_image, (x_offset, y_offset))
    """
    h, w = img_bgr.shape[:2]
    new_h = int(h * crop_ratio)
    new_w = int(w * crop_ratio)
    
    x0 = (w - new_w) // 2  # Always center horizontally
    
    if vertical_position == "top":
        y0 = 0
    elif vertical_position == "bottom":
        y0 = h - new_h
    else:  # center
        y0 = (h - new_h) // 2
    
    return img_bgr[y0:y0+new_h, x0:x0+new_w], (x0, y0)


def detect_board_quad_grid_aware(
    img_bgr: np.ndarray,
    crop_ratio: float = None,
    score_warp_size: int = 256,  # Changed from 512 - peak detection is tuned for 256
    debug: bool = False,
) -> Optional[np.ndarray]:
    """
    Detect chessboard quad using:
      1) center-crop (reduce text/artefacts)
      2) grid-aware scoring (peaks + morphological lines)

    Returns quad in *original image coordinates* or None.
    
    crop_ratio: Auto-detected by default. Try multiple crops.
    """
    # If crop_ratio not specified, try adaptive approach
    if crop_ratio is None:
        # Try multiple crop configurations
        # For portrait photos, the board may be at top or center
        crop_configs = [
            (0.95, "center"),  # Tight center crop
            (0.80, "center"),  # Loose center crop
            (0.70, "top"),     # Top crop for boards at top of portrait photos
            (0.70, "center"),  # Loose center
        ]
        
        best_quad = None
        best_score = -1.0
        best_offset = (0, 0)
        best_config = None
        
        for crop_ratio_try, vert_pos in crop_configs:
            crop, (x0, y0) = _crop_region(img_bgr, crop_ratio_try, vert_pos)
            quad = _detect_board_quad_grid_aware_impl(crop, score_warp_size=score_warp_size, debug=debug)
            
            if quad is None:
                continue
            
            quad_area = cv2.contourArea(quad)
            crop_area = crop.shape[0] * crop.shape[1]
            relative_area = quad_area / crop_area
            
            warped = warp_quad(img_bgr, offset_quad(quad, x0, y0), out_size=score_warp_size)
            grid_score = combined_grid_score(warped)
            
            # Skip if grid score is too low or cap if too high
            if grid_score < 0.15:
                if debug:
                    print(f"Crop({crop_ratio_try:.2f}, {vert_pos}): grid={grid_score:.3f} - SKIPPED (low grid score)")
                continue
            
            grid_score = min(grid_score, 1.0)  # Cap unrealistic scores
            
            # Calculate final score with area adjustment
            if relative_area > 0.75:
                area_adj = 0
            else:
                area_adj = (0.75 - relative_area) * 0.3
            
            score = grid_score + area_adj
            
            if debug:
                print(f"Crop({crop_ratio_try:.2f}, {vert_pos}): grid={grid_score:.3f}, area={relative_area:.2%}, final={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_quad = quad
                best_offset = (x0, y0)
                best_config = (crop_ratio_try, vert_pos)
                if debug:
                    print(f"    ^ NEW BEST")
        
        if best_quad is None:
            return None
        
        if debug:
            print(f"Selected: {best_config}")
        
        return offset_quad(best_quad, best_offset[0], best_offset[1])
    
    # Fixed crop_ratio specified
    crop, (x0, y0) = center_crop(img_bgr, crop_ratio=crop_ratio)
    quad_crop = _detect_board_quad_grid_aware_impl(crop, score_warp_size=score_warp_size, debug=debug)
    if quad_crop is None:
        return None
    return offset_quad(quad_crop, x0, y0)


def detect_and_warp_board(
    img_bgr: np.ndarray,
    out_size: int = 640,
    crop_ratio: float = 0.80,
) -> Optional[np.ndarray]:
    quad = detect_board_quad_grid_aware(img_bgr, crop_ratio=crop_ratio)
    if quad is None:
        return None
    return warp_quad(img_bgr, quad, out_size=out_size)


# ============================
# CLI debug
# ============================

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python detect_board.py <image_path>")
        raise SystemExit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        raise FileNotFoundError(sys.argv[1])

    quad = detect_board_quad_grid_aware(img, crop_ratio=0.80)
    if quad is None:
        print("Board not found")
        raise SystemExit(2)

    debug = img.copy()
    cv2.drawContours(debug, [quad], -1, (0, 255, 0), 4)

    warped = warp_quad(img, quad, out_size=640)

    cv2.imshow("Detected board", debug)
    cv2.imshow("Warped board", warped)
    print("✅ Board detected — press any key to close windows")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
