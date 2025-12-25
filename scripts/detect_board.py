import cv2
import numpy as np
import math
from typing import Optional, Tuple, List
from scipy import ndimage


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


def warp_quad(img: np.ndarray, quad: np.ndarray, out_size: int = 640, expand_ratio: float = 0.0, shrink_ratio: float = 0.0) -> np.ndarray:
    """
    Perspective warp a detected quadrilateral to a square.
    
    expand_ratio: If > 0, expand the quad outward by this ratio (e.g., 0.02 = 2% expansion)
                  Useful when detection cuts too tight.
    shrink_ratio: If > 0, shrink the quad inward by this ratio (e.g., 0.03 = 3% shrink)
                  Useful for digital chess screenshots where coordinate labels are inside the detection.
    """
    rect = order_points(quad)
    
    # Apply shrink first (for digital boards with coordinates inside)
    if shrink_ratio > 0:
        center = rect.mean(axis=0)
        rect = center + (rect - center) * (1 - shrink_ratio)
    
    # Then apply expand (for physical boards where detection is too tight)
    if expand_ratio > 0:
        center = rect.mean(axis=0)
        rect = center + (rect - center) * (1 + expand_ratio)
    
    dst = np.array(
        [[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (out_size, out_size))


def detect_edge_points(img_gray: np.ndarray, pt1: np.ndarray, pt2: np.ndarray, 
                       num_samples: int = 9, search_range: int = 30) -> np.ndarray:
    """
    Detect actual edge points along a line between two corners.
    This helps find curved board edges caused by lens distortion.
    
    Returns array of shape (num_samples, 2) with refined edge points.
    """
    # Generate sample points along the line
    t_values = np.linspace(0, 1, num_samples)
    sample_points = np.array([pt1 + t * (pt2 - pt1) for t in t_values])
    
    # Compute edge direction (perpendicular to the line)
    line_vec = pt2 - pt1
    line_len = np.linalg.norm(line_vec)
    if line_len < 1:
        return sample_points
    
    perp_vec = np.array([-line_vec[1], line_vec[0]]) / line_len
    
    # Compute gradient magnitude
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    
    h, w = img_gray.shape
    refined_points = []
    
    for pt in sample_points:
        best_offset = 0
        best_grad = 0
        
        # Search along perpendicular direction
        for offset in range(-search_range, search_range + 1):
            test_pt = pt + offset * perp_vec
            x, y = int(test_pt[0]), int(test_pt[1])
            
            if 0 <= x < w and 0 <= y < h:
                grad = gradient[y, x]
                if grad > best_grad:
                    best_grad = grad
                    best_offset = offset
        
        refined_pt = pt + best_offset * perp_vec
        refined_points.append(refined_pt)
    
    return np.array(refined_points, dtype=np.float32)


def warp_curved_quad(img: np.ndarray, quad: np.ndarray, out_size: int = 640,
                     grid_size: int = 9) -> np.ndarray:
    """
    Warp a quadrilateral with potentially curved edges to a square.
    Uses grid-based warping to handle lens distortion.
    
    grid_size: Number of points to sample along each edge (default 9 for 8 segments)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    rect = order_points(quad)
    
    # TL, TR, BR, BL
    tl, tr, br, bl = rect
    
    # Detect actual edge points (may be curved)
    top_edge = detect_edge_points(gray, tl, tr, grid_size, search_range=20)
    right_edge = detect_edge_points(gray, tr, br, grid_size, search_range=20)
    bottom_edge = detect_edge_points(gray, br, bl, grid_size, search_range=20)
    left_edge = detect_edge_points(gray, bl, tl, grid_size, search_range=20)
    
    # Build source grid (detected curved points)
    src_points = []
    for i in range(grid_size):
        row_points = []
        for j in range(grid_size):
            # Bilinear interpolation between edges
            t_x = j / (grid_size - 1)
            t_y = i / (grid_size - 1)
            
            # Interpolate along top and bottom edges
            top_pt = top_edge[j]
            bottom_pt = bottom_edge[grid_size - 1 - j]  # Reversed direction
            
            # Interpolate along left and right edges
            left_pt = left_edge[grid_size - 1 - i]  # Reversed direction
            right_pt = right_edge[i]
            
            # Bilinear blend
            pt_tb = (1 - t_y) * top_pt + t_y * bottom_pt
            pt_lr = (1 - t_x) * left_pt + t_x * right_pt
            
            # Average (transfinite interpolation simplified)
            pt = (pt_tb + pt_lr) / 2
            
            # Correct for double-counting corners
            corner_correction = (
                (1 - t_x) * (1 - t_y) * tl +
                t_x * (1 - t_y) * tr +
                t_x * t_y * br +
                (1 - t_x) * t_y * bl
            )
            pt = pt_tb + pt_lr - corner_correction
            
            row_points.append(pt)
        src_points.append(row_points)
    
    src_points = np.array(src_points, dtype=np.float32)
    
    # Build destination grid (perfect square)
    cell_size = out_size / (grid_size - 1)
    dst_points = np.zeros((grid_size, grid_size, 2), dtype=np.float32)
    for i in range(grid_size):
        for j in range(grid_size):
            dst_points[i, j] = [j * cell_size, i * cell_size]
    
    # Warp each cell using perspective transform
    result = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            # Source quad for this cell
            src_quad = np.array([
                src_points[i, j],
                src_points[i, j + 1],
                src_points[i + 1, j + 1],
                src_points[i + 1, j],
            ], dtype=np.float32)
            
            # Destination quad for this cell
            dst_quad = np.array([
                dst_points[i, j],
                dst_points[i, j + 1],
                dst_points[i + 1, j + 1],
                dst_points[i + 1, j],
            ], dtype=np.float32)
            
            # Get perspective transform
            M = cv2.getPerspectiveTransform(src_quad, dst_quad)
            
            # Warp the entire image
            cell_warped = cv2.warpPerspective(img, M, (out_size, out_size))
            
            # Copy only this cell's region
            y1 = int(dst_points[i, j, 1])
            y2 = int(dst_points[i + 1, j, 1])
            x1 = int(dst_points[i, j, 0])
            x2 = int(dst_points[i, j + 1, 0])
            
            result[y1:y2, x1:x2] = cell_warped[y1:y2, x1:x2]
    
    return result


def visualize_curved_edges(img: np.ndarray, quad: np.ndarray, grid_size: int = 9) -> np.ndarray:
    """
    Visualize detected curved edge points on the original image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    vis = img.copy()
    rect = order_points(quad)
    
    tl, tr, br, bl = rect
    
    # Detect edge points
    top_edge = detect_edge_points(gray, tl, tr, grid_size, search_range=20)
    right_edge = detect_edge_points(gray, tr, br, grid_size, search_range=20)
    bottom_edge = detect_edge_points(gray, br, bl, grid_size, search_range=20)
    left_edge = detect_edge_points(gray, bl, tl, grid_size, search_range=20)
    
    # Draw straight lines (what simple warp would use) in red
    cv2.line(vis, tuple(tl.astype(int)), tuple(tr.astype(int)), (0, 0, 255), 2)
    cv2.line(vis, tuple(tr.astype(int)), tuple(br.astype(int)), (0, 0, 255), 2)
    cv2.line(vis, tuple(br.astype(int)), tuple(bl.astype(int)), (0, 0, 255), 2)
    cv2.line(vis, tuple(bl.astype(int)), tuple(tl.astype(int)), (0, 0, 255), 2)
    
    # Draw detected curved edges in green
    for i in range(len(top_edge) - 1):
        cv2.line(vis, tuple(top_edge[i].astype(int)), tuple(top_edge[i+1].astype(int)), (0, 255, 0), 3)
        cv2.line(vis, tuple(right_edge[i].astype(int)), tuple(right_edge[i+1].astype(int)), (0, 255, 0), 3)
        cv2.line(vis, tuple(bottom_edge[i].astype(int)), tuple(bottom_edge[i+1].astype(int)), (0, 255, 0), 3)
        cv2.line(vis, tuple(left_edge[i].astype(int)), tuple(left_edge[i+1].astype(int)), (0, 255, 0), 3)
    
    # Draw detected points
    for pt in np.concatenate([top_edge, right_edge, bottom_edge, left_edge]):
        cv2.circle(vis, tuple(pt.astype(int)), 6, (0, 255, 255), -1)
    
    # Calculate and display curvature
    def measure_edge_curvature(edge_pts):
        """Measure max deviation from straight line."""
        start, end = edge_pts[0], edge_pts[-1]
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        if line_len < 1:
            return 0
        line_unit = line_vec / line_len
        perp = np.array([-line_unit[1], line_unit[0]])
        
        max_dev = 0
        for pt in edge_pts[1:-1]:
            vec = pt - start
            dev = abs(np.dot(vec, perp))
            max_dev = max(max_dev, dev)
        return max_dev
    
    curvatures = [
        measure_edge_curvature(top_edge),
        measure_edge_curvature(right_edge),
        measure_edge_curvature(bottom_edge),
        measure_edge_curvature(left_edge)
    ]
    
    cv2.putText(vis, f"Edge curvature: T={curvatures[0]:.1f} R={curvatures[1]:.1f} B={curvatures[2]:.1f} L={curvatures[3]:.1f}", 
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return vis


def detect_grid_intersections(warped: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect the 7x7 = 49 interior grid intersection points on a warped chess board.
    Uses line detection to find grid lines, then computes intersections.
    Returns array of shape (7, 7, 2) with (x, y) coordinates, or None if detection fails.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
    h, w = gray.shape
    cell_size = w // 8
    
    # Use edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                            minLineLength=cell_size*2, maxLineGap=cell_size//2)
    
    if lines is None or len(lines) < 14:  # Need at least 7 horizontal + 7 vertical
        return None
    
    # Classify lines as horizontal or vertical
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        if angle < 20 or angle > 160:  # Horizontal
            y_avg = (y1 + y2) / 2
            horizontal_lines.append(y_avg)
        elif 70 < angle < 110:  # Vertical
            x_avg = (x1 + x2) / 2
            vertical_lines.append(x_avg)
    
    if len(horizontal_lines) < 7 or len(vertical_lines) < 7:
        return None
    
    # Cluster lines to find the 7 grid lines (not 9 - we want interior lines only)
    def cluster_lines(positions, expected_count, cell_size):
        """Cluster nearby lines and return the centers."""
        positions = sorted(positions)
        clusters = []
        current_cluster = [positions[0]]
        
        for pos in positions[1:]:
            if pos - current_cluster[-1] < cell_size * 0.4:
                current_cluster.append(pos)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [pos]
        clusters.append(np.mean(current_cluster))
        
        # Filter to get lines near expected positions (interior lines at 1/8, 2/8, ..., 7/8)
        expected_positions = [(i + 1) * cell_size for i in range(7)]
        matched = []
        for exp in expected_positions:
            nearest = min(clusters, key=lambda x: abs(x - exp), default=exp)
            if abs(nearest - exp) < cell_size * 0.3:
                matched.append(nearest)
            else:
                matched.append(exp)  # Fall back to expected
        
        return matched
    
    h_lines = cluster_lines(horizontal_lines, 7, cell_size)
    v_lines = cluster_lines(vertical_lines, 7, cell_size)
    
    if len(h_lines) < 7 or len(v_lines) < 7:
        return None
    
    # Build intersection grid
    detected_points = np.zeros((7, 7, 2), dtype=np.float32)
    for row in range(7):
        for col in range(7):
            detected_points[row, col] = [v_lines[col], h_lines[row]]
    
    return detected_points


def measure_grid_curvature(grid_points: np.ndarray) -> Tuple[float, float]:
    """
    Measure the curvature of detected grid lines.
    Returns (horizontal_curvature, vertical_curvature) where:
    - 0 = perfectly straight
    - positive = barrel distortion (bowing outward)
    - negative = pincushion distortion (bowing inward)
    """
    # Check horizontal lines (each row should be straight)
    h_deviations = []
    for row in range(7):
        points = grid_points[row, :, :]  # 7 points in this row
        # Fit a line and measure deviation
        xs = points[:, 0]
        ys = points[:, 1]
        # Linear fit
        coeffs = np.polyfit(xs, ys, 1)
        expected_ys = np.polyval(coeffs, xs)
        deviation = ys - expected_ys
        # Curvature is the deviation at the middle
        h_deviations.append(deviation[3])  # Middle point
    
    # Check vertical lines (each column should be straight)
    v_deviations = []
    for col in range(7):
        points = grid_points[:, col, :]  # 7 points in this column
        xs = points[:, 0]
        ys = points[:, 1]
        coeffs = np.polyfit(ys, xs, 1)
        expected_xs = np.polyval(coeffs, ys)
        deviation = xs - expected_xs
        v_deviations.append(deviation[3])  # Middle point
    
    return np.mean(h_deviations), np.mean(v_deviations)


def correct_grid_curvature(warped: np.ndarray, grid_points: np.ndarray) -> np.ndarray:
    """
    Correct lens distortion by remapping the image based on detected grid points.
    Uses piecewise affine warping for each cell to straighten the grid.
    """
    h, w = warped.shape[:2]
    cell_size = w // 8
    
    # Create ideal grid points (where they should be)
    ideal_points = np.zeros((9, 9, 2), dtype=np.float32)
    for row in range(9):
        for col in range(9):
            ideal_points[row, col] = [col * cell_size, row * cell_size]
    
    # Extend detected 7x7 grid to 9x9 by extrapolating edges
    extended_detected = np.zeros((9, 9, 2), dtype=np.float32)
    
    # Copy inner 7x7 to positions [1:8, 1:8]
    extended_detected[1:8, 1:8] = grid_points
    
    # Extrapolate edges
    # Top row (row 0)
    for col in range(1, 8):
        diff = grid_points[0, col-1] - grid_points[1, col-1]
        extended_detected[0, col] = grid_points[0, col-1] + diff
    
    # Bottom row (row 8)
    for col in range(1, 8):
        diff = grid_points[6, col-1] - grid_points[5, col-1]
        extended_detected[8, col] = grid_points[6, col-1] + diff
    
    # Left column (col 0)
    for row in range(1, 8):
        diff = grid_points[row-1, 0] - grid_points[row-1, 1]
        extended_detected[row, 0] = grid_points[row-1, 0] + diff
    
    # Right column (col 8)
    for row in range(1, 8):
        diff = grid_points[row-1, 6] - grid_points[row-1, 5]
        extended_detected[row, 8] = grid_points[row-1, 6] + diff
    
    # Corners (extrapolate from adjacent)
    extended_detected[0, 0] = extended_detected[0, 1] + (extended_detected[1, 0] - extended_detected[1, 1])
    extended_detected[0, 8] = extended_detected[0, 7] + (extended_detected[1, 8] - extended_detected[1, 7])
    extended_detected[8, 0] = extended_detected[8, 1] + (extended_detected[7, 0] - extended_detected[7, 1])
    extended_detected[8, 8] = extended_detected[8, 7] + (extended_detected[7, 8] - extended_detected[7, 7])
    
    # Create output image
    corrected = np.zeros_like(warped)
    
    # Process each cell with a local perspective transform
    for row in range(8):
        for col in range(8):
            # Source quad (detected/distorted positions)
            src = np.array([
                extended_detected[row, col],
                extended_detected[row, col + 1],
                extended_detected[row + 1, col + 1],
                extended_detected[row + 1, col],
            ], dtype=np.float32)
            
            # Destination quad (ideal positions)
            dst = np.array([
                ideal_points[row, col],
                ideal_points[row, col + 1],
                ideal_points[row + 1, col + 1],
                ideal_points[row + 1, col],
            ], dtype=np.float32)
            
            # Get perspective transform for this cell
            M = cv2.getPerspectiveTransform(src, dst)
            
            # Warp the entire image and extract just this cell
            cell_warped = cv2.warpPerspective(warped, M, (w, h))
            
            # Copy the cell region to output
            y1, y2 = row * cell_size, (row + 1) * cell_size
            x1, x2 = col * cell_size, (col + 1) * cell_size
            corrected[y1:y2, x1:x2] = cell_warped[y1:y2, x1:x2]
    
    return corrected


def dewarp_board(warped: np.ndarray, curvature_threshold: float = 2.0) -> Tuple[np.ndarray, bool]:
    """
    Detect and correct lens distortion on a warped chess board.
    
    Returns:
        (corrected_image, was_corrected) - the dewarped image and whether correction was applied
    """
    # Detect grid intersections
    grid_points = detect_grid_intersections(warped)
    
    if grid_points is None:
        return warped, False
    
    # Measure curvature
    h_curv, v_curv = measure_grid_curvature(grid_points)
    max_curvature = max(abs(h_curv), abs(v_curv))
    
    # Only correct if curvature exceeds threshold (in pixels)
    if max_curvature > curvature_threshold:
        corrected = correct_grid_curvature(warped, grid_points)
        return corrected, True
    
    return warped, False


def visualize_grid_detection(warped: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], float, float]:
    """
    Visualize grid line detection and measured curvature.
    Returns: (visualization_image, grid_points, h_curvature, v_curvature)
    """
    vis = warped.copy()
    h, w = warped.shape[:2]
    cell_size = w // 8
    
    # Draw ideal grid in blue (dashed effect via short lines)
    for i in range(1, 8):
        pos = i * cell_size
        # Horizontal
        cv2.line(vis, (0, pos), (w, pos), (255, 100, 100), 1)
        # Vertical  
        cv2.line(vis, (pos, 0), (pos, h), (255, 100, 100), 1)
    
    # Detect grid
    grid_points = detect_grid_intersections(warped)
    
    if grid_points is None:
        cv2.putText(vis, "Grid detection failed", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return vis, None, 0.0, 0.0
    
    # Measure curvature
    h_curv, v_curv = measure_grid_curvature(grid_points)
    
    # Draw detected grid lines in green
    for row in range(7):
        for col in range(6):
            pt1 = tuple(grid_points[row, col].astype(int))
            pt2 = tuple(grid_points[row, col+1].astype(int))
            cv2.line(vis, pt1, pt2, (0, 255, 0), 2)
    
    for col in range(7):
        for row in range(6):
            pt1 = tuple(grid_points[row, col].astype(int))
            pt2 = tuple(grid_points[row+1, col].astype(int))
            cv2.line(vis, pt1, pt2, (0, 255, 0), 2)
    
    # Draw intersection points
    for row in range(7):
        for col in range(7):
            pt = tuple(grid_points[row, col].astype(int))
            cv2.circle(vis, pt, 5, (0, 255, 255), -1)
    
    # Add curvature info
    cv2.putText(vis, f"H-curve: {h_curv:.1f}px, V-curve: {v_curv:.1f}px", 
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return vis, grid_points, h_curv, v_curv


def get_square_brightness(warped: np.ndarray, row: int, col: int) -> float:
    """
    Get the average brightness of a square on the warped board.
    row 0 = rank 8, row 7 = rank 1
    col 0 = file a, col 7 = file h
    """
    h, w = warped.shape[:2]
    cell_h, cell_w = h // 8, w // 8
    
    # Get center region of square (avoid pieces at edges)
    margin = cell_h // 4
    y1 = row * cell_h + margin
    y2 = (row + 1) * cell_h - margin
    x1 = col * cell_w + margin
    x2 = (col + 1) * cell_w - margin
    
    cell = warped[y1:y2, x1:x2]
    
    # Convert to grayscale if color
    if len(cell.shape) == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    
    return np.mean(cell)


def detect_board_orientation(warped: np.ndarray) -> int:
    """
    Detect the correct orientation of a warped chess board.
    
    Uses the fact that:
    - h1 (row 7, col 7) is ALWAYS a light square
    - a1 (row 7, col 0) is ALWAYS a dark square
    - a8 (row 0, col 0) is ALWAYS a light square
    - h8 (row 0, col 7) is ALWAYS a dark square
    
    Returns:
        Rotation needed: 0, 90, 180, or 270 degrees clockwise
    """
    # Get corner square brightnesses
    # In correct orientation:
    #   a8 (0,0) = light, h8 (0,7) = dark
    #   a1 (7,0) = dark,  h1 (7,7) = light
    
    corners = {
        'top_left': get_square_brightness(warped, 0, 0),      # a8 if correct
        'top_right': get_square_brightness(warped, 0, 7),     # h8 if correct
        'bottom_left': get_square_brightness(warped, 7, 0),   # a1 if correct
        'bottom_right': get_square_brightness(warped, 7, 7),  # h1 if correct
    }
    
    # Also check a few more squares for reliability
    # Light squares (in correct orientation): a8, c8, e8, g8, b7, d7, f7, h7, h1, f1, d1, b1
    # Let's check the 4 corners and use majority voting
    
    # Pattern for correct orientation: TL=light, TR=dark, BL=dark, BR=light
    # After 90° CW rotation: TL=dark, TR=light, BL=light, BR=dark
    # After 180° rotation: TL=dark, TR=light, BL=light, BR=dark  (same pattern, need more squares)
    # After 270° CW rotation: TL=light, TR=dark, BL=dark, BR=light (same as 0°)
    
    # Better approach: check diagonal pattern
    # Sample multiple squares along the board
    
    light_sum = 0  # Sum of brightness for squares that SHOULD be light
    dark_sum = 0   # Sum of brightness for squares that SHOULD be dark
    
    for row in range(8):
        for col in range(8):
            brightness = get_square_brightness(warped, row, col)
            # In correct orientation, (row + col) % 2 == 0 means light square
            if (row + col) % 2 == 0:
                light_sum += brightness
            else:
                dark_sum += brightness
    
    # If light squares are brighter than dark squares, orientation is correct (0 or 180)
    # We need to disambiguate between 0° and 180°
    
    if light_sum > dark_sum:
        # Light/dark pattern is correct, but need to check if board is flipped 180°
        # Check if h1 corner (bottom-right) is lighter than a1 (bottom-left)
        # In correct orientation: BR (h1) is light, BL (a1) is dark
        if corners['bottom_right'] > corners['bottom_left']:
            return 0  # Correct orientation
        else:
            return 180  # Upside down
    else:
        # Board is rotated 90° - light/dark pattern is inverted
        # Check which way it's rotated
        if corners['top_right'] > corners['top_left']:
            return 90  # Rotated 90° clockwise
        else:
            return 270  # Rotated 270° clockwise (or 90° counter-clockwise)


def correct_board_orientation(warped: np.ndarray) -> np.ndarray:
    """
    Detect and correct the orientation of a warped chess board.
    
    Returns the correctly oriented board where:
    - a8 is top-left
    - h8 is top-right
    - a1 is bottom-left
    - h1 is bottom-right (and is a light square)
    """
    rotation = detect_board_orientation(warped)
    
    if rotation == 0:
        return warped
    elif rotation == 90:
        return cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(warped, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    
    return warped


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
# Image Source Detection
# ============================

class ImageSource:
    """Enum-like class for image source types."""
    UNKNOWN = "unknown"
    BOOK_DIAGRAM = "book_diagram"       # Photo of printed book/magazine chess diagram
    DIGITAL_SCREENSHOT = "digital_screenshot"  # Screenshot from chess app/website
    PHYSICAL_BOARD = "physical_board"   # Photo of real 3D chess board
    SCAN = "scan"                       # Scanned document


def detect_image_source(img_bgr: np.ndarray, filename: str = "") -> str:
    """
    Attempt to identify the source type of a chess board image.
    
    Heuristics used:
    - File naming patterns (e.g., "screenshot", "scan")
    - Image characteristics (aspect ratio, color distribution, edge sharpness)
    - Background patterns (white paper vs varied background)
    
    Returns: One of the ImageSource values
    """
    h, w = img_bgr.shape[:2]
    aspect = w / h if h > 0 else 1.0
    
    # Check filename hints
    filename_lower = filename.lower()
    if "screenshot" in filename_lower or "screen" in filename_lower:
        return ImageSource.DIGITAL_SCREENSHOT
    if "scan" in filename_lower:
        return ImageSource.SCAN
    if "book" in filename_lower or "diagram" in filename_lower:
        return ImageSource.BOOK_DIAGRAM
    
    # Analyze image characteristics
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Check for large white/light border regions (suggests book/scan)
    # Sample larger border areas to be more robust
    border_size = min(h, w) // 10
    top_border = gray[:border_size, :].mean()
    bottom_border = gray[-border_size:, :].mean()
    left_border = gray[:, :border_size].mean()
    right_border = gray[:, -border_size:].mean()
    avg_border = (top_border + bottom_border + left_border + right_border) / 4
    
    # Also check corner brightness (book pages have bright corners)
    corner_size = min(h, w) // 8
    corners = [
        gray[:corner_size, :corner_size].mean(),  # top-left
        gray[:corner_size, -corner_size:].mean(),  # top-right
        gray[-corner_size:, :corner_size].mean(),  # bottom-left
        gray[-corner_size:, -corner_size:].mean(),  # bottom-right
    ]
    avg_corner = np.mean(corners)
    max_corner = max(corners)  # At least one bright corner suggests paper
    bright_corners_count = sum(1 for c in corners if c > 150)
    
    # Check edge sharpness (digital images have very sharp edges)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)
    
    # Check for uniform color regions (digital boards have perfect flat colors)
    # Sample the center region
    center_h, center_w = h // 3, w // 3
    center = gray[center_h:2*center_h, center_w:2*center_w]
    
    # Calculate local variance using sliding window
    local_var = ndimage.generic_filter(center.astype(float), np.var, size=5)
    mean_local_var = np.mean(local_var)
    
    # Book/printed diagram detection:
    # - Portrait orientation (phone photo of book)
    # - At least some bright borders/corners (white paper, may have shadows)
    # - iOS/phone photo naming pattern
    is_portrait = aspect < 0.8
    is_phone_photo = "ios" in filename_lower or "img_" in filename_lower or "photo" in filename_lower
    has_bright_background = avg_border > 140 or max_corner > 180 or bright_corners_count >= 2
    
    if is_portrait and (has_bright_background or is_phone_photo):
        return ImageSource.BOOK_DIAGRAM
    
    # High avg_border + moderate edge density = likely book/scan (landscape)
    if avg_border > 200 and edge_density < 0.08:
        return ImageSource.BOOK_DIAGRAM
    
    # Very low local variance = digital (perfect flat colors)
    if mean_local_var < 50 and edge_density > 0.02:
        return ImageSource.DIGITAL_SCREENSHOT
    
    # Higher local variance = physical board (texture, lighting variations)
    if mean_local_var > 200:
        return ImageSource.PHYSICAL_BOARD
    
    # Default based on aspect ratio
    # Phone photos are typically tall (portrait), screenshots vary
    if aspect < 0.8:  # Tall portrait
        return ImageSource.BOOK_DIAGRAM  # Likely phone photo of book page
    elif 0.9 < aspect < 1.1:  # Nearly square
        return ImageSource.DIGITAL_SCREENSHOT  # Likely cropped screenshot
    
    return ImageSource.UNKNOWN


# ============================
# Image Preprocessing
# ============================

def preprocess_for_detection(
    img_bgr: np.ndarray,
    source_type: str = ImageSource.UNKNOWN,
    apply_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: int = 8,
    apply_sharpening: bool = False,
    sharpen_strength: float = 1.0,
    apply_denoise: bool = False,
    denoise_strength: int = 10,
    debug: bool = False,
) -> np.ndarray:
    """
    Preprocess an image before board detection.
    
    This enhances contrast and edges to improve detection accuracy,
    especially for photos of printed diagrams (books, magazines).
    
    Args:
        img_bgr: Input BGR image
        source_type: Type of source (affects preprocessing strategy)
        apply_clahe: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe_clip_limit: CLAHE clip limit (higher = more contrast)
        clahe_grid_size: CLAHE tile grid size
        apply_sharpening: Apply unsharp mask sharpening
        sharpen_strength: Sharpening strength (1.0 = normal)
        apply_denoise: Apply bilateral filtering for noise reduction
        denoise_strength: Bilateral filter strength
        debug: Print debug info
        
    Returns:
        Preprocessed BGR image
    """
    result = img_bgr.copy()
    
    # Source-specific preprocessing adjustments
    if source_type == ImageSource.BOOK_DIAGRAM:
        # Book diagrams: contrast enhancement helps, but sharpening can create too many edges
        # which disrupts contour detection. Use moderate CLAHE only.
        apply_clahe = True
        clahe_clip_limit = 2.5
        clahe_grid_size = 8
        apply_sharpening = False  # Sharpening disrupts contour detection
        apply_denoise = False
        if debug:
            print(f"[Preprocess] Book diagram mode: CLAHE={clahe_clip_limit}, no sharpen")
            
    elif source_type == ImageSource.DIGITAL_SCREENSHOT:
        # Digital screenshots already have good contrast, minimal processing
        apply_clahe = False
        apply_sharpening = False
        apply_denoise = False
        if debug:
            print("[Preprocess] Digital screenshot mode: minimal processing")
            
    elif source_type == ImageSource.PHYSICAL_BOARD:
        # Physical boards: moderate contrast enhancement
        apply_clahe = True
        clahe_clip_limit = 2.0
        apply_sharpening = False
        apply_denoise = True
        denoise_strength = 7
        if debug:
            print("[Preprocess] Physical board mode: moderate CLAHE, denoise")
            
    elif source_type == ImageSource.SCAN:
        # Scans: similar to book but may have less noise
        apply_clahe = True
        clahe_clip_limit = 2.5
        apply_sharpening = True
        sharpen_strength = 1.2
        if debug:
            print("[Preprocess] Scan mode: CLAHE + light sharpening")
    
    # Apply CLAHE for contrast enhancement
    if apply_clahe:
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_grid_size, clahe_grid_size))
        l_enhanced = clahe.apply(l_channel)
        
        # Merge and convert back
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        if debug:
            print(f"[Preprocess] Applied CLAHE: clip={clahe_clip_limit}, grid={clahe_grid_size}")
    
    # Apply bilateral filtering for noise reduction (preserves edges)
    if apply_denoise:
        result = cv2.bilateralFilter(result, d=9, sigmaColor=denoise_strength*5, sigmaSpace=denoise_strength*5)
        if debug:
            print(f"[Preprocess] Applied bilateral denoise: strength={denoise_strength}")
    
    # Apply unsharp mask sharpening
    if apply_sharpening:
        # Gaussian blur
        blurred = cv2.GaussianBlur(result, (0, 0), 3)
        # Unsharp mask: original + strength * (original - blurred)
        result = cv2.addWeighted(result, 1.0 + sharpen_strength, blurred, -sharpen_strength, 0)
        if debug:
            print(f"[Preprocess] Applied sharpening: strength={sharpen_strength}")
    
    return result


def preprocess_auto(img_bgr: np.ndarray, filename: str = "", debug: bool = False) -> Tuple[np.ndarray, str]:
    """
    Automatically detect source type and apply appropriate preprocessing.
    
    Args:
        img_bgr: Input BGR image
        filename: Original filename (used for source type hints)
        debug: Print debug info
        
    Returns:
        Tuple of (preprocessed_image, detected_source_type)
    """
    source_type = detect_image_source(img_bgr, filename)
    
    if debug:
        print(f"[Preprocess] Detected source type: {source_type}")
    
    preprocessed = preprocess_for_detection(img_bgr, source_type=source_type, debug=debug)
    
    return preprocessed, source_type


# ============================
# Grid line detection (for book diagrams)
# ============================

def detect_grid_lines(
    img_bgr: np.ndarray,
    min_line_length: int = None,
    max_line_gap: int = 10,
    angle_tolerance: float = 5.0,
    debug: bool = False,
) -> Tuple[List[float], List[float]]:
    """
    Detect horizontal and vertical grid lines in the image using Hough Line Transform.
    
    Returns:
        Tuple of (horizontal_y_positions, vertical_x_positions) - sorted lists of line positions
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    if min_line_length is None:
        min_line_length = min(h, w) // 12  # Lines should be at least 1/12 of image size
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Dilate edges slightly to connect broken lines
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Use probabilistic Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    if lines is None:
        return [], []
    
    horizontal_lines = []  # Store y-coordinates
    vertical_lines = []    # Store x-coordinates
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate angle
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        
        # Horizontal lines (angle near 0 or 180)
        if abs(angle) < angle_tolerance or abs(abs(angle) - 180) < angle_tolerance:
            y_avg = (y1 + y2) / 2
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            horizontal_lines.append((y_avg, line_length))
        
        # Vertical lines (angle near 90 or -90)
        elif abs(abs(angle) - 90) < angle_tolerance:
            x_avg = (x1 + x2) / 2
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            vertical_lines.append((x_avg, line_length))
    
    if debug:
        print(f"[GridLines] Found {len(horizontal_lines)} horizontal, {len(vertical_lines)} vertical line segments")
    
    # Cluster nearby lines (within 2% of image dimension)
    h_tolerance = h * 0.02
    v_tolerance = w * 0.02
    
    def cluster_lines(lines_with_length, tolerance):
        """Cluster lines that are close together, weighted by length."""
        if not lines_with_length:
            return []
        
        # Sort by position
        sorted_lines = sorted(lines_with_length, key=lambda x: x[0])
        
        clusters = []
        current_cluster = [sorted_lines[0]]
        
        for pos, length in sorted_lines[1:]:
            if pos - current_cluster[-1][0] < tolerance:
                current_cluster.append((pos, length))
            else:
                # Finish current cluster - use weighted average by line length
                total_length = sum(l for _, l in current_cluster)
                weighted_pos = sum(p * l for p, l in current_cluster) / total_length
                clusters.append(weighted_pos)
                current_cluster = [(pos, length)]
        
        # Don't forget last cluster
        if current_cluster:
            total_length = sum(l for _, l in current_cluster)
            weighted_pos = sum(p * l for p, l in current_cluster) / total_length
            clusters.append(weighted_pos)
        
        return clusters
    
    h_clusters = cluster_lines(horizontal_lines, h_tolerance)
    v_clusters = cluster_lines(vertical_lines, v_tolerance)
    
    if debug:
        print(f"[GridLines] Clustered to {len(h_clusters)} horizontal, {len(v_clusters)} vertical lines")
    
    return h_clusters, v_clusters


def find_9_grid_lines(lines: List[float], expected_spacing: float = None, tolerance_ratio: float = 0.3) -> Optional[List[float]]:
    """
    From a list of line positions, find 9 evenly-spaced lines (8 spaces = chess board).
    
    Args:
        lines: Sorted list of line positions
        expected_spacing: Expected spacing between lines (if known)
        tolerance_ratio: How much variance in spacing to allow (0.3 = 30%)
        
    Returns:
        List of 9 line positions if found, None otherwise
    """
    if len(lines) < 9:
        return None
    
    # Try each possible starting line
    best_set = None
    best_variance = float('inf')
    
    for start_idx in range(len(lines) - 8):
        # Take 9 consecutive lines
        candidate = lines[start_idx:start_idx + 9]
        
        # Calculate spacings
        spacings = [candidate[i+1] - candidate[i] for i in range(8)]
        mean_spacing = np.mean(spacings)
        variance = np.std(spacings) / mean_spacing if mean_spacing > 0 else float('inf')
        
        # Check if spacings are reasonably uniform
        if variance < tolerance_ratio and variance < best_variance:
            best_variance = variance
            best_set = candidate
    
    # Also try non-consecutive lines if we have more than 9
    if len(lines) > 9:
        from itertools import combinations
        
        # Limit search to avoid combinatorial explosion
        if len(lines) <= 15:
            for combo in combinations(range(len(lines)), 9):
                candidate = [lines[i] for i in combo]
                spacings = [candidate[i+1] - candidate[i] for i in range(8)]
                mean_spacing = np.mean(spacings)
                variance = np.std(spacings) / mean_spacing if mean_spacing > 0 else float('inf')
                
                if variance < tolerance_ratio * 0.5 and variance < best_variance:  # Stricter for non-consecutive
                    best_variance = variance
                    best_set = candidate
    
    return best_set


def detect_board_from_grid_lines(
    img_bgr: np.ndarray,
    debug: bool = False,
) -> Optional[np.ndarray]:
    """
    Detect chess board by finding 9 horizontal and 9 vertical grid lines.
    This is specifically designed for book diagrams with printed grid lines.
    
    Returns:
        Quad (4 corners) or None if not found
    """
    h, w = img_bgr.shape[:2]
    
    # Detect all grid lines
    h_lines, v_lines = detect_grid_lines(img_bgr, debug=debug)
    
    if debug:
        print(f"[GridBoard] H-lines: {len(h_lines)}, V-lines: {len(v_lines)}")
    
    # Find 9 evenly-spaced lines in each direction
    h_grid = find_9_grid_lines(h_lines)
    v_grid = find_9_grid_lines(v_lines)
    
    if h_grid is None or v_grid is None:
        if debug:
            print(f"[GridBoard] Could not find 9 evenly-spaced lines (H: {h_grid is not None}, V: {v_grid is not None})")
        return None
    
    if debug:
        h_spacing = np.mean([h_grid[i+1] - h_grid[i] for i in range(8)])
        v_spacing = np.mean([v_grid[i+1] - v_grid[i] for i in range(8)])
        print(f"[GridBoard] Found 9x9 grid: H-spacing={h_spacing:.1f}, V-spacing={v_spacing:.1f}")
    
    # The corners are the intersections of the outermost lines
    top = h_grid[0]
    bottom = h_grid[8]
    left = v_grid[0]
    right = v_grid[8]
    
    # Create quad: TL, TR, BR, BL
    quad = np.array([
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom]
    ], dtype=np.int32).reshape(-1, 1, 2)
    
    if debug:
        print(f"[GridBoard] Board corners: TL=({left:.0f},{top:.0f}), BR=({right:.0f},{bottom:.0f})")
    
    return quad


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


def refine_quad_edges(img_bgr: np.ndarray, quad: np.ndarray, search_range: int = 30) -> np.ndarray:
    """
    Refine quad edges by finding actual board boundaries using edge detection.
    Looks for strong edges near the detected quad corners and adjusts.
    
    This helps when the initial detection is slightly off (cutting into the board).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Get edges with Canny
    edges = cv2.Canny(gray, 50, 150)
    
    # Order points: TL, TR, BR, BL
    rect = order_points(quad)
    refined = rect.copy()
    
    # For each corner, search in a small region to find the strongest edge
    for i, (px, py) in enumerate(rect):
        px, py = int(px), int(py)
        
        # Define search region
        x1 = max(0, px - search_range)
        x2 = min(w, px + search_range)
        y1 = max(0, py - search_range)
        y2 = min(h, py + search_range)
        
        # Extract region
        region = edges[y1:y2, x1:x2]
        
        if region.size == 0:
            continue
        
        # Find edge pixels in region
        edge_points = np.where(region > 0)
        if len(edge_points[0]) == 0:
            continue
        
        # For corner refinement, we want to find where two edges meet
        # Use the centroid of edge pixels as a simple approach
        edge_y = edge_points[0].mean() + y1
        edge_x = edge_points[1].mean() + x1
        
        # Only adjust if the change is reasonable (not too far from original)
        dist = np.sqrt((edge_x - px)**2 + (edge_y - py)**2)
        if dist < search_range * 0.8:
            # Weight between original and detected (favor detected if edge is strong)
            edge_strength = len(edge_points[0]) / region.size
            if edge_strength > 0.05:  # Enough edge pixels
                refined[i] = [edge_x, edge_y]
    
    return refined.reshape(-1, 1, 2).astype(np.int32)


def expand_quad_to_edges(img_bgr: np.ndarray, quad: np.ndarray, max_expand: int = 20) -> np.ndarray:
    """
    Expand quad outward to find the true board edge.
    
    Strategy: Look for a clear gradient peak (edge) within the search range.
    Only expand if we find a distinct edge that's stronger than the current position.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Compute gradient magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    
    # Order points: TL, TR, BR, BL
    rect = order_points(quad)
    expanded = rect.copy()
    
    min_edge_strength = 30  # Minimum gradient to be considered a real edge
    
    def find_nearest_strong_edge(gradients: np.ndarray, current_grad: float) -> int:
        """Find the nearest position with a strong edge peak."""
        if len(gradients) == 0:
            return 0
        # Need the edge to be notably stronger than current position
        threshold = max(min_edge_strength, current_grad * 1.3)
        for i, g in enumerate(gradients):
            if g > threshold:
                return i
        return 0  # No strong edge found, don't expand
    
    # Top edge (TL to TR) - scan upward
    x1, x2 = int(min(rect[0][0], rect[1][0])), int(max(rect[0][0], rect[1][0]))
    current_y = int(min(rect[0][1], rect[1][1]))
    if x1 < x2 and x1 >= 0 and x2 <= w:
        current_grad = gradient[max(0, current_y), x1:x2].mean() if current_y >= 0 else 0
        gradients = []
        for step in range(1, max_expand + 1):
            new_y = current_y - step
            if new_y < 0:
                break
            gradients.append(gradient[new_y, x1:x2].mean())
        offset = find_nearest_strong_edge(gradients, current_grad)
        if offset > 0:
            expanded[0][1] = current_y - offset
            expanded[1][1] = current_y - offset
    
    # Bottom edge (BL to BR) - scan downward
    x1, x2 = int(min(rect[2][0], rect[3][0])), int(max(rect[2][0], rect[3][0]))
    current_y = int(max(rect[2][1], rect[3][1]))
    if x1 < x2 and x1 >= 0 and x2 <= w:
        current_grad = gradient[min(h-1, current_y), x1:x2].mean() if current_y < h else 0
        gradients = []
        for step in range(1, max_expand + 1):
            new_y = current_y + step
            if new_y >= h:
                break
            gradients.append(gradient[new_y, x1:x2].mean())
        offset = find_nearest_strong_edge(gradients, current_grad)
        if offset > 0:
            expanded[2][1] = current_y + offset
            expanded[3][1] = current_y + offset
    
    # Left edge (TL to BL) - scan leftward
    y1, y2 = int(min(rect[0][1], rect[3][1])), int(max(rect[0][1], rect[3][1]))
    current_x = int(min(rect[0][0], rect[3][0]))
    if y1 < y2 and y1 >= 0 and y2 <= h:
        current_grad = gradient[y1:y2, max(0, current_x)].mean() if current_x >= 0 else 0
        gradients = []
        for step in range(1, max_expand + 1):
            new_x = current_x - step
            if new_x < 0:
                break
            gradients.append(gradient[y1:y2, new_x].mean())
        offset = find_nearest_strong_edge(gradients, current_grad)
        if offset > 0:
            expanded[0][0] = current_x - offset
            expanded[3][0] = current_x - offset
    
    # Right edge (TR to BR) - scan rightward
    y1, y2 = int(min(rect[1][1], rect[2][1])), int(max(rect[1][1], rect[2][1]))
    current_x = int(max(rect[1][0], rect[2][0]))
    if y1 < y2 and y1 >= 0 and y2 <= h:
        current_grad = gradient[y1:y2, min(w-1, current_x)].mean() if current_x < w else 0
        gradients = []
        for step in range(1, max_expand + 1):
            new_x = current_x + step
            if new_x >= w:
                break
            gradients.append(gradient[y1:y2, new_x].mean())
        offset = find_nearest_strong_edge(gradients, current_grad)
        if offset > 0:
            expanded[1][0] = current_x + offset
            expanded[2][0] = current_x + offset
    
    return expanded.reshape(-1, 1, 2).astype(np.int32)


def refine_quad_for_digital_board(img_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """
    Refine quad for digital chess screenshots.
    
    Digital boards have:
    - Sharp, precise edges (no shadows or 3D effects)
    - Clear transition from board to black/white background
    - Should be perfectly square
    
    This function finds the exact pixel boundaries by scanning for
    the transition from background to board.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    rect = order_points(quad)
    
    # Get approximate center and size of quad
    center_x = rect[:, 0].mean()
    center_y = rect[:, 1].mean()
    approx_size = max(rect[1][0] - rect[0][0], rect[3][1] - rect[0][1])
    
    # Define search regions around each edge
    margin = 30  # pixels to search beyond current edge
    
    def find_edge_transition(scan_line: np.ndarray, from_dark: bool = True) -> int:
        """Find where the scan line transitions from dark to light (or vice versa)."""
        threshold = 30  # brightness threshold for "dark"
        
        if from_dark:
            # Scanning from dark background into board
            for i, val in enumerate(scan_line):
                if val > threshold:
                    return i
        else:
            # Scanning from light into dark
            for i, val in enumerate(scan_line):
                if val < threshold:
                    return i - 1
        return 0
    
    # Find left edge: scan from left until we hit board content
    left_x = int(rect[0][0])
    scan_start = max(0, left_x - margin)
    scan_end = min(w, left_x + margin)
    # Take a vertical strip and find median brightness at each x
    y1, y2 = int(rect[0][1] + approx_size * 0.2), int(rect[3][1] - approx_size * 0.2)
    left_scan = np.array([gray[y1:y2, x].mean() for x in range(scan_start, scan_end)])
    left_offset = find_edge_transition(left_scan, from_dark=True)
    new_left = scan_start + left_offset
    
    # Find right edge: scan from right edge into board
    right_x = int(rect[1][0])
    scan_start = max(0, right_x - margin)
    scan_end = min(w, right_x + margin)
    right_scan = np.array([gray[y1:y2, x].mean() for x in range(scan_end - 1, scan_start - 1, -1)])
    right_offset = find_edge_transition(right_scan, from_dark=True)
    new_right = scan_end - 1 - right_offset
    
    # Find top edge
    top_y = int(rect[0][1])
    scan_start = max(0, top_y - margin)
    scan_end = min(h, top_y + margin)
    x1, x2 = int(rect[0][0] + approx_size * 0.2), int(rect[1][0] - approx_size * 0.2)
    top_scan = np.array([gray[y, x1:x2].mean() for y in range(scan_start, scan_end)])
    top_offset = find_edge_transition(top_scan, from_dark=True)
    new_top = scan_start + top_offset
    
    # Find bottom edge
    bottom_y = int(rect[2][1])
    scan_start = max(0, bottom_y - margin)
    scan_end = min(h, bottom_y + margin)
    bottom_scan = np.array([gray[y, x1:x2].mean() for y in range(scan_end - 1, scan_start - 1, -1)])
    bottom_offset = find_edge_transition(bottom_scan, from_dark=True)
    new_bottom = scan_end - 1 - bottom_offset
    
    # Enforce squareness: use the larger dimension
    detected_width = new_right - new_left
    detected_height = new_bottom - new_top
    
    if abs(detected_width - detected_height) > 5:
        # Not square - adjust to make square
        target_size = max(detected_width, detected_height)
        
        # Center the adjustment
        if detected_width < target_size:
            diff = target_size - detected_width
            new_left = max(0, new_left - diff // 2)
            new_right = min(w - 1, new_right + (diff - diff // 2))
        
        if detected_height < target_size:
            diff = target_size - detected_height
            new_top = max(0, new_top - diff // 2)
            new_bottom = min(h - 1, new_bottom + (diff - diff // 2))
    
    # Build refined quad
    refined = np.array([
        [new_left, new_top],
        [new_right, new_top],
        [new_right, new_bottom],
        [new_left, new_bottom]
    ], dtype=np.float32)
    
    return refined.reshape(-1, 1, 2).astype(np.int32)


def get_tile_uniformity(warped: np.ndarray, row: int, col: int, margin_ratio: float = 0.15) -> float:
    """
    Measure how uniform a tile's coloring is.
    A properly aligned empty square should have low variance.
    A misaligned tile will show color bleeding with high variance.
    
    Returns: coefficient of variation (std/mean) - lower is more uniform.
    """
    h, w = warped.shape[:2]
    cell_h, cell_w = h // 8, w // 8
    
    # Get tile with small margin to avoid grid lines
    margin_y = int(cell_h * margin_ratio)
    margin_x = int(cell_w * margin_ratio)
    
    y1 = row * cell_h + margin_y
    y2 = (row + 1) * cell_h - margin_y
    x1 = col * cell_w + margin_x
    x2 = (col + 1) * cell_w - margin_x
    
    tile = warped[y1:y2, x1:x2]
    
    # Convert to grayscale if color
    if len(tile.shape) == 3:
        tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    
    mean_val = np.mean(tile)
    std_val = np.std(tile)
    
    if mean_val < 1:
        return 1.0  # Avoid division by zero
    
    return std_val / mean_val  # Coefficient of variation


def get_edge_tiles_uniformity(warped: np.ndarray, edge: str) -> float:
    """
    Get average uniformity of tiles along an edge.
    edge: 'top', 'bottom', 'left', 'right'
    """
    uniformities = []
    
    if edge == 'top':  # row 0, all columns
        for col in range(8):
            uniformities.append(get_tile_uniformity(warped, 0, col))
    elif edge == 'bottom':  # row 7, all columns
        for col in range(8):
            uniformities.append(get_tile_uniformity(warped, 7, col))
    elif edge == 'left':  # column 0, all rows
        for row in range(8):
            uniformities.append(get_tile_uniformity(warped, row, 0))
    elif edge == 'right':  # column 7, all rows
        for row in range(8):
            uniformities.append(get_tile_uniformity(warped, row, 7))
    
    return np.mean(uniformities) if uniformities else 1.0


def detect_grid_offset(warped: np.ndarray, max_offset: int = 10, debug: bool = False) -> Tuple[int, int]:
    """
    Detect if the grid is misaligned by analyzing empty squares.
    
    Empty squares should have uniform color. If an "empty" square has
    color intrusion on one edge, the grid is shifted that direction.
    
    Returns: (dx, dy) pixel offset to correct the grid.
             Positive dx means shift grid right, positive dy means shift grid down.
    """
    h, w = warped.shape[:2]
    cell_h, cell_w = h // 8, w // 8
    
    # Analyze each tile to determine if it's likely empty
    likely_empty = []
    
    for row in range(8):
        for col in range(8):
            # Get tile
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            tile = warped[y1:y2, x1:x2]
            
            # Convert to grayscale
            if len(tile.shape) == 3:
                gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            else:
                gray_tile = tile
            
            # Check center region (avoid edges) for uniformity
            margin = cell_h // 4
            center = gray_tile[margin:-margin, margin:-margin]
            center_std = np.std(center)
            center_mean = np.mean(center)
            
            # Very uniform center suggests empty square
            # Chess squares are typically one solid color
            cv_center = center_std / (center_mean + 1)
            
            if cv_center < 0.08:  # Very uniform center
                likely_empty.append((row, col, cv_center, center_mean))
    
    if len(likely_empty) < 4:
        return 0, 0  # Not enough empty squares to analyze
    
    # For each likely-empty square, check for edge intrusions
    edge_scores = {'left': [], 'right': [], 'top': [], 'bottom': []}
    
    for row, col, cv, mean_val in likely_empty:
        y1, y2 = row * cell_h, (row + 1) * cell_h
        x1, x2 = col * cell_w, (col + 1) * cell_w
        tile = warped[y1:y2, x1:x2]
        
        if len(tile.shape) == 3:
            gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        else:
            gray_tile = tile
        
        # Check each edge strip for variance (intrusion = high variance)
        strip_width = max(2, cell_w // 10)
        
        # Left edge strip
        left_strip = gray_tile[:, :strip_width]
        left_var = np.std(left_strip)
        
        # Right edge strip
        right_strip = gray_tile[:, -strip_width:]
        right_var = np.std(right_strip)
        
        # Top edge strip
        top_strip = gray_tile[:strip_width, :]
        top_var = np.std(top_strip)
        
        # Bottom edge strip
        bottom_strip = gray_tile[-strip_width:, :]
        bottom_var = np.std(bottom_strip)
        
        # Center variance (baseline)
        margin = cell_h // 4
        center_var = np.std(gray_tile[margin:-margin, margin:-margin])
        
        # Score is how much worse the edge is compared to center
        # High score means intrusion from that direction
        if center_var > 0:
            edge_scores['left'].append(left_var / (center_var + 1))
            edge_scores['right'].append(right_var / (center_var + 1))
            edge_scores['top'].append(top_var / (center_var + 1))
            edge_scores['bottom'].append(bottom_var / (center_var + 1))
    
    # Compute average edge scores
    avg_scores = {}
    for edge, scores in edge_scores.items():
        if scores:
            avg_scores[edge] = np.mean(scores)
        else:
            avg_scores[edge] = 1.0
    
    if debug:
        print(f"Grid offset detection - empty squares found: {len(likely_empty)}")
        print(f"  Edge scores: left={avg_scores.get('left', 0):.2f}, right={avg_scores.get('right', 0):.2f}, "
              f"top={avg_scores.get('top', 0):.2f}, bottom={avg_scores.get('bottom', 0):.2f}")
    
    # Determine offset based on imbalance
    # If left edge has higher variance than right, grid should shift right (positive dx)
    dx = 0
    dy = 0
    
    threshold = 1.5  # Edge must be 50% worse than opposite to trigger correction
    
    left_score = avg_scores.get('left', 1.0)
    right_score = avg_scores.get('right', 1.0)
    if left_score > right_score * threshold:
        # Intrusion from left = pieces bleeding in from left neighbor
        # Need to shift grid RIGHT (positive dx)
        dx = min(max_offset, int((left_score - right_score) * 2))
    elif right_score > left_score * threshold:
        # Intrusion from right = shift grid LEFT
        dx = -min(max_offset, int((right_score - left_score) * 2))
    
    top_score = avg_scores.get('top', 1.0)
    bottom_score = avg_scores.get('bottom', 1.0)
    if top_score > bottom_score * threshold:
        # Intrusion from top = shift grid DOWN
        dy = min(max_offset, int((top_score - bottom_score) * 2))
    elif bottom_score > top_score * threshold:
        # Intrusion from bottom = shift grid UP
        dy = -min(max_offset, int((bottom_score - top_score) * 2))
    
    if debug and (dx != 0 or dy != 0):
        print(f"  Suggested grid offset: dx={dx}, dy={dy}")
    
    return dx, dy


def apply_grid_offset(warped: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    Apply a pixel offset to the warped board by padding/cropping.
    
    dx > 0: shift grid right (add padding on left, crop right)
    dy > 0: shift grid down (add padding on top, crop bottom)
    """
    if dx == 0 and dy == 0:
        return warped
    
    h, w = warped.shape[:2]
    
    # Create translation matrix
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    
    # Warp with border replication to avoid black edges
    shifted = cv2.warpAffine(warped, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    return shifted


def refine_grid_alignment(warped: np.ndarray, max_iterations: int = 3, 
                          max_offset: int = 8, debug: bool = False) -> Tuple[np.ndarray, int, int]:
    """
    Iteratively refine grid alignment by detecting and correcting offsets.
    
    Returns: (aligned_warped, total_dx, total_dy)
    """
    current = warped.copy()
    total_dx, total_dy = 0, 0
    
    for iteration in range(max_iterations):
        dx, dy = detect_grid_offset(current, max_offset=max_offset, debug=debug)
        
        if dx == 0 and dy == 0:
            break
        
        current = apply_grid_offset(current, dx, dy)
        total_dx += dx
        total_dy += dy
        
        if debug:
            print(f"  Iteration {iteration+1}: applied offset dx={dx}, dy={dy}")
    
    return current, total_dx, total_dy


def refine_quad_by_tile_uniformity(img_bgr: np.ndarray, quad: np.ndarray, 
                                    max_adjust: int = 15, step: int = 2,
                                    warp_size: int = 640) -> np.ndarray:
    """
    Refine quad corners by optimizing tile uniformity.
    
    For each edge, try adjusting it in/out and pick the position
    that gives the most uniform edge tiles.
    
    This catches cases where gradient detection fails but the tiles
    show clear color bleeding from misalignment.
    """
    h, w = img_bgr.shape[:2]
    rect = order_points(quad)
    best_rect = rect.copy()
    
    # Test each edge independently
    edges = ['top', 'bottom', 'left', 'right']
    
    for edge in edges:
        best_uniformity = float('inf')
        best_offset = 0
        
        # Try adjustments from -max_adjust to +max_adjust
        for offset in range(-max_adjust, max_adjust + 1, step):
            test_rect = best_rect.copy()
            
            if edge == 'top':
                new_y = best_rect[0][1] + offset
                if new_y < 0 or new_y >= h:
                    continue
                test_rect[0][1] = new_y  # TL
                test_rect[1][1] = new_y  # TR
            elif edge == 'bottom':
                new_y = best_rect[2][1] + offset
                if new_y < 0 or new_y >= h:
                    continue
                test_rect[2][1] = new_y  # BR
                test_rect[3][1] = new_y  # BL
            elif edge == 'left':
                new_x = best_rect[0][0] + offset
                if new_x < 0 or new_x >= w:
                    continue
                test_rect[0][0] = new_x  # TL
                test_rect[3][0] = new_x  # BL
            elif edge == 'right':
                new_x = best_rect[1][0] + offset
                if new_x < 0 or new_x >= w:
                    continue
                test_rect[1][0] = new_x  # TR
                test_rect[2][0] = new_x  # BR
            
            # Warp with test rect
            try:
                warped = warp_quad(img_bgr, test_rect, out_size=warp_size)
                uniformity = get_edge_tiles_uniformity(warped, edge)
                
                if uniformity < best_uniformity:
                    best_uniformity = uniformity
                    best_offset = offset
            except:
                continue
        
        # Apply best offset for this edge (with bounds checking)
        if edge == 'top':
            new_y = max(0, min(h-1, best_rect[0][1] + best_offset))
            best_rect[0][1] = new_y
            best_rect[1][1] = new_y
        elif edge == 'bottom':
            new_y = max(0, min(h-1, best_rect[2][1] + best_offset))
            best_rect[2][1] = new_y
            best_rect[3][1] = new_y
        elif edge == 'left':
            new_x = max(0, min(w-1, best_rect[0][0] + best_offset))
            best_rect[0][0] = new_x
            best_rect[3][0] = new_x
        elif edge == 'right':
            new_x = max(0, min(w-1, best_rect[1][0] + best_offset))
            best_rect[1][0] = new_x
            best_rect[2][0] = new_x
    
    return best_rect.reshape(-1, 1, 2).astype(np.int32)


def enforce_square_aspect(img_bgr: np.ndarray, quad: np.ndarray, 
                          warp_size: int = 640, debug: bool = False) -> np.ndarray:
    """
    For printed/book diagrams, the board should be perfectly square.
    Adjust the quad to have 1:1 aspect ratio while maximizing grid score.
    
    Strategy: Keep the smaller dimension and adjust the larger one.
    Try both expanding and shrinking, pick the one with best grid score.
    """
    rect = order_points(quad)
    
    # Calculate current dimensions
    width = np.linalg.norm(rect[1] - rect[0])   # Top edge
    height = np.linalg.norm(rect[3] - rect[0])  # Left edge
    
    if debug:
        print(f"[SquareAspect] Current: {width:.0f}x{height:.0f}, ratio={width/height:.3f}")
    
    # Already square enough
    if 0.98 <= width/height <= 1.02:
        if debug:
            print("[SquareAspect] Already square, no adjustment needed")
        return quad
    
    candidates = []
    
    # Original (for comparison)
    warped_orig = warp_quad(img_bgr, quad, out_size=warp_size)
    score_orig = combined_grid_score(warped_orig)
    candidates.append(('original', quad, score_orig))
    
    # Try making it square by adjusting width to match height
    if width > height:
        # Width is larger - shrink it OR expand height
        # Option 1: Shrink width
        scale = height / width
        center_x = (rect[0][0] + rect[1][0]) / 2
        new_left = center_x - (height / 2)
        new_right = center_x + (height / 2)
        shrink_rect = rect.copy()
        shrink_rect[0][0] = shrink_rect[3][0] = new_left
        shrink_rect[1][0] = shrink_rect[2][0] = new_right
        shrink_quad = shrink_rect.reshape(-1, 1, 2).astype(np.int32)
        warped = warp_quad(img_bgr, shrink_quad, out_size=warp_size)
        score = combined_grid_score(warped)
        candidates.append(('shrink_width', shrink_quad, score))
        
        # Option 2: Expand height
        center_y = (rect[0][1] + rect[3][1]) / 2
        new_top = center_y - (width / 2)
        new_bottom = center_y + (width / 2)
        expand_rect = rect.copy()
        expand_rect[0][1] = expand_rect[1][1] = new_top
        expand_rect[2][1] = expand_rect[3][1] = new_bottom
        expand_quad = expand_rect.reshape(-1, 1, 2).astype(np.int32)
        warped = warp_quad(img_bgr, expand_quad, out_size=warp_size)
        score = combined_grid_score(warped)
        candidates.append(('expand_height', expand_quad, score))
    else:
        # Height is larger - shrink it OR expand width
        # Option 1: Shrink height
        center_y = (rect[0][1] + rect[3][1]) / 2
        new_top = center_y - (width / 2)
        new_bottom = center_y + (width / 2)
        shrink_rect = rect.copy()
        shrink_rect[0][1] = shrink_rect[1][1] = new_top
        shrink_rect[2][1] = shrink_rect[3][1] = new_bottom
        shrink_quad = shrink_rect.reshape(-1, 1, 2).astype(np.int32)
        warped = warp_quad(img_bgr, shrink_quad, out_size=warp_size)
        score = combined_grid_score(warped)
        candidates.append(('shrink_height', shrink_quad, score))
        
        # Option 2: Expand width
        center_x = (rect[0][0] + rect[1][0]) / 2
        new_left = center_x - (height / 2)
        new_right = center_x + (height / 2)
        expand_rect = rect.copy()
        expand_rect[0][0] = expand_rect[3][0] = new_left
        expand_rect[1][0] = expand_rect[2][0] = new_right
        expand_quad = expand_rect.reshape(-1, 1, 2).astype(np.int32)
        warped = warp_quad(img_bgr, expand_quad, out_size=warp_size)
        score = combined_grid_score(warped)
        candidates.append(('expand_width', expand_quad, score))
    
    # Pick best candidate
    best_name, best_quad, best_score = max(candidates, key=lambda x: x[2])
    
    if debug:
        for name, q, s in candidates:
            marker = " <-- BEST" if name == best_name else ""
            print(f"[SquareAspect] {name}: score={s:.3f}{marker}")
    
    return best_quad


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
    filename: str = "",
    source_type: str = None,  # Override auto-detection
    preprocess: bool = True,  # Apply preprocessing for better edge detection
    debug: bool = False,
) -> Optional[np.ndarray]:
    """
    Detect chessboard quad using:
      1) Preprocessing (CLAHE, sharpening based on source type)
      2) center-crop (reduce text/artefacts)
      3) grid-aware scoring (peaks + morphological lines)

    Returns quad in *original image coordinates* or None.
    
    Args:
        img_bgr: Input BGR image
        crop_ratio: Auto-detected by default. Try multiple crops.
        filename: Original filename (helps detect source type)
        source_type: Override source type detection (ImageSource.BOOK_DIAGRAM, etc.)
        preprocess: Whether to apply preprocessing (default: True)
        debug: Print debug info
    """
    # Detect source type first
    if source_type is None:
        detected_source = detect_image_source(img_bgr, filename)
    else:
        detected_source = source_type
    
    if debug:
        print(f"[Detection] Source type: {detected_source}")
    
    # For book diagrams, try grid-line detection first (most accurate for printed grids)
    if detected_source == ImageSource.BOOK_DIAGRAM:
        if debug:
            print("[Detection] Trying grid-line detection for book diagram...")
        
        # Apply preprocessing for grid line detection
        img_processed = preprocess_for_detection(img_bgr, source_type=detected_source, debug=debug)
        
        # Try grid-line detection on preprocessed image
        grid_quad = detect_board_from_grid_lines(img_processed, debug=debug)
        
        if grid_quad is not None:
            # Validate with grid score
            warped = warp_quad(img_bgr, grid_quad, out_size=256)
            grid_score = combined_grid_score(warped)
            
            if debug:
                print(f"[Detection] Grid-line detection score: {grid_score:.3f}")
            
            if grid_score > 0.3:  # Good enough grid score
                if debug:
                    print("[Detection] Using grid-line detection result")
                return grid_quad
            elif debug:
                print("[Detection] Grid-line result had low score, falling back to contour detection")
        elif debug:
            print("[Detection] Grid-line detection failed, falling back to contour detection")
    
    # Apply preprocessing for contour-based detection
    if preprocess:
        if source_type is None:
            img_processed, _ = preprocess_auto(img_bgr, filename=filename, debug=debug)
        else:
            img_processed = preprocess_for_detection(img_bgr, source_type=source_type, debug=debug)
        
        if debug:
            print(f"[Detection] Preprocessing applied for contour detection")
    else:
        img_processed = img_bgr
        if debug:
            print("[Detection] Preprocessing disabled")
    
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
            # Use preprocessed image for detection
            crop, (x0, y0) = _crop_region(img_processed, crop_ratio_try, vert_pos)
            quad = _detect_board_quad_grid_aware_impl(crop, score_warp_size=score_warp_size, debug=debug)
            
            if quad is None:
                continue
            
            quad_area = cv2.contourArea(quad)
            crop_area = crop.shape[0] * crop.shape[1]
            relative_area = quad_area / crop_area
            
            # Use original image for scoring (to avoid scoring preprocessing artifacts)
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
        
        final_quad = offset_quad(best_quad, best_offset[0], best_offset[1])
        
        # For book diagrams, enforce square aspect ratio
        if detected_source == ImageSource.BOOK_DIAGRAM:
            if debug:
                print("[Detection] Applying square aspect enforcement for book diagram")
            final_quad = enforce_square_aspect(img_bgr, final_quad, debug=debug)
        
        return final_quad
    
    # Fixed crop_ratio specified - use preprocessed image for detection
    crop, (x0, y0) = center_crop(img_processed, crop_ratio=crop_ratio)
    quad_crop = _detect_board_quad_grid_aware_impl(crop, score_warp_size=score_warp_size, debug=debug)
    if quad_crop is None:
        return None
    return offset_quad(quad_crop, x0, y0)


def detect_and_warp_board(
    img_bgr: np.ndarray,
    out_size: int = 640,
    crop_ratio: float = 0.80,
    filename: str = "",
    source_type: str = None,
    preprocess: bool = True,
) -> Optional[np.ndarray]:
    quad = detect_board_quad_grid_aware(
        img_bgr, 
        crop_ratio=crop_ratio,
        filename=filename,
        source_type=source_type,
        preprocess=preprocess,
    )
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
