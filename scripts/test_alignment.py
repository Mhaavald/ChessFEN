"""
Test script demonstrating the full piece detection and grid alignment flow.

Shows:
1. Original warped board with grid overlay
2. Piece detection with alignment analysis
3. Adjusted grid (if offset detected)
4. Sample extracted tiles from both original and adjusted grids
"""

import sys
from pathlib import Path
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from detect_board import detect_board_quad_grid_aware, warp_quad
from extract import draw_grid_overlay, split_board_to_squares, square_name
from piece_detect import suggest_grid_adjustment, draw_piece_detection_overlay


def create_comparison_image(
    warped: np.ndarray,
    alignment_result: dict,
    output_dir: Path,
    name: str = "test"
):
    """Create a full comparison showing original, detection, and tiles."""
    
    h, w = warped.shape[:2]
    sq_size = w // 8
    
    # 1. Original with grid
    original_overlay = draw_grid_overlay(warped.copy(), sq_size)
    
    # 2. Piece detection overlay
    detection_overlay = draw_piece_detection_overlay(warped, alignment_result, sq_size)
    
    # 3. Adjusted grid (if offset found)
    dx, dy = 0, 0
    if alignment_result.get("piece_found"):
        dx, dy = alignment_result["offset"]
    
    # Create adjusted view by shifting
    adjusted = warped.copy()
    if abs(dx) > 2 or abs(dy) > 2:
        # Shift the image to compensate for offset
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        adjusted = cv2.warpAffine(warped, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    adjusted_overlay = draw_grid_overlay(adjusted.copy(), sq_size)
    
    # Add labels
    cv2.putText(original_overlay, "Original Grid", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(detection_overlay, "Piece Detection", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(adjusted_overlay, f"Adjusted ({-dx:.0f}, {-dy:.0f})px", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Combine top row
    top_row = np.hstack([original_overlay, detection_overlay, adjusted_overlay])
    
    # 4. Extract sample tiles - corners and center
    sample_positions = [
        (0, 0, "a8"), (0, 7, "h8"),  # Top corners
        (3, 3, "d5"), (3, 4, "e5"),  # Center
        (7, 0, "a1"), (7, 7, "h1"),  # Bottom corners
    ]
    
    # Extract tiles from original
    orig_tiles = []
    for row, col, label in sample_positions:
        y1, y2 = row * sq_size, (row + 1) * sq_size
        x1, x2 = col * sq_size, (col + 1) * sq_size
        tile = warped[y1:y2, x1:x2].copy()
        # Add label
        cv2.putText(tile, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        orig_tiles.append(tile)
    
    # Extract tiles from adjusted
    adj_tiles = []
    for row, col, label in sample_positions:
        y1, y2 = row * sq_size, (row + 1) * sq_size
        x1, x2 = col * sq_size, (col + 1) * sq_size
        tile = adjusted[y1:y2, x1:x2].copy()
        cv2.putText(tile, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        adj_tiles.append(tile)
    
    # Resize tiles to consistent size for display
    tile_display_size = sq_size
    
    # Create tile comparison rows
    orig_row = np.hstack([cv2.resize(t, (tile_display_size, tile_display_size)) for t in orig_tiles])
    adj_row = np.hstack([cv2.resize(t, (tile_display_size, tile_display_size)) for t in adj_tiles])
    
    # Add row labels
    label_width = 100
    orig_label = np.zeros((tile_display_size, label_width, 3), dtype=np.uint8)
    adj_label = np.zeros((tile_display_size, label_width, 3), dtype=np.uint8)
    cv2.putText(orig_label, "Original", (5, tile_display_size//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(adj_label, "Adjusted", (5, tile_display_size//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    orig_row_labeled = np.hstack([orig_label, orig_row])
    adj_row_labeled = np.hstack([adj_label, adj_row])
    
    # Pad tile rows to match top row width
    top_width = top_row.shape[1]
    tile_row_width = orig_row_labeled.shape[1]
    
    if tile_row_width < top_width:
        pad = np.zeros((tile_display_size, top_width - tile_row_width, 3), dtype=np.uint8)
        orig_row_labeled = np.hstack([orig_row_labeled, pad])
        adj_row_labeled = np.hstack([adj_row_labeled, pad])
    elif tile_row_width > top_width:
        # Resize top row
        scale = tile_row_width / top_width
        new_h = int(top_row.shape[0] * scale)
        top_row = cv2.resize(top_row, (tile_row_width, new_h))
    
    # Combine everything
    result = np.vstack([top_row, orig_row_labeled, adj_row_labeled])
    
    # Save
    output_path = output_dir / f"{name}_comparison.png"
    cv2.imwrite(str(output_path), result)
    print(f"Saved comparison: {output_path}")
    
    return output_path


def process_image(image_path: Path, output_dir: Path):
    """Process a single image through the full pipeline."""
    
    print(f"\n{'='*60}")
    print(f"Processing: {image_path.name}")
    print(f"{'='*60}")
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ERROR: Failed to load image")
        return None
    
    print(f"  Image size: {img.shape[1]}x{img.shape[0]}")
    
    # Detect board
    quad = detect_board_quad_grid_aware(img)
    if quad is None:
        print(f"  ERROR: Could not detect board")
        return None
    
    # Warp to square
    warped = warp_quad(img, quad, out_size=640)
    print(f"  Warped size: {warped.shape[1]}x{warped.shape[0]}")
    
    # Run piece detection
    alignment = suggest_grid_adjustment(warped)
    
    if alignment["piece_found"]:
        cx, cy = alignment["piece_center"]
        row, col = alignment["piece_square"]
        dx, dy = alignment["offset"]
        files = "abcdefgh"
        square = f"{files[col]}{8-row}"
        
        print(f"  Piece detected in: {square}")
        print(f"  Piece center: ({cx:.1f}, {cy:.1f})")
        print(f"  Grid offset: dx={dx:.1f}, dy={dy:.1f}")
        print(f"  Confidence: {alignment['confidence']:.2f}")
        
        if abs(dx) > 5 or abs(dy) > 5:
            print(f"  ⚠ MISALIGNED - suggest shifting by ({-dx:.0f}, {-dy:.0f}) pixels")
        else:
            print(f"  ✓ Grid looks aligned")
    else:
        print(f"  No piece detected (empty board or low contrast)")
    
    # Create comparison
    name = image_path.stem
    output_path = create_comparison_image(warped, alignment, output_dir, name)
    
    return output_path


def main():
    output_dir = REPO_ROOT / "debug_out" / "alignment_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("GRID ALIGNMENT TEST")
    print("="*60)
    print(f"Output directory: {output_dir}")
    
    # Test with warped boards from batch_001
    warped_dir = REPO_ROOT / "data" / "boards_warped" / "batch_001"
    
    if warped_dir.exists():
        images = sorted(warped_dir.glob("*.png"))[:3]  # Test first 3
        print(f"\nTesting with {len(images)} warped boards from batch_001")
        
        for img_path in images:
            # For already warped images, just run alignment
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            print(f"\n{'='*60}")
            print(f"Analyzing: {img_path.name}")
            
            alignment = suggest_grid_adjustment(img)
            
            if alignment["piece_found"]:
                cx, cy = alignment["piece_center"]
                row, col = alignment["piece_square"]
                dx, dy = alignment["offset"]
                files = "abcdefgh"
                square = f"{files[col]}{8-row}"
                
                print(f"  Piece in: {square} at ({cx:.1f}, {cy:.1f})")
                print(f"  Offset: dx={dx:.1f}, dy={dy:.1f}")
                
                if abs(dx) > 5 or abs(dy) > 5:
                    print(f"  ⚠ MISALIGNED")
                else:
                    print(f"  ✓ Aligned")
            else:
                print(f"  No piece detected")
            
            create_comparison_image(img, alignment, output_dir, img_path.stem)
    
    # Also test with raw images
    raw_dir = REPO_ROOT / "data" / "raw" / "batch_001"
    if raw_dir.exists():
        extensions = ('.jpg', '.jpeg', '.png')
        raw_images = [f for f in raw_dir.iterdir() 
                      if f.suffix.lower() in extensions and 'debug' not in f.name.lower()][:2]
        
        if raw_images:
            print(f"\n\nTesting with {len(raw_images)} raw images")
            for img_path in raw_images:
                process_image(img_path, output_dir)
    
    print(f"\n{'='*60}")
    print(f"DONE - Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
