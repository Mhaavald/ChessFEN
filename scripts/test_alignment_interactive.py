"""
Interactive Grid Alignment Test Script

Run this on any chess board image to see:
1. Board detection and warping
2. Grid overlay with square labels
3. Piece detection with alignment analysis
4. Visual indicators showing offset

Usage:
    python scripts/test_alignment_interactive.py <image_path> [output_path]
    python scripts/test_alignment_interactive.py data/raw/batch_001/20251219_142831522_iOS.jpg

If no output path specified, saves to debug_out/alignment_<image_name>.png
"""

import sys
from pathlib import Path
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from detect_board import detect_board_quad_grid_aware, warp_quad
from piece_detect import suggest_grid_adjustment


def create_alignment_visualization(image_path: Path, output_path: Path = None):
    """
    Create a complete alignment visualization for a chess board image.
    
    Shows:
    - Original image with detected board outline
    - Warped board with grid overlay
    - Piece detection with alignment analysis
    """
    
    print(f"\n{'='*70}")
    print(f"GRID ALIGNMENT ANALYSIS")
    print(f"{'='*70}")
    print(f"Input: {image_path}")
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: Failed to load image: {image_path}")
        return None
    
    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Step 1: Detect board
    print("\n[1] Detecting chess board...")
    quad = detect_board_quad_grid_aware(img)
    
    if quad is None:
        print("   ❌ Could not detect chess board!")
        # Create output showing original with message
        output = img.copy()
        cv2.putText(output, "No board detected", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        if output_path:
            cv2.imwrite(str(output_path), output)
            print(f"   Saved to: {output_path}")
        return None
    
    print("   ✓ Board detected!")
    
    # Create detection overlay on original
    detection_img = img.copy()
    pts = quad.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(detection_img, [pts], True, (0, 255, 0), 4)
    cv2.putText(detection_img, "Detected Board", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Step 2: Warp to square
    print("\n[2] Warping to 640x640 square...")
    warped = warp_quad(img, quad, out_size=640)
    print(f"   ✓ Warped size: {warped.shape[1]}x{warped.shape[0]}")
    
    # Step 3: Analyze piece positions
    print("\n[3] Analyzing piece positions...")
    detection = suggest_grid_adjustment(warped)
    
    # Step 4: Create detailed overlay
    wh, ww = warped.shape[:2]
    sq_size = ww // 8
    overlay = warped.copy()
    
    # Draw grid lines (thicker)
    for i in range(9):
        p = i * sq_size
        cv2.line(overlay, (0, p), (ww, p), (0, 255, 0), 2)
        cv2.line(overlay, (p, 0), (p, wh), (0, 255, 0), 2)
    
    # Draw square labels
    files = "abcdefgh"
    for row in range(8):
        for col in range(8):
            label = f"{files[col]}{8-row}"
            x = col * sq_size + 5
            y = row * sq_size + 18
            cv2.putText(overlay, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Step 5: Draw piece detection results
    if detection["piece_found"]:
        cx, cy = detection["piece_center"]
        row, col = detection["piece_square"]
        dx, dy = detection["offset"]
        square_name = f"{files[col]}{8-row}"
        
        # Expected center (green circle)
        expected_cx = int((col + 0.5) * sq_size)
        expected_cy = int((row + 0.5) * sq_size)
        cv2.circle(overlay, (expected_cx, expected_cy), 15, (0, 255, 0), 3)
        
        # Actual center (red filled circle)
        cv2.circle(overlay, (int(cx), int(cy)), 10, (0, 0, 255), -1)
        
        # Offset arrow (magenta)
        cv2.arrowedLine(overlay, (expected_cx, expected_cy), (int(cx), int(cy)), 
                        (255, 0, 255), 3, tipLength=0.3)
        
        # Info text at bottom
        info_text = f"Piece in {square_name}, offset: ({dx:.0f}, {dy:.0f})px"
        cv2.putText(overlay, info_text, (10, wh - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if abs(dx) > 5 or abs(dy) > 5:
            status = "MISALIGNED"
            color = (0, 165, 255)  # Orange
            adjust_text = f"Adjust grid by ({-dx:.0f}, {-dy:.0f})px"
            cv2.putText(overlay, adjust_text, (10, wh - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            status = "ALIGNED OK"
            color = (0, 255, 0)  # Green
            cv2.putText(overlay, "Grid aligned OK", (10, wh - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        print(f"   ✓ Piece detected in square: {square_name}")
        print(f"   • Piece center: ({cx:.1f}, {cy:.1f})")
        print(f"   • Expected center: ({expected_cx}, {expected_cy})")
        print(f"   • Offset: dx={dx:.1f}px, dy={dy:.1f}px")
        print(f"   • Status: {status}")
        
        if abs(dx) > 5 or abs(dy) > 5:
            print(f"\n   ⚠️  SUGGESTION: Shift grid by ({-dx:.0f}, {-dy:.0f}) pixels")
    else:
        cv2.putText(overlay, "No piece detected", (10, wh - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print("   ❌ No piece detected (board may be empty or low contrast)")
    
    # Step 6: Create combined output
    # Resize detection image to match overlay height
    det_h = overlay.shape[0]
    det_scale = det_h / detection_img.shape[0]
    det_w = int(detection_img.shape[1] * det_scale)
    detection_resized = cv2.resize(detection_img, (det_w, det_h))
    
    # Add legend
    legend_h = 80
    legend = np.zeros((legend_h, overlay.shape[1] + det_w, 3), dtype=np.uint8)
    legend[:] = (40, 40, 40)
    
    cv2.putText(legend, "LEGEND:", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.circle(legend, (100, 50), 10, (0, 255, 0), 2)
    cv2.putText(legend, "= Expected center", (120, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(legend, (300, 50), 8, (0, 0, 255), -1)
    cv2.putText(legend, "= Actual piece center", (320, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.arrowedLine(legend, (520, 50), (570, 50), (255, 0, 255), 2)
    cv2.putText(legend, "= Offset direction", (580, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Combine side by side
    combined_top = np.hstack([detection_resized, overlay])
    combined = np.vstack([combined_top, legend])
    
    # Add title
    title_h = 40
    title = np.zeros((title_h, combined.shape[1], 3), dtype=np.uint8)
    title[:] = (60, 60, 60)
    cv2.putText(title, f"Grid Alignment Analysis: {image_path.name}", (10, 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    final = np.vstack([title, combined])
    
    # Save output
    if output_path is None:
        output_dir = REPO_ROOT / "debug_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"alignment_{image_path.stem}.png"
    
    cv2.imwrite(str(output_path), final)
    print(f"\n{'='*70}")
    print(f"OUTPUT SAVED: {output_path}")
    print(f"{'='*70}")
    
    return output_path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable test images:")
        
        # List some available images
        raw_dir = REPO_ROOT / "data" / "raw" / "batch_001"
        if raw_dir.exists():
            images = list(raw_dir.glob("*.jpg"))[:5]
            for img in images:
                print(f"  {img}")
        
        warped_dir = REPO_ROOT / "data" / "boards_warped" / "batch_001"
        if warped_dir.exists():
            images = list(warped_dir.glob("*.png"))[:3]
            print("\nWarped boards:")
            for img in images:
                print(f"  {img}")
        
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    if not image_path.is_absolute():
        image_path = REPO_ROOT / image_path
    
    if not image_path.exists():
        print(f"ERROR: File not found: {image_path}")
        sys.exit(1)
    
    output_path = None
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
    
    create_alignment_visualization(image_path, output_path)


if __name__ == "__main__":
    main()
