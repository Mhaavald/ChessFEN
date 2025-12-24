#!/usr/bin/env python3
"""Save detection results for batch_003 images for manual review."""

import cv2
import sys
from pathlib import Path

sys.path.append('.')
from scripts.detect_board import detect_board_quad_grid_aware, warp_quad

def save_detection_result(img_path: Path, out_dir: Path):
    """Save original with detection overlay and warped result."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"ERROR: Could not load {img_path.name}")
        return False
    
    print(f"\nProcessing: {img_path.name}")
    quad = detect_board_quad_grid_aware(img, debug=True)
    
    if quad is None:
        print(f"  FAILED - no board detected")
        return False
    
    # Draw the detected quad on original
    img_with_quad = img.copy()
    cv2.polylines(img_with_quad, [quad], True, (0, 255, 0), 3)
    
    # Warp the detected board
    warped = warp_quad(img, quad, out_size=800)
    
    # Save both
    base_name = img_path.stem
    detect_path = out_dir / f"{base_name}_detect.png"
    warped_path = out_dir / f"{base_name}_warped.png"
    
    cv2.imwrite(str(detect_path), img_with_quad)
    cv2.imwrite(str(warped_path), warped)
    
    print(f"  Saved: {detect_path.name}")
    print(f"  Saved: {warped_path.name}")
    
    return True

def main():
    batch_003 = Path("data/raw/batch_003")
    out_dir = Path("debug_out/batch_003_review")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    images = sorted(batch_003.glob("*.png"))
    
    print(f"Processing {len(images)} images from batch_003...")
    print(f"Output directory: {out_dir}")
    print("="*60)
    
    success_count = 0
    for img_path in images:
        if save_detection_result(img_path, out_dir):
            success_count += 1
    
    print("="*60)
    print(f"Complete: {success_count}/{len(images)} successful")
    print(f"\nReview the results in: {out_dir}")

if __name__ == "__main__":
    main()
