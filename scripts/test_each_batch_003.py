#!/usr/bin/env python3
"""Show what each image is detecting in detail."""

import cv2
import sys
from pathlib import Path

sys.path.append('.')
from scripts.detect_board import detect_board_quad_grid_aware, warp_quad

def test_one_image(img_path: Path):
    """Test a single image with debug output."""
    print(f"\n{'='*70}")
    print(f"Testing: {img_path.name}")
    print('='*70)
    
    img = cv2.imread(str(img_path))
    if img is None:
        print("ERROR: Could not load image")
        return False
    
    # Run detection with debug ON
    quad = detect_board_quad_grid_aware(img, debug=True)
    
    if quad is None:
        print("\n[FAILED] No board detected")
        return False
    
    # Show what was detected
    warped = warp_quad(img, quad, out_size=800)
    print(f"\n[OK] Board detected, warped to 800x800")
    
    return True

def main():
    batch_003 = Path("data/raw/batch_003")
    
    # Test all images
    for img_path in sorted(batch_003.glob("*.png")):
        test_one_image(img_path)

if __name__ == "__main__":
    main()
