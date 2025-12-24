#!/usr/bin/env python3
"""Test batch_003 images with verbose debug output."""

import cv2
import sys
from pathlib import Path

sys.path.append('.')
from scripts.detect_board import detect_board_quad_grid_aware

def test_image(img_path: Path):
    print(f"\n{'='*60}")
    print(f"Testing: {img_path.name}")
    print('='*60)
    
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"ERROR: Could not load image")
        return
    
    quad = detect_board_quad_grid_aware(img, debug=True)
    
    if quad is None:
        print("\n[FAILED] Detection did not find board")
    else:
        print("\n[OK] Detection successful")
    
    return quad is not None

def main():
    batch_003 = Path("data/raw/batch_003")
    images = sorted(batch_003.glob("*.png"))
    
    results = []
    for img_path in images:
        success = test_image(img_path)
        results.append((img_path.name, success))
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for name, success in results:
        status = "[OK]" if success else "[FAILED]"
        print(f"{status} - {name}")
    
    success_count = sum(1 for _, s in results if s)
    print(f"\nTotal: {success_count}/{len(results)} successful")

if __name__ == "__main__":
    main()
