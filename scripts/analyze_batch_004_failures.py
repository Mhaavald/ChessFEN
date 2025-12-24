#!/usr/bin/env python3
"""Analyze batch_004 failed images to understand why detection failed."""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append('.')
from scripts.detect_board import detect_board_quad_grid_aware

def analyze_image(img_path: Path):
    """Analyze why detection might be failing."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not load: {img_path.name}")
        return
    
    print(f"\n{'='*70}")
    print(f"Image: {img_path.name}")
    print('='*70)
    
    h, w = img.shape[:2]
    print(f"Size: {w}x{h}")
    
    # Analyze brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Brightness: mean={gray.mean():.1f}, min={gray.min()}, max={gray.max()}")
    print(f"Percentiles: 25th={np.percentile(gray, 25):.1f}, 50th={np.percentile(gray, 50):.1f}, 75th={np.percentile(gray, 75):.1f}")
    
    # Try detection with debug
    print("\nDetection attempt:")
    quad = detect_board_quad_grid_aware(img, debug=True)
    
    if quad is None:
        print("❌ FAILED")
    else:
        print("✓ SUCCESS")

def main():
    batch_004 = Path("data/raw/batch_004")
    
    # Analyze the first 5 failed images
    failed_images = [
        "20251222_154316323_iOS.jpg",
        "20251222_154340338_iOS.jpg",
        "20251222_154346707_iOS.jpg",
        "20251222_154350261_iOS.jpg",
        "20251222_154358811_iOS.jpg",
    ]
    
    for img_name in failed_images:
        img_path = batch_004 / img_name
        if img_path.exists():
            analyze_image(img_path)

if __name__ == "__main__":
    main()
