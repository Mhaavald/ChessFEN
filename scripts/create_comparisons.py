#!/usr/bin/env python3
"""Visual comparison - show original + detected region side by side."""

import cv2
import sys
import numpy as np
from pathlib import Path

sys.path.append('.')
from scripts.detect_board import detect_board_quad_grid_aware, warp_quad

def create_comparison(img_path: Path, out_path: Path):
    """Create a comparison image showing original and warped."""
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    
    quad = detect_board_quad_grid_aware(img, debug=False)
    if quad is None:
        return False
    
    # Draw quad on original
    img_with_quad = img.copy()
    cv2.polylines(img_with_quad, [quad], True, (0, 255, 0), 3)
    
    # Warp the board
    warped = warp_quad(img, quad, out_size=400)
    
    # Resize original to same height as warped for side-by-side
    h_target = 400
    h_orig, w_orig = img_with_quad.shape[:2]
    w_target = int(w_orig * h_target / h_orig)
    resized_orig = cv2.resize(img_with_quad, (w_target, h_target))
    
    # Create side-by-side
    comparison = np.hstack([resized_orig, warped])
    
    # Add labels
    cv2.putText(comparison, "Original + Detection", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(comparison, "Warped Board", (w_target + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imwrite(str(out_path), comparison)
    return True

def main():
    batch_003 = Path("data/raw/batch_003")
    out_dir = Path("debug_out/batch_003_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating comparison images...")
    
    for img_path in sorted(batch_003.glob("*.png")):
        out_path = out_dir / f"{img_path.stem}_comparison.png"
        if create_comparison(img_path, out_path):
            print(f"  Created: {out_path.name}")
        else:
            print(f"  FAILED: {img_path.name}")
    
    print(f"\nComparisons saved to: {out_dir}")

if __name__ == "__main__":
    main()
