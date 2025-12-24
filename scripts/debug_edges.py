#!/usr/bin/env python3
"""Debug edge detection to see what contours are found."""

import cv2
import sys
import numpy as np
from pathlib import Path

sys.path.append('.')

def show_edge_detection(img_path: Path, out_dir: Path):
    """Show edge detection process for debugging."""
    img = cv2.imread(str(img_path))
    if img is None:
        return
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {img_path.name}")
    print('='*70)
    
    # Try both crops
    for crop_name, crop_ratio in [("TIGHT", 0.95), ("LOOSE", 0.80)]:
        print(f"\n--- {crop_name} CROP ({crop_ratio}) ---")
        
        h, w = img.shape[:2]
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        y0 = (h - crop_h) // 2
        x0 = (w - crop_w) // 2
        crop = img[y0:y0+crop_h, x0:x0+crop_w].copy()
        
        print(f"Crop size: {crop.shape[1]}x{crop.shape[0]}")
        
        # Edge detection (same as in detect_board.py)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try multiple Canny threshold combinations
        for low, high in [(30, 120), (20, 100), (40, 150)]:
            edges = cv2.Canny(gray, low, high)
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            crop_area = crop.shape[0] * crop.shape[1]
            
            # Filter contours
            valid_quads = []
            for c in contours:
                area = cv2.contourArea(c)
                if area < 0.003 * crop_area or area > 0.7 * crop_area:
                    continue
                
                rect = cv2.minAreaRect(c)
                (_, _), (bw, bh), _ = rect
                if bw < 150 or bh < 150:
                    continue
                
                aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
                if aspect > 1.8:
                    continue
                
                box = cv2.boxPoints(rect).astype(np.int32).reshape(-1, 1, 2)
                valid_quads.append((area, box, aspect))
            
            print(f"  Canny({low},{high}): {len(valid_quads)} valid quads", end="")
            if valid_quads:
                valid_quads.sort(key=lambda t: t[0], reverse=True)
                best_area, _, best_aspect = valid_quads[0]
                rel_area = best_area / crop_area
                print(f" - Best: {rel_area:.1%} area, aspect={best_aspect:.2f}")
            else:
                print()
        
        # Save edge visualization with default (30, 120)
        edges = cv2.Canny(gray, 30, 120)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        
        edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        base_name = img_path.stem
        out_path = out_dir / f"{base_name}_{crop_name.lower()}_edges.png"
        cv2.imwrite(str(out_path), edges_vis)

def main():
    batch_003 = Path("data/raw/batch_003")
    out_dir = Path("debug_out/edge_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in sorted(batch_003.glob("*.png")):
        show_edge_detection(img_path, out_dir)
    
    print(f"\n{'='*70}")
    print(f"Edge visualizations saved to: {out_dir}")

if __name__ == "__main__":
    main()
