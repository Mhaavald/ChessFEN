#!/usr/bin/env python3
"""Detailed analysis of batch_003 detection issues."""

import cv2
import sys
from pathlib import Path

sys.path.append('.')
from scripts.detect_board import _candidate_quads_from_edges, combined_grid_score, warp_quad, center_crop

def analyze_image(img_path: Path, is_correct: bool):
    """Analyze detection candidates for an image."""
    print(f"\n{'='*70}")
    status = "[CORRECT]" if is_correct else "[WRONG]"
    print(f"{status} {img_path.name}")
    print('='*70)
    
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"ERROR: Could not load image")
        return
    
    h, w = img.shape[:2]
    print(f"Image size: {w}x{h} (aspect: {w/h:.3f})")
    
    # Test both tight and loose crops
    for crop_name, crop_ratio in [("TIGHT (0.95)", 0.95), ("LOOSE (0.80)", 0.80)]:
        print(f"\n--- {crop_name} ---")
        crop, (x0, y0) = center_crop(img, crop_ratio=crop_ratio)
        crop_h, crop_w = crop.shape[:2]
        crop_area = crop_h * crop_w
        print(f"Crop size: {crop_w}x{crop_h}, area: {crop_area}")
        
        candidates = _candidate_quads_from_edges(crop)
        print(f"Candidates found: {len(candidates)}")
        
        # Analyze top 5 candidates
        for i, quad in enumerate(candidates[:5]):
            quad_area = cv2.contourArea(quad)
            relative_area = quad_area / crop_area
            
            # Get bounding rect for shape info
            rect = cv2.minAreaRect(quad)
            (cx, cy), (bw, bh), angle = rect
            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            
            # Calculate grid score
            warped = warp_quad(crop, quad, out_size=512)
            grid_score = combined_grid_score(warped)
            
            # Apply the scoring logic
            if relative_area < 0.45:
                penalty = (0.45 - relative_area) * 2.0
                final_score = grid_score - penalty
            elif relative_area > 0.80:
                penalty = (relative_area - 0.80) * 1.5
                final_score = grid_score - penalty
            else:
                bonus = (relative_area - 0.35) * 0.5
                final_score = grid_score + bonus
            
            marker = " <-- BEST" if i == 0 else ""
            print(f"  [{i}] area={relative_area:5.1%}, aspect={aspect:.2f}, "
                  f"grid={grid_score:.3f}, final={final_score:.3f}{marker}")

def main():
    batch_003 = Path("data/raw/batch_003")
    
    # Image 3 is the only correct one
    correct_images = {"Skjermbilde 2025-12-22 134826.png"}
    
    for img_path in sorted(batch_003.glob("*.png")):
        is_correct = img_path.name in correct_images
        analyze_image(img_path, is_correct)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print('='*70)

if __name__ == "__main__":
    main()
