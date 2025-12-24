#!/usr/bin/env python3
"""Debug color thresholding for image 134747."""

import cv2
import numpy as np
from pathlib import Path

img_path = Path("data/raw/batch_003/Skjermbilde 2025-12-22 134747.png")
img = cv2.imread(str(img_path))

print(f"Image: {img_path.name}")
print(f"Shape: {img.shape}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"Mean brightness: {gray.mean():.1f}")
print(f"Min: {gray.min()}, Max: {gray.max()}")
print(f"85th percentile: {np.percentile(gray, 85):.1f}")

# Try different thresholds
for thresh_val in [160, 170, 180, 190, 200]:
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_area = img.shape[0] * img.shape[1]
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.10 * img_area or area > 0.98 * img_area:
            continue
        
        rect = cv2.minAreaRect(c)
        (_, _), (bw, bh), _ = rect
        if bw < 100 or bh < 100:
            continue
        
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect > 1.15:
            continue
        
        valid_contours.append((area, aspect))
    
    if valid_contours:
        print(f"Threshold {thresh_val}: {len(valid_contours)} valid contours")
        for area, aspect in sorted(valid_contours, key=lambda x: -x[0]):
            rel_area = area / img_area
            print(f"  - {rel_area:.1%} area, aspect={aspect:.2f}")
    else:
        print(f"Threshold {thresh_val}: no valid contours")
