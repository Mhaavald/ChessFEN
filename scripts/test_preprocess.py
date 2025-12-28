#!/usr/bin/env python3
"""
Test script for image preprocessing.
Applies various enhancement techniques to chess board images.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


def auto_levels(img: np.ndarray) -> np.ndarray:
    """
    Auto-levels: stretches histogram so darkest becomes 0, lightest becomes 255.
    This is what the current JS implementation does.
    """
    result = img.copy()
    for i in range(3):  # For each channel
        channel = img[:, :, i]
        min_val = channel.min()
        max_val = channel.max()
        if max_val > min_val:
            result[:, :, i] = ((channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return result


def increase_contrast(img: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """
    Increase contrast by multiplying difference from mean.
    factor > 1 increases contrast, factor < 1 decreases.
    """
    mean = img.mean()
    result = mean + (img.astype(float) - mean) * factor
    return np.clip(result, 0, 255).astype(np.uint8)


def adjust_levels(img: np.ndarray, black_point: int = 20, white_point: int = 235) -> np.ndarray:
    """
    Adjust black and white points.
    Pixels below black_point become 0, above white_point become 255.
    """
    result = img.astype(float)
    result = (result - black_point) / (white_point - black_point) * 255
    return np.clip(result, 0, 255).astype(np.uint8)


def sharpen(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """
    Sharpen image using unsharp masking.
    """
    # Create gaussian blur
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    # Unsharp mask: original + amount * (original - blurred)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return sharpened


def enhance_blacks(img: np.ndarray, strength: float = 0.2) -> np.ndarray:
    """
    Make dark areas darker (increase black).
    Uses a curve that compresses shadows.
    """
    # Create lookup table
    lut = np.arange(256, dtype=np.float32)
    # Apply curve: darker values get pushed more towards black
    lut = lut - strength * (255 - lut) * (lut / 255) * ((255 - lut) / 255)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    
    # Apply to each channel
    if len(img.shape) == 3:
        return cv2.LUT(img, lut)
    return cv2.LUT(img, lut)


def reduce_whites(img: np.ndarray, strength: float = 0.2) -> np.ndarray:
    """
    Make bright areas slightly less bright (reduce white).
    Uses a curve that compresses highlights.
    """
    # Create lookup table
    lut = np.arange(256, dtype=np.float32)
    # Apply curve: brighter values get pushed slightly down
    lut = lut - strength * lut * (lut / 255) * (lut / 255)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    
    if len(img.shape) == 3:
        return cv2.LUT(img, lut)
    return cv2.LUT(img, lut)


def clahe_enhance(img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Good for images with varying lighting.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    
    # Merge and convert back
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def full_preprocess(img: np.ndarray, 
                    contrast: float = 1.3,
                    sharpen_amount: float = 0.5,
                    black_enhance: float = 0.15,
                    white_reduce: float = 0.1,
                    use_clahe: bool = False) -> np.ndarray:
    """
    Full preprocessing pipeline:
    1. Auto-levels (stretch histogram)
    2. Increase contrast
    3. Sharpen
    4. Enhance blacks (make darker)
    5. Reduce whites (make less bright)
    """
    result = img.copy()
    
    # Step 1: Auto-levels
    result = auto_levels(result)
    
    # Step 2: CLAHE or simple contrast
    if use_clahe:
        result = clahe_enhance(result)
    else:
        result = increase_contrast(result, contrast)
    
    # Step 3: Sharpen
    if sharpen_amount > 0:
        result = sharpen(result, sharpen_amount)
    
    # Step 4: Enhance blacks
    if black_enhance > 0:
        result = enhance_blacks(result, black_enhance)
    
    # Step 5: Reduce whites
    if white_reduce > 0:
        result = reduce_whites(result, white_reduce)
    
    return result


def create_comparison(original: np.ndarray, processed: np.ndarray, label: str) -> np.ndarray:
    """Create side-by-side comparison with labels."""
    h, w = original.shape[:2]
    
    # Create combined image
    combined = np.zeros((h + 30, w * 2 + 10, 3), dtype=np.uint8)
    combined[:] = (40, 40, 40)  # Dark gray background
    
    # Add images
    combined[30:30+h, 0:w] = original
    combined[30:30+h, w+10:w*2+10] = processed
    
    # Add labels
    cv2.putText(combined, "Original", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(combined, label, (w + 20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return combined


def main():
    parser = argparse.ArgumentParser(description='Test image preprocessing for chess board images')
    parser.add_argument('input', help='Input image path or directory')
    parser.add_argument('--output', '-o', default='debug_out/preprocess_test', help='Output directory')
    parser.add_argument('--contrast', '-c', type=float, default=1.3, help='Contrast factor (default: 1.3)')
    parser.add_argument('--sharpen', '-s', type=float, default=0.5, help='Sharpen amount (default: 0.5)')
    parser.add_argument('--black', '-b', type=float, default=0.15, help='Black enhancement (default: 0.15)')
    parser.add_argument('--white', '-w', type=float, default=0.1, help='White reduction (default: 0.1)')
    parser.add_argument('--clahe', action='store_true', help='Use CLAHE instead of simple contrast')
    parser.add_argument('--all-steps', action='store_true', help='Show all intermediate steps')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get image files
    if input_path.is_dir():
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
    else:
        images = [input_path]
    
    print(f"Processing {len(images)} image(s)...")
    print(f"Settings: contrast={args.contrast}, sharpen={args.sharpen}, black={args.black}, white={args.white}")
    print(f"Output: {output_dir.absolute()}")
    print()
    
    for img_path in images:
        print(f"  Processing: {img_path.name}")
        
        original = cv2.imread(str(img_path))
        if original is None:
            print(f"    ERROR: Could not load image")
            continue
        
        # Resize for processing if too large
        max_dim = 512
        h, w = original.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            original = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        if args.all_steps:
            # Show each step separately
            steps = []
            
            # Step 1: Auto-levels
            step1 = auto_levels(original)
            steps.append(create_comparison(original, step1, "1. Auto-levels"))
            
            # Step 2: Contrast
            if args.clahe:
                step2 = clahe_enhance(step1)
                steps.append(create_comparison(step1, step2, "2. CLAHE"))
            else:
                step2 = increase_contrast(step1, args.contrast)
                steps.append(create_comparison(step1, step2, f"2. Contrast x{args.contrast}"))
            
            # Step 3: Sharpen
            step3 = sharpen(step2, args.sharpen)
            steps.append(create_comparison(step2, step3, f"3. Sharpen {args.sharpen}"))
            
            # Step 4: Black enhancement
            step4 = enhance_blacks(step3, args.black)
            steps.append(create_comparison(step3, step4, f"4. Black +{args.black}"))
            
            # Step 5: White reduction
            step5 = reduce_whites(step4, args.white)
            steps.append(create_comparison(step4, step5, f"5. White -{args.white}"))
            
            # Stack all steps vertically
            combined = np.vstack(steps)
            cv2.imwrite(str(output_dir / f"{img_path.stem}_steps.png"), combined)
            
            # Also save final result
            cv2.imwrite(str(output_dir / f"{img_path.stem}_final.png"), step5)
            
        else:
            # Just show before/after
            processed = full_preprocess(
                original,
                contrast=args.contrast,
                sharpen_amount=args.sharpen,
                black_enhance=args.black,
                white_reduce=args.white,
                use_clahe=args.clahe
            )
            
            comparison = create_comparison(original, processed, "Processed")
            cv2.imwrite(str(output_dir / f"{img_path.stem}_comparison.png"), comparison)
            cv2.imwrite(str(output_dir / f"{img_path.stem}_processed.png"), processed)
    
    print()
    print(f"Done! Output saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
