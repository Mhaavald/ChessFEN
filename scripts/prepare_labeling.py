"""
Extract tiles from images for manual labeling.

Usage:
    python scripts/prepare_labeling.py data/raw/batch_003

This will:
1. Process all images in the batch folder
2. Extract 64 tiles per image
3. Save tiles to data/to_label/batch_003/ with unique names
4. Create a labeling guide showing the predicted positions

After labeling, run:
    python scripts/apply_labels.py data/to_label/batch_003

To move labeled tiles to data/labeled_squares/
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from pathlib import Path
import json

from src.inference.inference_service import (
    detect_board_quad_grid_aware, warp_quad, correct_board_orientation
)
from scripts.detect_board import expand_quad_to_edges, refine_quad_by_tile_uniformity, refine_quad_for_digital_board
from scripts.extract import split_board_to_squares, square_name


def is_digital_board(img_bgr: np.ndarray) -> bool:
    """
    Detect if the image is a digital chess screenshot vs physical board photo.
    
    Digital boards typically have:
    - Sharp edges with dark/black borders
    - Image aspect ratio close to 1:1 or wider
    - Uniform borders (black or solid color)
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Digital screenshots are typically wider than tall or square
    # Physical board photos from phones are often taller than wide (portrait)
    aspect = w / h
    if aspect > 1.2:  # Wider than tall - likely digital screenshot
        return True
    
    # Check for very dark left/right edges (common in digital chess apps)
    left_edge = gray[:, 0:20].mean()
    right_edge = gray[:, -20:].mean()
    
    if left_edge < 30 or right_edge < 30:  # Very dark edges
        return True
    
    # If image is small (under 1000px), likely a screenshot not a photo
    if max(h, w) < 1000:
        return True
    
    return False


def process_image(image_path: Path, output_dir: Path, image_index: int):
    """Extract tiles from a single image."""
    
    print(f"\nProcessing: {image_path.name}")
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"  ERROR: Could not load image")
        return None
    
    # Detect board
    quad = detect_board_quad_grid_aware(img_bgr)
    if quad is None:
        print(f"  ERROR: Could not detect board")
        return None
    
    # Check if digital or physical board and use appropriate refinement
    if is_digital_board(img_bgr):
        print(f"  Detected as DIGITAL board")
        quad_refined = refine_quad_for_digital_board(img_bgr, quad)
    else:
        print(f"  Detected as PHYSICAL board")
        quad_expanded = expand_quad_to_edges(img_bgr, quad, max_expand=50)
        quad_refined = refine_quad_by_tile_uniformity(img_bgr, quad_expanded, max_adjust=15, step=2)
    
    # Warp and orient
    warped_raw = warp_quad(img_bgr, quad_refined, out_size=640)
    warped = correct_board_orientation(warped_raw)
    
    # Extract tiles
    squares, sq_size = split_board_to_squares(warped, do_autocrop=True)
    
    # Save tiles with unique names
    tiles_saved = []
    prefix = f"img{image_index:03d}"
    
    for row in range(8):
        for col in range(8):
            sq_name = square_name(row, col)
            tile = squares[row][col]
            
            # Unique filename: img001_a8.png
            filename = f"{prefix}_{sq_name}.png"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), tile)
            tiles_saved.append(filename)
    
    # Also save the warped board with grid for reference
    from scripts.extract import draw_grid_overlay
    grid_img = draw_grid_overlay(warped, sq_size)
    grid_path = output_dir / f"{prefix}_board.png"
    cv2.imwrite(str(grid_path), grid_img)
    
    print(f"  Saved {len(tiles_saved)} tiles + board reference")
    return {
        'image': image_path.name,
        'prefix': prefix,
        'tiles': tiles_saved
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/prepare_labeling.py <batch_folder>")
        print("Example: python scripts/prepare_labeling.py data/raw/batch_003")
        return
    
    batch_path = Path(sys.argv[1])
    if not batch_path.exists():
        print(f"ERROR: Folder not found: {batch_path}")
        return
    
    # Get batch name
    batch_name = batch_path.name
    
    # Create output directory
    output_dir = Path("data/to_label") / batch_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting tiles from: {batch_path}")
    print(f"Output directory: {output_dir}")
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [f for f in batch_path.iterdir() 
              if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not images:
        print("No images found!")
        return
    
    print(f"Found {len(images)} images")
    
    # Process each image
    results = []
    for i, img_path in enumerate(sorted(images)):
        result = process_image(img_path, output_dir, i + 1)
        if result:
            results.append(result)
    
    # Save manifest
    manifest = {
        'batch': batch_name,
        'source': str(batch_path),
        'images': results,
        'total_tiles': sum(len(r['tiles']) for r in results),
        'classes': ['empty', 'wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 
                   'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']
    }
    
    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total images: {len(results)}")
    print(f"  Total tiles: {manifest['total_tiles']}")
    print(f"  Output: {output_dir}")
    print()
    print("LABELING INSTRUCTIONS:")
    print("-" * 40)
    print("1. Open the output folder in a file browser")
    print("2. Create subfolders for each piece class:")
    print("   empty, wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK")
    print("3. Move each tile image to the correct class folder")
    print("4. Use the *_board.png files as reference")
    print()
    print("After labeling, run:")
    print(f"  python scripts/apply_labels.py {output_dir}")


if __name__ == "__main__":
    main()
