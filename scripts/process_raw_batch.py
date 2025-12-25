"""
Process raw images: detect board, warp, and extract tiles for labeling.

Usage:
    python scripts/process_raw_batch.py data/raw/batch_006
"""

import sys
from pathlib import Path
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from scripts.detect_board import detect_board_quad_grid_aware, warp_quad, order_points, correct_board_orientation


def extract_tiles(warped: np.ndarray, tile_size: int = 64):
    """Split warped board into 64 tiles."""
    h, w = warped.shape[:2]
    cell_h = h // 8
    cell_w = w // 8
    
    tiles = []
    for row in range(8):
        for col in range(8):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            tile = warped[y1:y2, x1:x2]
            
            # Resize to standard size
            tile = cv2.resize(tile, (tile_size, tile_size))
            tiles.append(tile)
    
    return tiles


def process_raw_batch(raw_dir: Path, output_base: Path = None):
    """
    Process all raw images in a directory.
    
    Creates:
        - warped/ - warped board images
        - tiles/  - individual tiles ready for labeling
    """
    raw_dir = Path(raw_dir).resolve()
    
    if output_base is None:
        output_base = raw_dir
    
    warped_dir = output_base / "warped"
    tiles_dir = output_base / "tiles_unlabeled"
    warped_dir.mkdir(exist_ok=True)
    tiles_dir.mkdir(exist_ok=True)
    
    # Find all images
    images = sorted(
        p for p in raw_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        and not p.stem.endswith("_warped")
    )
    
    print(f"Found {len(images)} images in {raw_dir}")
    
    ok = 0
    fail = 0
    files = "abcdefgh"
    
    for img_path in images:
        print(f"\nProcessing: {img_path.name}")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ❌ Failed to read image")
            fail += 1
            continue
        
        # Detect board
        quad = detect_board_quad_grid_aware(img, crop_ratio=0.80)
        if quad is None:
            print(f"  ❌ Could not detect board")
            fail += 1
            continue
        
        # Warp board
        warped = warp_quad(img, quad, out_size=512)
        
        # Correct orientation (ensure h1 is light, a1 is dark)
        warped = correct_board_orientation(warped)
        print(f"  ✅ Board orientation corrected")
        
        # Save warped image
        warped_path = warped_dir / f"{img_path.stem}_warped.png"
        cv2.imwrite(str(warped_path), warped)
        print(f"  ✅ Saved warped: {warped_path.name}")
        
        # Extract tiles
        tiles = extract_tiles(warped, tile_size=64)
        
        # Save tiles
        for idx, tile in enumerate(tiles):
            row = idx // 8
            col = idx % 8
            rank = 8 - row
            file = files[col]
            square = f"{file}{rank}"
            
            tile_path = tiles_dir / f"{img_path.stem}_{square}.png"
            cv2.imwrite(str(tile_path), tile)
        
        print(f"  ✅ Extracted 64 tiles to tiles_unlabeled/")
        ok += 1
    
    print(f"\n{'='*50}")
    print(f"Done! Processed {ok} images, {fail} failed")
    print(f"\nWarped images: {warped_dir}")
    print(f"Tiles for labeling: {tiles_dir}")
    print(f"\nNext step: Label tiles using:")
    print(f"  python scripts/label_batch.py \"{tiles_dir}\" \"data/labeled_squares\"")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/process_raw_batch.py <raw_images_dir>")
        print("Example: python scripts/process_raw_batch.py data/raw/batch_006")
        sys.exit(1)
    
    raw_dir = Path(sys.argv[1])
    if not raw_dir.exists():
        print(f"Error: Directory not found: {raw_dir}")
        sys.exit(1)
    
    process_raw_batch(raw_dir)
