"""
Apply labels by copying tiles from to_label subfolder structure to labeled_squares.

Usage:
    python scripts/apply_labels.py data/to_label/batch_003

This expects the folder structure:
    data/to_label/batch_003/
        empty/
            img001_a5.png
            img001_b4.png
            ...
        wP/
            img001_d4.png
            ...
        bN/
            img001_e4.png
            ...
        ... other class folders ...

Tiles will be copied to data/labeled_squares/<class>/ with unique names.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
from pathlib import Path
from datetime import datetime


CLASSES = ['empty', 'wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 
           'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/apply_labels.py <labeled_folder>")
        print("Example: python scripts/apply_labels.py data/to_label/batch_003")
        return
    
    source_dir = Path(sys.argv[1])
    if not source_dir.exists():
        print(f"ERROR: Folder not found: {source_dir}")
        return
    
    dest_dir = Path("data/labeled_squares")
    batch_name = source_dir.name
    
    print(f"Applying labels from: {source_dir}")
    print(f"Destination: {dest_dir}")
    
    # Count existing files per class
    existing_counts = {}
    for cls in CLASSES:
        cls_dir = dest_dir / cls
        if cls_dir.exists():
            existing_counts[cls] = len(list(cls_dir.glob("*.png")))
        else:
            existing_counts[cls] = 0
    
    # Process each class folder
    total_copied = 0
    stats = {}
    
    for cls in CLASSES:
        cls_source = source_dir / cls
        if not cls_source.exists():
            continue
        
        cls_dest = dest_dir / cls
        cls_dest.mkdir(parents=True, exist_ok=True)
        
        # Find all tiles in this class
        tiles = list(cls_source.glob("*.png"))
        if not tiles:
            continue
        
        # Copy with unique names
        count = 0
        for tile in tiles:
            # Create unique name: batch_003_img001_a5.png
            new_name = f"{batch_name}_{tile.name}"
            dest_path = cls_dest / new_name
            
            # Handle duplicates
            if dest_path.exists():
                base = dest_path.stem
                suffix = 1
                while dest_path.exists():
                    dest_path = cls_dest / f"{base}_{suffix}.png"
                    suffix += 1
            
            shutil.copy2(tile, dest_path)
            count += 1
        
        stats[cls] = count
        total_copied += count
        print(f"  {cls}: {count} tiles (was {existing_counts[cls]}, now {existing_counts[cls] + count})")
    
    print(f"\n{'='*60}")
    print(f"LABELING APPLIED")
    print(f"{'='*60}")
    print(f"  Total tiles copied: {total_copied}")
    print(f"  From: {source_dir}")
    print(f"  To: {dest_dir}")
    
    # Check for unlabeled tiles (still in root)
    root_tiles = list(source_dir.glob("img*.png"))
    board_refs = [t for t in root_tiles if "_board" in t.name]
    unlabeled = [t for t in root_tiles if "_board" not in t.name]
    
    if unlabeled:
        print(f"\n  WARNING: {len(unlabeled)} unlabeled tiles remaining in root folder!")
        print(f"  These tiles were NOT copied.")
        for t in unlabeled[:5]:
            print(f"    - {t.name}")
        if len(unlabeled) > 5:
            print(f"    ... and {len(unlabeled) - 5} more")


if __name__ == "__main__":
    main()
