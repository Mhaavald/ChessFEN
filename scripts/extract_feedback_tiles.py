#!/usr/bin/env python3
"""
Extract tiles from user feedback images for labeling and retraining.

This script:
1. Reads feedback JSON files that contain corrections
2. Extracts all 64 tiles from the associated board image
3. Uses the corrected FEN to determine the correct label for each tile
4. Saves tiles to the labeled_squares folder structure

Usage:
    python scripts/extract_feedback_tiles.py [--feedback-dir DIR] [--output-dir DIR] [--dry-run]
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path


# FEN to piece mapping
FEN_TO_PIECE = {
    'r': 'bR', 'n': 'bN', 'b': 'bB', 'q': 'bQ', 'k': 'bK', 'p': 'bP',
    'R': 'wR', 'N': 'wN', 'B': 'wB', 'Q': 'wQ', 'K': 'wK', 'P': 'wP',
    '.': 'empty'
}


def convert_to_grayscale(img_bgr):
    """Convert image to grayscale with preprocessing to match training data."""
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Auto-levels (histogram stretch)
    min_val = gray.min()
    max_val = gray.max()
    if max_val > min_val:
        gray = ((gray.astype(np.float32) - min_val) / (max_val - min_val) * 255.0)
    
    # Contrast enhancement (1.3x around midpoint)
    contrast_factor = 1.3
    gray = 128 + (gray - 128) * contrast_factor
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    
    # Return as 3-channel BGR for consistency
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def fen_to_board(fen: str) -> list:
    """Convert FEN string to 8x8 board array."""
    board = []
    ranks = fen.split()[0].split('/')
    
    for rank in ranks:
        row = []
        for char in rank:
            if char.isdigit():
                row.extend(['.'] * int(char))
            else:
                row.append(char)
        board.append(row)
    
    return board


def extract_tiles_from_feedback(feedback_path: Path, output_dir: Path, dry_run: bool = False):
    """Extract all tiles from a feedback image using corrected FEN labels."""
    
    # Load feedback JSON
    with open(feedback_path) as f:
        feedback = json.load(f)
    
    feedback_id = feedback['id']
    corrected_fen = feedback['corrected_fen']
    corrected_squares = feedback.get('corrected_squares', {})
    
    # Load corresponding image
    image_path = feedback_path.with_suffix('.png')
    if not image_path.exists():
        print(f"  WARNING: No image file found for {feedback_id}")
        return 0, 0
    
    warped = cv2.imread(str(image_path))
    if warped is None:
        print(f"  ERROR: Could not load {image_path}")
        return 0, 0
    
    # Determine tile size from image
    h, w = warped.shape[:2]
    tile_size = w // 8
    
    # Parse the corrected FEN
    board = fen_to_board(corrected_fen)
    
    saved = 0
    skipped = 0
    
    # Extract all 64 tiles
    for row in range(8):
        for col in range(8):
            fen_char = board[row][col]
            piece = FEN_TO_PIECE.get(fen_char, 'empty')
            
            # Get square name (a1-h8)
            square = chr(ord('a') + col) + str(8 - row)
            
            # Extract tile
            y1 = row * tile_size
            y2 = (row + 1) * tile_size
            x1 = col * tile_size
            x2 = (col + 1) * tile_size
            tile = warped[y1:y2, x1:x2]
            
            # Convert to grayscale to match training data
            tile = convert_to_grayscale(tile)
            
            # Determine output path
            piece_dir = output_dir / piece
            tile_name = f"feedback_{feedback_id}_{square}.png"
            tile_path = piece_dir / tile_name
            
            # Check if already exists
            if tile_path.exists():
                skipped += 1
                continue
            
            if dry_run:
                is_correction = square in corrected_squares
                marker = " [CORRECTED]" if is_correction else ""
                print(f"    Would save: {piece}/{tile_name}{marker}")
                saved += 1
            else:
                piece_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(tile_path), tile)
                saved += 1
    
    return saved, skipped


def main():
    parser = argparse.ArgumentParser(description='Extract tiles from feedback for retraining')
    parser.add_argument('--feedback-dir', type=Path, default=Path('data/user_feedback'),
                        help='Directory containing feedback JSON files')
    parser.add_argument('--output-dir', type=Path, default=Path('data/labeled_squares'),
                        help='Output directory for labeled tiles')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without saving')
    parser.add_argument('--only-corrections', action='store_true',
                        help='Only extract tiles that were corrected by user')
    args = parser.parse_args()
    
    if not args.feedback_dir.exists():
        print(f"Feedback directory not found: {args.feedback_dir}")
        return
    
    # Find all feedback JSON files
    feedback_files = sorted(args.feedback_dir.glob('*.json'))
    
    if not feedback_files:
        print("No feedback files found")
        return
    
    print(f"Found {len(feedback_files)} feedback files")
    print(f"Output directory: {args.output_dir}")
    if args.dry_run:
        print("DRY RUN - no files will be saved")
    print()
    
    total_saved = 0
    total_skipped = 0
    
    for feedback_path in feedback_files:
        print(f"Processing {feedback_path.name}...")
        
        # Load to show corrections
        with open(feedback_path) as f:
            feedback = json.load(f)
        
        corrections = feedback.get('corrected_squares', {})
        if corrections:
            print(f"  Corrections: {corrections}")
        
        saved, skipped = extract_tiles_from_feedback(
            feedback_path, args.output_dir, args.dry_run
        )
        
        total_saved += saved
        total_skipped += skipped
        print(f"  Saved: {saved}, Skipped (existing): {skipped}")
    
    print()
    print(f"Total: {total_saved} tiles saved, {total_skipped} skipped")
    
    if not args.dry_run and total_saved > 0:
        print()
        print("Next steps:")
        print("1. Review the extracted tiles for quality")
        print("2. Retrain the model with: python scripts/train_classifier.py")


if __name__ == '__main__':
    main()
