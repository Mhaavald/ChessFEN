"""
Process user feedback corrections and extract squares for retraining.
Reads from data/user_feedback/ and adds corrected squares to data/labeled_squares/
"""

import sys
from pathlib import Path
import json
import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

FEEDBACK_DIR = REPO_ROOT / "data" / "user_feedback"
OUTPUT_DIR = REPO_ROOT / "data" / "labeled_squares"

# Square names in order (a8 to h1, row by row from top)
def get_square_name(row: int, col: int) -> str:
    """Convert row/col (0-indexed from top-left) to chess notation."""
    file = chr(ord('a') + col)
    rank = 8 - row
    return f"{file}{rank}"

def get_row_col(square: str) -> tuple[int, int]:
    """Convert chess notation to row/col (0-indexed from top-left)."""
    file = square[0]
    rank = int(square[1])
    col = ord(file) - ord('a')
    row = 8 - rank
    return row, col

def extract_square(board_img: np.ndarray, row: int, col: int, size: int = 80) -> np.ndarray:
    """Extract a single square from a board image."""
    h, w = board_img.shape[:2]
    sq_h, sq_w = h // 8, w // 8
    
    y1 = row * sq_h
    y2 = y1 + sq_h
    x1 = col * sq_w
    x2 = x1 + sq_w
    
    square = board_img[y1:y2, x1:x2]
    
    # Resize to standard size
    if square.shape[0] != size or square.shape[1] != size:
        square = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    
    return square


def process_feedback():
    """Process all user feedback and extract corrected squares."""
    
    if not FEEDBACK_DIR.exists():
        print(f"No feedback directory found at {FEEDBACK_DIR}")
        return
    
    # Get all feedback files
    feedback_files = list(FEEDBACK_DIR.glob("*.json"))
    print(f"Found {len(feedback_files)} feedback submissions")
    
    if not feedback_files:
        print("No feedback to process")
        return
    
    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_squares = 0
    processed_feedback = 0
    
    for json_path in feedback_files:
        # Load feedback data
        with open(json_path, 'r') as f:
            feedback = json.load(f)
        
        feedback_id = feedback.get('id', json_path.stem)
        corrected_squares = feedback.get('corrected_squares', {})
        
        if not corrected_squares:
            print(f"  {feedback_id}: No corrections, skipping")
            continue
        
        # Load corresponding image
        img_path = json_path.with_suffix('.png')
        if not img_path.exists():
            print(f"  {feedback_id}: Image not found, skipping")
            continue
        
        board_img = cv2.imread(str(img_path))
        if board_img is None:
            print(f"  {feedback_id}: Failed to load image, skipping")
            continue
        
        # Extract each corrected square
        for square_name, piece in corrected_squares.items():
            # Skip invalid entries (like "cleared": "all")
            if len(square_name) != 2 or not square_name[0].isalpha() or not square_name[1].isdigit():
                continue
            
            row, col = get_row_col(square_name)
            square_img = extract_square(board_img, row, col)
            
            # Ensure piece directory exists
            piece_dir = OUTPUT_DIR / piece
            piece_dir.mkdir(parents=True, exist_ok=True)
            
            # Save with unique name
            output_name = f"feedback_{feedback_id}_{square_name}.png"
            output_path = piece_dir / output_name
            
            cv2.imwrite(str(output_path), square_img)
            print(f"  {feedback_id}: {square_name} -> {piece}")
            total_squares += 1
        
        processed_feedback += 1
    
    print()
    print(f"Processed {processed_feedback} feedback submissions")
    print(f"Extracted {total_squares} corrected squares")
    print(f"Output saved to: {OUTPUT_DIR}")


def show_stats():
    """Show current labeled squares statistics."""
    print("\nCurrent labeled squares:")
    print("-" * 30)
    
    total = 0
    for piece_dir in sorted(OUTPUT_DIR.iterdir()):
        if piece_dir.is_dir():
            count = len(list(piece_dir.glob("*.png"))) + len(list(piece_dir.glob("*.jpg")))
            feedback_count = len(list(piece_dir.glob("feedback_*.png")))
            total += count
            print(f"  {piece_dir.name:6}: {count:5} ({feedback_count} from feedback)")
    
    print("-" * 30)
    print(f"  Total: {total}")


if __name__ == "__main__":
    print("=== Processing User Feedback ===\n")
    process_feedback()
    show_stats()
