import sys
from pathlib import Path
import cv2

# Make repo imports work
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from scripts.extract import split_board_to_squares, write_squares, draw_grid_overlay


def extract_batch(
    in_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
    save_overlays: bool = True,
):
    """
    Extract squares from all warped board images in a directory.
    
    Args:
        in_dir: Directory containing warped board images
        out_dir: Base directory for output (will create subdirectories per board)
        overwrite: If False, skip boards that already have extracted squares
        save_overlays: If True, save grid overlay visualizations
    """
    in_dir = in_dir.resolve()
    out_dir = out_dir.resolve()
    
    images = sorted(
        p for p in in_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    
    if not images:
        print(f"No images found in {in_dir}")
        return
    
    ok = 0
    fail = 0
    
    for img_path in images:
        board_name = img_path.stem.replace("_warped", "")
        board_out_dir = out_dir / board_name
        
        # Check if already processed
        if not overwrite and board_out_dir.exists():
            tiles = list(board_out_dir.glob(f"{board_name}_*.png"))
            if len(tiles) >= 64:
                print(f"⏩ Skipping (exists): {img_path.name}")
                continue
        
        # Load warped board
        board = cv2.imread(str(img_path))
        if board is None:
            print(f"❌ Failed to read: {img_path.name}")
            fail += 1
            continue
        
        try:
            # Extract squares
            squares, sq_size = split_board_to_squares(board, do_autocrop=True)
            
            # Write all 64 squares
            write_squares(squares, board_out_dir, board_name)
            
            # Optionally save overlay
            if save_overlays:
                overlay = draw_grid_overlay(board, sq_size)
                overlay_path = board_out_dir / f"{board_name}_overlay.png"
                cv2.imwrite(str(overlay_path), overlay)
            
            print(f"✅ Extracted: {img_path.name} → {board_out_dir.name}/ (64 squares)")
            ok += 1
            
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")
            fail += 1
    
    print("\nDone.")
    print(f"  Success: {ok}")
    print(f"  Failed:  {fail}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python scripts/extract_batch.py <warped_boards_dir> <output_dir>")
        print("")
        print("Example:")
        print("  python scripts/extract_batch.py data/boards_warped/batch_001 data/squares/batch_001")
        raise SystemExit(1)
    
    in_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    
    if not in_dir.exists():
        raise FileNotFoundError(in_dir)
    
    extract_batch(in_dir, out_dir)
