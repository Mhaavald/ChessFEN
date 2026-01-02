import sys
from pathlib import Path
import cv2
import os

# Make repo imports work when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from scripts.detect_board import detect_board_quad_grid_aware, warp_quad

# Check if running in debug mode
DEBUG_MODE = os.getenv('PYTHONDEBUG') or hasattr(sys, 'gettrace') and sys.gettrace() is not None


def warp_boards(
    in_dir: Path,
    out_dir: Path,
    out_size: int = 640,
    overwrite: bool = False,
    crop_ratio: float = 1.0,  # Use full image by default (no center crop)
    board_only: bool = False,  # If True, skip detection - images are already just the board
):
    in_dir = in_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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
        out_path = out_dir / f"{img_path.stem}_warped.png"

        if out_path.exists() and not overwrite:
            print(f"⏩ Skipping (exists): {img_path.name}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Failed to read: {img_path.name}")
            fail += 1
            continue

        # Display image only when debugging
        if DEBUG_MODE:
            cv2.imshow(f"Processing: {img_path.name}", img)
            cv2.waitKey(1)  # Brief pause to display

        if board_only:
            # Image is already the board - just resize to square
            warped = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_path), warped)
            print(f"✅ Resized (board only): {img_path.name} → {out_path.name}")
            ok += 1
            continue

        quad = detect_board_quad_grid_aware(img, crop_ratio=crop_ratio)
        if quad is None:
            print(f"❌ Board not detected: {img_path.name}")
            fail += 1
            continue

        warped = warp_quad(img, quad, out_size=out_size)
        cv2.imwrite(str(out_path), warped)

        print(f"✅ Warped: {img_path.name} → {out_path.name}")
        ok += 1

    if DEBUG_MODE:
        cv2.destroyAllWindows()
    print("\nDone.")
    print(f"  Success: {ok}")
    print(f"  Failed:  {fail}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Warp chess board images to square")
    parser.add_argument("in_dir", type=Path, help="Input directory with raw images")
    parser.add_argument("out_dir", type=Path, help="Output directory for warped boards")
    parser.add_argument("--size", type=int, default=640, help="Output size (default: 640)")
    parser.add_argument("--board-only", action="store_true", help="Skip detection - images are already just the board")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    
    args = parser.parse_args()
    
    if not args.in_dir.exists():
        raise FileNotFoundError(args.in_dir)
    
    warp_boards(args.in_dir, args.out_dir, out_size=args.size, 
                overwrite=args.overwrite, board_only=args.board_only)
