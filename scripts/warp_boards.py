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

        quad = detect_board_quad_grid_aware(img)
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
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python scripts/warp_boards.py <in_dir> <out_dir> [out_size]")
        print("")
        print("Example:")
        print("  python scripts/warp_boards.py data/raw/batch_001 data/boards_warped/batch_001")
        raise SystemExit(1)

    in_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_size = int(sys.argv[3]) if len(sys.argv) >= 4 else 640

    if not in_dir.exists():
        raise FileNotFoundError(in_dir)

    warp_boards(in_dir, out_dir, out_size=out_size)
