import sys
from pathlib import Path
import cv2
from tkinter import Tk, filedialog

# Make repo imports work when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from scripts.detect_board import detect_board_quad_grid_aware, warp_quad
from scripts.extract import split_board_to_squares, write_squares, draw_grid_overlay


def main(
    image_path: str,
    out_dir: str = "debug_out",
    prefix: str = "test",
):
    image_path = str(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    # --- 1) detect board quad ---
    quad = detect_board_quad_grid_aware(img)
    if quad is None:
        print("❌ Board not detected")
        return

    # --- 2) debug image with detected quad ---
    debug_detect = img.copy()
    cv2.drawContours(debug_detect, [quad], -1, (0, 255, 0), 4)
    detect_path = out_dir / f"{prefix}_detect.png"
    cv2.imwrite(str(detect_path), debug_detect)

    # --- 3) warp ---
    warped = warp_quad(img, quad, out_size=640)
    warped_path = out_dir / f"{prefix}_warped.png"
    cv2.imwrite(str(warped_path), warped)

    # --- 4) extract squares from warped ---
    squares, sq = split_board_to_squares(warped, do_autocrop=True)
    tiles_dir = out_dir / f"{prefix}_tiles"
    write_squares(squares, tiles_dir, prefix)

    # --- 5) overlay for verification ---
    overlay = draw_grid_overlay(warped, sq)
    overlay_path = out_dir / f"{prefix}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    print("✅ Done")
    print(f"  Detected outline: {detect_path}")
    print(f"  Warped board:     {warped_path}")
    print(f"  Tiles folder:     {tiles_dir}")
    print(f"  Overlay:          {overlay_path}")

    # Optional: show results interactively
    cv2.imshow("Detected board", debug_detect)
    cv2.imshow("Warped board", warped)
    cv2.imshow("Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # If no arguments provided, show file selection dialog
    if len(sys.argv) == 1:
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        file_path = filedialog.askopenfilename(
            title="Select a chess board image",
            initialdir=REPO_ROOT / "data" / "raw",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            print("No file selected. Exiting.")
            sys.exit(0)
        
        print(f"Selected: {file_path}")
        raw_photo = file_path
        out_dir = "debug_out"
        prefix = Path(raw_photo).stem
        
        main(raw_photo, out_dir, prefix)
        
    elif len(sys.argv) >= 2:
        raw_photo = sys.argv[1]
        out_dir = sys.argv[2] if len(sys.argv) >= 3 else "debug_out"
        prefix = sys.argv[3] if len(sys.argv) >= 4 else Path(raw_photo).stem
        
        main(raw_photo, out_dir, prefix)
        
    else:
        print("Usage: python scripts/test_warp_and_extract.py [raw_photo_path] [out_dir] [prefix]")
        print("  If no arguments provided, a file selection dialog will open.")
        raise SystemExit(1)
