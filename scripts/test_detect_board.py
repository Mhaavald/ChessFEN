import cv2
import sys
from pathlib import Path
from tkinter import Tk, filedialog

# Adjust import if your src layout differs
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from scripts.detect_board import (
    detect_board_quad_grid_aware,
    warp_quad,
)

def main(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    quad = detect_board_quad_grid_aware(img)
    if quad is None:
        print("❌ Board not detected")
        return

    # Draw detected quad
    debug = img.copy()
    cv2.drawContours(debug, [quad], -1, (0, 255, 0), 4)

    # Warp board
    warped = warp_quad(img, quad, out_size=640)

    # Show results
    cv2.imshow("Original (with detected board)", debug)
    cv2.imshow("Warped board", warped)
    print("✅ Board detected — press any key to close windows")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # If no argument provided, show file dialog
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
        main(file_path)
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Usage: python scripts/test_detect_board.py [image_path]")
        print("  If no image_path provided, a file dialog will open.")
        raise SystemExit(1)
