import cv2
import sys
from pathlib import Path
from tkinter import Tk, filedialog

# Make repo imports work
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from scripts.detect_board import detect_board_quad_grid_aware, warp_quad

# File selection dialog
root = Tk()
root.withdraw()  # Hide the main window
root.attributes('-topmost', True)  # Bring dialog to front

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

img = cv2.imread(file_path)
assert img is not None, f"Failed to read image: {file_path}"

cv2.imshow("Original", img)
cv2.waitKey(1)

# Detect and warp
quad = detect_board_quad_grid_aware(img)
if quad is None:
    print("❌ Board not detected")
    cv2.waitKey(0)
    sys.exit(1)

# Draw detected quad
debug_img = img.copy()
cv2.drawContours(debug_img, [quad], -1, (0, 255, 0), 4)
cv2.imshow("Detected Board", debug_img)
cv2.waitKey(1)

# Warp the board
warped = warp_quad(img, quad, out_size=640)
cv2.imshow("Warped Board", warped)

print("✅ Board detected and warped. Press any key to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()

