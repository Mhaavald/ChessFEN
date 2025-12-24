"""
Batch labeling tool - select multiple tiles and apply the same label.

Shows a grid of tiles. Click to select/deselect tiles.
Press a label key to apply to all selected tiles.

Usage:
    python scripts/label_batch.py [in_dir] [out_dir]
"""

import cv2
import sys
import shutil
import numpy as np
from pathlib import Path
from tkinter import Tk, filedialog

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

# ============================
# Configuration
# ============================

GRID_COLS = 10  # tiles per row
TILE_DISPLAY_SIZE = 80  # pixels per tile in grid
TILES_PER_PAGE = 50  # how many tiles to show at once

KEY_TO_CLASS = {
    ord("1"): "empty",
    ord("P"): "wP", ord("N"): "wN", ord("B"): "wB",
    ord("R"): "wR", ord("Q"): "wQ", ord("K"): "wK",
    ord("p"): "bP", ord("n"): "bN", ord("b"): "bB",
    ord("r"): "bR", ord("q"): "bQ", ord("k"): "bK",
}

HELP_TEXT = """
=== Batch Labeling Tool ===

Click tiles to select/deselect (green border = selected)
Right-click to deselect all

Label keys:
  1 = empty
  P/N/B/R/Q/K = white pieces
  p/n/b/r/q/k = black pieces

Navigation:
  Space/Enter = next page
  Backspace   = prev page
  a           = select all on page
  c           = clear selection
  Esc         = quit
"""


# ============================
# Helpers
# ============================

def ensure_class_dirs(out_root: Path):
    for cls in set(KEY_TO_CLASS.values()):
        (out_root / cls).mkdir(parents=True, exist_ok=True)


def move_with_rename(src: Path, dst_dir: Path):
    dst = dst_dir / src.name
    counter = 1
    while dst.exists():
        dst = dst_dir / f"{src.stem}_dup{counter}{src.suffix}"
        counter += 1
    shutil.move(str(src), str(dst))
    return dst


# ============================
# Grid display
# ============================

class BatchLabeler:
    def __init__(self, images: list, out_dir: Path):
        self.images = images
        self.out_dir = out_dir
        self.page = 0
        self.selected = set()  # indices of selected tiles on current page
        self.history = []  # for undo
        
    def get_page_images(self):
        """Get images for current page."""
        start = self.page * TILES_PER_PAGE
        end = min(start + TILES_PER_PAGE, len(self.images))
        return self.images[start:end], start
    
    def render_grid(self):
        """Render the grid of tiles."""
        page_images, start_idx = self.get_page_images()
        n = len(page_images)
        
        if n == 0:
            return None
        
        rows = (n + GRID_COLS - 1) // GRID_COLS
        grid_h = rows * TILE_DISPLAY_SIZE
        grid_w = GRID_COLS * TILE_DISPLAY_SIZE
        
        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 50  # dark gray bg
        
        for i, img_path in enumerate(page_images):
            row = i // GRID_COLS
            col = i % GRID_COLS
            
            y1 = row * TILE_DISPLAY_SIZE
            x1 = col * TILE_DISPLAY_SIZE
            
            # Load and resize tile
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            tile = cv2.resize(img, (TILE_DISPLAY_SIZE - 4, TILE_DISPLAY_SIZE - 4))
            
            # Add border (green if selected, gray otherwise)
            if i in self.selected:
                border_color = (0, 255, 0)  # green
                thickness = 3
            else:
                border_color = (100, 100, 100)  # gray
                thickness = 1
            
            cv2.rectangle(tile, (0, 0), (tile.shape[1]-1, tile.shape[0]-1), 
                         border_color, thickness)
            
            # Place in grid
            grid[y1+2:y1+TILE_DISPLAY_SIZE-2, x1+2:x1+TILE_DISPLAY_SIZE-2] = tile
        
        # Add page info at bottom
        info = f"Page {self.page+1}/{(len(self.images)+TILES_PER_PAGE-1)//TILES_PER_PAGE} | " \
               f"Selected: {len(self.selected)} | Remaining: {len(self.images)}"
        cv2.putText(grid, info, (10, grid_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        return grid
    
    def click_handler(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Calculate which tile was clicked
            col = x // TILE_DISPLAY_SIZE
            row = y // TILE_DISPLAY_SIZE
            idx = row * GRID_COLS + col
            
            page_images, _ = self.get_page_images()
            if idx < len(page_images):
                if idx in self.selected:
                    self.selected.remove(idx)
                else:
                    self.selected.add(idx)
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click to deselect all
            self.selected.clear()
    
    def apply_label(self, cls: str):
        """Apply label to all selected tiles."""
        page_images, start_idx = self.get_page_images()
        
        labeled = []
        for i in sorted(self.selected, reverse=True):
            if i < len(page_images):
                img_path = page_images[i]
                dst = move_with_rename(img_path, self.out_dir / cls)
                self.history.append((img_path, dst))
                labeled.append(img_path.name)
        
        if labeled:
            print(f"Labeled {len(labeled)} tiles as '{cls}'")
            # Remove labeled images from list
            for i in sorted(self.selected, reverse=True):
                if i < len(page_images):
                    global_idx = start_idx + i
                    if global_idx < len(self.images):
                        self.images.pop(global_idx)
        
        self.selected.clear()
        
        # Adjust page if needed
        max_page = max(0, (len(self.images) - 1) // TILES_PER_PAGE)
        if self.page > max_page:
            self.page = max_page
    
    def run(self):
        """Main loop."""
        print(HELP_TEXT)
        ensure_class_dirs(self.out_dir)
        
        cv2.namedWindow("Batch Label")
        cv2.setMouseCallback("Batch Label", self.click_handler)
        
        while True:
            if not self.images:
                print("All tiles labeled!")
                break
            
            grid = self.render_grid()
            if grid is None:
                break
            
            cv2.imshow("Batch Label", grid)
            key = cv2.waitKey(50)
            
            if key == 27:  # ESC
                print("Exiting.")
                break
            
            elif key in (ord(" "), 13):  # Space or Enter - next page
                max_page = (len(self.images) - 1) // TILES_PER_PAGE
                if self.page < max_page:
                    self.page += 1
                    self.selected.clear()
            
            elif key == 8:  # Backspace - prev page
                if self.page > 0:
                    self.page -= 1
                    self.selected.clear()
            
            elif key == ord("a"):  # Select all
                page_images, _ = self.get_page_images()
                self.selected = set(range(len(page_images)))
            
            elif key == ord("c"):  # Clear selection
                self.selected.clear()
            
            elif key == ord("u"):  # Undo
                if self.history:
                    src, dst = self.history.pop()
                    shutil.move(str(dst), str(src))
                    self.images.append(src)
                    self.images.sort()
                    print(f"Undo: {src.name}")
            
            elif key in KEY_TO_CLASS:
                cls = KEY_TO_CLASS[key]
                self.apply_label(cls)
        
        cv2.destroyAllWindows()


# ============================
# CLI
# ============================

def main():
    if len(sys.argv) == 1:
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        print("Select input folder containing tiles to label...")
        in_dir_str = filedialog.askdirectory(
            title="Select input folder with tiles to label",
            initialdir=REPO_ROOT / "data" / "tiles"
        )
        
        if not in_dir_str:
            print("No input folder selected.")
            return
        
        out_dir_str = filedialog.askdirectory(
            title="Select output folder for labeled tiles",
            initialdir=REPO_ROOT / "data" / "labeled_squares"
        )
        
        if not out_dir_str:
            print("No output folder selected.")
            return
        
        in_dir = Path(in_dir_str)
        out_dir = Path(out_dir_str)
        
    elif len(sys.argv) == 3:
        in_dir = Path(sys.argv[1])
        out_dir = Path(sys.argv[2])
    else:
        print("Usage: python scripts/label_batch.py [in_dir] [out_dir]")
        return
    
    print(f"Input:  {in_dir}")
    print(f"Output: {out_dir}")
    
    # Collect images
    images = sorted([
        p for p in in_dir.iterdir() 
        if p.suffix.lower() in (".png", ".jpg")
    ])
    
    if not images:
        print("No images found.")
        return
    
    print(f"Found {len(images)} tiles to label")
    
    labeler = BatchLabeler(images, out_dir)
    labeler.run()


if __name__ == "__main__":
    main()
