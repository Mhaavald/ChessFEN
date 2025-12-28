"""
Interactive Tile Labeling Tool

Shows a grid of tiles and lets you click to label them.
Click on a tile, then press a key to assign the label:
  - 0-9: empty (0), wK(1), wQ(2), wR(3), wB(4), wN(5), wP(6)
  - Shift+1-6: bK, bQ, bR, bB, bN, bP
  - Or click the piece buttons on the right side

Usage:
    python scripts/label_interactive.py data/to_label/batch_006
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
import sys

# Piece codes and display
PIECES = ['empty', 'wK', 'wQ', 'wR', 'wB', 'wN', 'wP', 'bK', 'bQ', 'bR', 'bB', 'bN', 'bP']
PIECE_SYMBOLS = {
    'empty': '·', 
    'wK': '♔', 'wQ': '♕', 'wR': '♖', 'wB': '♗', 'wN': '♘', 'wP': '♙',
    'bK': '♚', 'bQ': '♛', 'bR': '♜', 'bB': '♝', 'bN': '♞', 'bP': '♟'
}

# Keyboard shortcuts - intuitive: lowercase=black, UPPERCASE=white, 1=empty
KEY_MAP = {
    ord('1'): 'empty', ord(' '): 'empty', ord('.'): 'empty',
    # White pieces (SHIFT + letter)
    ord('K'): 'wK', ord('Q'): 'wQ', ord('R'): 'wR', ord('B'): 'wB', ord('N'): 'wN', ord('P'): 'wP',
    # Black pieces (lowercase)
    ord('k'): 'bK', ord('q'): 'bQ', ord('r'): 'bR', ord('b'): 'bB', ord('n'): 'bN', ord('p'): 'bP',
}


class InteractiveLabeler:
    def __init__(self, label_dir: Path):
        self.label_dir = Path(label_dir)
        self.unlabeled_dir = self.label_dir / '_unlabeled'
        
        # Load unlabeled tiles
        self.tiles = sorted(self.unlabeled_dir.glob('*.png'))
        self.labels = {}  # filename -> piece
        self.undo_stack = []  # for undo: list of (filename, old_label or None)
        
        # Display settings
        self.tile_size = 80
        self.grid_cols = 8
        self.grid_rows = 5
        self.tiles_per_page = self.grid_cols * self.grid_rows
        self.current_page = 0
        self.selected_tile = None
        
        # Button panel
        self.panel_width = 200
        self.button_height = 40
        
        print(f"Loaded {len(self.tiles)} unlabeled tiles")
    
    def get_page_tiles(self):
        start = self.current_page * self.tiles_per_page
        end = start + self.tiles_per_page
        return self.tiles[start:end]
    
    def total_pages(self):
        return (len(self.tiles) + self.tiles_per_page - 1) // self.tiles_per_page
    
    def draw_interface(self):
        # Calculate dimensions
        grid_width = self.grid_cols * self.tile_size
        grid_height = self.grid_rows * self.tile_size
        total_width = grid_width + self.panel_width
        total_height = grid_height + 60  # Extra for status bar
        
        img = np.ones((total_height, total_width, 3), dtype=np.uint8) * 50
        
        # Draw tiles
        page_tiles = self.get_page_tiles()
        for i, tile_path in enumerate(page_tiles):
            row = i // self.grid_cols
            col = i % self.grid_cols
            
            x = col * self.tile_size
            y = row * self.tile_size
            
            tile = cv2.imread(str(tile_path))
            if tile is not None:
                tile = cv2.resize(tile, (self.tile_size, self.tile_size))
                img[y:y+self.tile_size, x:x+self.tile_size] = tile
            
            # Draw border
            color = (0, 255, 0) if self.selected_tile == i else (100, 100, 100)
            thickness = 3 if self.selected_tile == i else 1
            cv2.rectangle(img, (x, y), (x+self.tile_size-1, y+self.tile_size-1), color, thickness)
            
            # Show label if assigned
            if tile_path.name in self.labels:
                label = self.labels[tile_path.name]
                symbol = PIECE_SYMBOLS.get(label, label)
                # Draw label background
                cv2.rectangle(img, (x, y), (x+25, y+20), (0, 100, 0), -1)
                cv2.putText(img, label[:2], (x+2, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw piece buttons panel
        panel_x = grid_width + 10
        for i, piece in enumerate(PIECES):
            btn_y = 10 + i * (self.button_height + 5)
            btn_color = (80, 80, 80)
            cv2.rectangle(img, (panel_x, btn_y), (panel_x + self.panel_width - 20, btn_y + self.button_height), btn_color, -1)
            cv2.rectangle(img, (panel_x, btn_y), (panel_x + self.panel_width - 20, btn_y + self.button_height), (150, 150, 150), 1)
            
            # Piece name and shortcut
            shortcut = '0/SPC' if piece == 'empty' else str(i) if i <= 6 else f'q-y'
            text = f"{piece} ({PIECE_SYMBOLS[piece]})"
            cv2.putText(img, text, (panel_x + 5, btn_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Status bar
        status_y = grid_height + 10
        labeled = len(self.labels)
        total = len(self.tiles)
        page_info = f"Page {self.current_page + 1}/{self.total_pages()}"
        status = f"Labeled: {labeled}/{total} ({100*labeled/total:.0f}%)  |  {page_info}  |  [A/D] Prev/Next  [Z] Undo  [S] Save  [ESC] Quit"
        cv2.putText(img, status, (10, status_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Instructions
        cv2.putText(img, "Click tile, then: 1=empty, P/R/N/B/Q/K=white, p/r/n/b/q/k=black  [Z]=undo", 
                    (10, status_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        return img
    
    def handle_click(self, x, y):
        grid_width = self.grid_cols * self.tile_size
        
        if x < grid_width:
            # Click on tile grid
            col = x // self.tile_size
            row = y // self.tile_size
            idx = row * self.grid_cols + col
            page_tiles = self.get_page_tiles()
            if idx < len(page_tiles):
                self.selected_tile = idx
        else:
            # Click on button panel
            panel_x = grid_width + 10
            for i, piece in enumerate(PIECES):
                btn_y = 10 + i * (self.button_height + 5)
                if panel_x <= x <= panel_x + self.panel_width - 20:
                    if btn_y <= y <= btn_y + self.button_height:
                        self.label_selected(piece)
                        break
    
    def label_selected(self, piece):
        if self.selected_tile is None:
            return
        
        page_tiles = self.get_page_tiles()
        if self.selected_tile < len(page_tiles):
            tile_path = page_tiles[self.selected_tile]
            # Save to undo stack
            old_label = self.labels.get(tile_path.name, None)
            self.undo_stack.append((tile_path.name, old_label, self.selected_tile, self.current_page))
            self.labels[tile_path.name] = piece
            print(f"Labeled {tile_path.name} as {piece}")
            
            # Auto-advance to next tile
            self.selected_tile += 1
            if self.selected_tile >= len(page_tiles):
                if self.current_page < self.total_pages() - 1:
                    self.current_page += 1
                    self.selected_tile = 0
                else:
                    self.selected_tile = None
    
    def undo(self):
        """Undo the last label action"""
        if not self.undo_stack:
            print("Nothing to undo")
            return
        
        filename, old_label, old_selected, old_page = self.undo_stack.pop()
        
        if old_label is None:
            # Remove the label
            if filename in self.labels:
                del self.labels[filename]
        else:
            # Restore old label
            self.labels[filename] = old_label
        
        # Restore position
        self.current_page = old_page
        self.selected_tile = old_selected
        print(f"Undo: {filename} -> {old_label if old_label else '(unlabeled)'}")
    
    def save_labels(self):
        """Move labeled tiles to their piece folders"""
        count = 0
        for filename, piece in self.labels.items():
            src = self.unlabeled_dir / filename
            dst_dir = self.label_dir / piece
            dst_dir.mkdir(exist_ok=True)
            dst = dst_dir / filename
            
            if src.exists():
                shutil.move(str(src), str(dst))
                count += 1
        
        print(f"Saved {count} labeled tiles")
        self.labels.clear()
        
        # Reload remaining tiles
        self.tiles = sorted(self.unlabeled_dir.glob('*.png'))
        self.current_page = 0
        self.selected_tile = None
        
        return count
    
    def run(self):
        cv2.namedWindow('Tile Labeler', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Tile Labeler', lambda event, x, y, flags, param: 
                             self.handle_click(x, y) if event == cv2.EVENT_LBUTTONDOWN else None)
        
        print("\nControls:")
        print("  Click tile to select, then:")
        print("  1 or SPACE or . = empty")
        print("  P R N B Q K (uppercase) = white pieces")
        print("  p r n b q k (lowercase) = black pieces")
        print("  Z = undo last label")
        print("  A/D = prev/next page")
        print("  S = save and move labeled tiles")
        print("  ESC = quit")
        print()
        
        while True:
            img = self.draw_interface()
            cv2.imshow('Tile Labeler', img)
            
            key = cv2.waitKey(50) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('z') or key == ord('Z'):  # Undo
                self.undo()
            elif key == ord('a'):  # Prev page
                if self.current_page > 0:
                    self.current_page -= 1
                    self.selected_tile = None
            elif key == ord('d'):  # Next page
                if self.current_page < self.total_pages() - 1:
                    self.current_page += 1
                    self.selected_tile = None
            elif key == ord('s'):  # Save
                self.save_labels()
            elif key in KEY_MAP:
                self.label_selected(KEY_MAP[key])
        
        cv2.destroyAllWindows()
        
        # Final save prompt
        if self.labels:
            print(f"\n{len(self.labels)} labels not saved. Saving now...")
            self.save_labels()


def main():
    if len(sys.argv) < 2:
        print("Usage: python label_interactive.py <label_dir>")
        print("Example: python label_interactive.py data/to_label/batch_006")
        sys.exit(1)
    
    label_dir = Path(sys.argv[1])
    if not label_dir.exists():
        print(f"Error: {label_dir} not found")
        sys.exit(1)
    
    labeler = InteractiveLabeler(label_dir)
    labeler.run()


if __name__ == "__main__":
    main()
