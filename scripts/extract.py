from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


CLASSES_13 = [
    "empty",
    "wP", "wN", "wB", "wR", "wQ", "wK",
    "bP", "bN", "bB", "bR", "bQ", "bK",
]


def square_name(row: int, col: int) -> str:
    """
    row, col are 0-based in image coordinates (row 0 = top).
    Returns algebraic like a8..h1.
    """
    file_ = chr(ord("a") + col)
    rank = 8 - row
    return f"{file_}{rank}"


def autocrop_inner_board(warped_bgr: np.ndarray, thresh: int = 70) -> np.ndarray:
    """
    Optional: remove a thin dark frame from a warped board.
    Designed to be conservative (should not eat real squares).
    """
    g = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(g, thresh, 255, cv2.THRESH_BINARY_INV)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return warped_bgr

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Conservative inward pad to remove border/frame pixels
    # Reduced from 0.015 to 0.008 (about 0.8% inward crop) to avoid cutting into pieces
    pad = max(1, int(0.008 * min(w, h)))
    x2, y2 = max(0, x + pad), max(0, y + pad)
    x3, y3 = min(warped_bgr.shape[1], x + w - pad), min(warped_bgr.shape[0], y + h - pad)

    # sanity: don't accept huge crop mistakes
    if (x3 - x2) < 0.7 * warped_bgr.shape[1] or (y3 - y2) < 0.7 * warped_bgr.shape[0]:
        return warped_bgr

    return warped_bgr[y2:y3, x2:x3]


def make_square(img: np.ndarray) -> np.ndarray:
    """
    Ensures the image is square by center-cropping.
    """
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


def split_board_to_squares(
    board_bgr: np.ndarray,
    do_autocrop: bool = False,
) -> Tuple[List[List[np.ndarray]], int]:
    """
    Split a warped board image into an 8x8 grid of square images.

    Returns:
      squares: squares[row][col] with row 0 = rank 8 (top), col 0 = file a (left)
      sq_size: pixel size of each square crop
    """
    img = board_bgr
    if do_autocrop:
        img = autocrop_inner_board(img)

    img = make_square(img)
    h, w = img.shape[:2]
    s = min(h, w)

    # For safety, resize to a size divisible by 8 (prevents uneven last row/col)
    s2 = (s // 8) * 8
    if s2 != s:
        img = cv2.resize(img, (s2, s2), interpolation=cv2.INTER_AREA)
        s = s2

    sq = s // 8
    squares: List[List[np.ndarray]] = []

    for r in range(8):
        row = []
        y0, y1 = r * sq, (r + 1) * sq
        for c in range(8):
            x0, x1 = c * sq, (c + 1) * sq
            tile = img[y0:y1, x0:x1].copy()
            row.append(tile)
        squares.append(row)

    return squares, sq


def write_squares(
    squares: List[List[np.ndarray]],
    out_dir: str | Path,
    prefix: str,
    ext: str = "png",
) -> None:
    """
    Writes each square to disk with filename: <prefix>_a8.png ... <prefix>_h1.png
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in range(8):
        for c in range(8):
            name = square_name(r, c)
            path = out_dir / f"{prefix}_{name}.{ext}"
            cv2.imwrite(str(path), squares[r][c])


def draw_grid_overlay(
    board_bgr: np.ndarray,
    sq_size: int,
) -> np.ndarray:
    """
    Draws an 8x8 grid + a8..h1 labels on top of a warped (square) board image.
    """
    img = board_bgr.copy()
    img = make_square(img)

    # resize to match sq_size exactly
    s2 = sq_size * 8
    if img.shape[0] != s2:
        img = cv2.resize(img, (s2, s2), interpolation=cv2.INTER_AREA)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # grid lines
    for i in range(9):
        p = i * sq_size
        cv2.line(img, (0, p), (s2, p), (0, 255, 0), 1)
        cv2.line(img, (p, 0), (p, s2), (0, 255, 0), 1)

    # square labels
    for r in range(8):
        for c in range(8):
            label = square_name(r, c)
            x = c * sq_size + 3
            y = r * sq_size + 14
            cv2.putText(img, label, (x, y), font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    return img


# ----------------------------
# CLI test usage
# ----------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.squares.extract <warped_board.png> <out_dir> [prefix]")
        raise SystemExit(1)

    board_path = sys.argv[1]
    out_dir = sys.argv[2]
    prefix = sys.argv[3] if len(sys.argv) >= 4 else Path(board_path).stem

    board = cv2.imread(board_path)
    if board is None:
        raise FileNotFoundError(board_path)

    squares, sq = split_board_to_squares(board, do_autocrop=True)
    write_squares(squares, out_dir, prefix)

    overlay = draw_grid_overlay(board, sq)
    overlay_path = Path(out_dir) / f"{prefix}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    print(f"Wrote 64 squares to: {out_dir}")
    print(f"Wrote overlay to: {overlay_path}")
