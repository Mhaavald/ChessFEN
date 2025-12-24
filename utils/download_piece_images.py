from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import requests

COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# Map your piece code -> Commons categories that contain diagram/icon-like files.
# You can add more categories over time to increase variety.
PIECE_TO_CATEGORIES = {
    # White pieces (symbols/icons)
    "wK": ["Category:White_chess_king_symbols", "Category:Chess_piece_icons"],
    "wQ": ["Category:White_chess_queen_symbols", "Category:Chess_piece_icons"],
    "wR": ["Category:White_chess_rook_symbols", "Category:Chess_piece_icons"],
    "wB": ["Category:White_chess_bishop_symbols", "Category:Chess_piece_icons"],
    "wN": ["Category:White_chess_knight_symbols", "Category:Chess_piece_icons"],
    "wP": ["Category:White_chess_pawn_symbols", "Category:Chess_piece_icons"],

    # Black pieces (symbols/icons)
    "bK": ["Category:Black_chess_king_symbols", "Category:Chess_piece_icons"],
    "bQ": ["Category:Black_chess_queen_symbols", "Category:Chess_piece_icons"],
    "bR": ["Category:Black_chess_rook_symbols", "Category:Chess_piece_icons"],
    "bB": ["Category:Black_chess_bishop_symbols", "Category:Chess_piece_icons"],
    "bN": ["Category:Black_chess_knight_symbols", "Category:Chess_piece_icons"],
    "bP": ["Category:Black_chess_pawn_symbols", "Category:Chess_piece_icons"],
}

# Some icon categories include both colors in filenames. We'll filter by piece code.
# This is a pragmatic filter for Category:Chess_piece_icons.
ICON_FILTERS = {
    "wK": re.compile(r"(king.*white|white.*king)", re.I),
    "wQ": re.compile(r"(queen.*white|white.*queen)", re.I),
    "wR": re.compile(r"(rook.*white|white.*rook)", re.I),
    "wB": re.compile(r"(bishop.*white|white.*bishop)", re.I),
    "wN": re.compile(r"(knight.*white|white.*knight)", re.I),
    "wP": re.compile(r"(pawn.*white|white.*pawn)", re.I),

    "bK": re.compile(r"(king.*black|black.*king)", re.I),
    "bQ": re.compile(r"(queen.*black|black.*queen)", re.I),
    "bR": re.compile(r"(rook.*black|black.*rook)", re.I),
    "bB": re.compile(r"(bishop.*black|black.*bishop)", re.I),
    "bN": re.compile(r"(knight.*black|black.*knight)", re.I),
    "bP": re.compile(r"(pawn.*black|black.*pawn)", re.I),
}

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def commons_get(params: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(COMMONS_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if r.status_code == 403 and attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise
    return {}


def list_category_files(category: str, limit: int = 200) -> List[str]:
    """
    Returns a list of file page titles like 'File:Chess glt45.svg'
    """
    titles: List[str] = []
    cmcontinue = None

    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category,
            "cmnamespace": 6,  # File namespace
            "cmlimit": "max",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        data = commons_get(params)
        cms = data.get("query", {}).get("categorymembers", [])
        for item in cms:
            t = item.get("title")
            if t and t.startswith("File:"):
                titles.append(t)

        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue or len(titles) >= limit:
            break

    return titles[:limit]


def get_file_download_url(file_title: str, png_width: int = 256) -> Tuple[str, str]:
    """
    Returns (url, ext). For SVG we request a PNG thumbnail at png_width.
    For PNG/JPG we request the original URL.
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "titles": file_title,
        "iiprop": "url|mime",
    }

    data = commons_get(params)
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))
    info = (page.get("imageinfo") or [{}])[0]

    url = info.get("url")
    mime = info.get("mime", "")

    if not url:
        raise RuntimeError(f"No URL for {file_title}")

    # If SVG, use thumbnail PNG rendering
    if "svg" in mime or url.lower().endswith(".svg"):
        params2 = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "titles": file_title,
            "iiprop": "url",
            "iiurlwidth": str(png_width),
        }
        data2 = commons_get(params2)
        pages2 = data2.get("query", {}).get("pages", {})
        page2 = next(iter(pages2.values()))
        info2 = (page2.get("imageinfo") or [{}])[0]
        thumb = info2.get("thumburl")
        if not thumb:
            raise RuntimeError(f"No thumburl for {file_title}")
        return thumb, "png"

    # otherwise download original
    ext = os.path.splitext(url)[1].lstrip(".").lower()
    if ext not in ("png", "jpg", "jpeg", "webp"):
        # still download; but normalize ext to 'bin'
        ext = "bin"
    return url, ext


def sanitize_filename(s: str) -> str:
    s = s.replace("File:", "")
    s = re.sub(r"[^\w\-.() ]+", "_", s)
    return s.strip().replace(" ", "_")


def download_file(url: str, out_path: Path) -> None:
    r = requests.get(url, stream=True, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piece", required=True, help="Piece code: wQ,wK,wR,wB,wN,wP,bQ,bK,bR,bB,bN,bP")
    ap.add_argument("--n", type=int, default=50, help="Number of images to download")
    ap.add_argument("--out", default="data/piece_images", help="Output directory")
    ap.add_argument("--png_width", type=int, default=256, help="PNG width for SVG thumbnails")
    ap.add_argument("--sleep", type=float, default=0.2, help="Delay between downloads (be nice)")
    args = ap.parse_args()

    piece = args.piece
    if piece not in PIECE_TO_CATEGORIES:
        raise SystemExit(f"Unknown piece '{piece}'. Valid: {sorted(PIECE_TO_CATEGORIES.keys())}")

    out_dir = Path(args.out) / piece
    out_dir.mkdir(parents=True, exist_ok=True)

    titles: List[str] = []
    for cat in PIECE_TO_CATEGORIES[piece]:
        titles.extend(list_category_files(cat, limit=500))

    # De-dupe while preserving order
    seen = set()
    titles = [t for t in titles if not (t in seen or seen.add(t))]

    # If using the mixed icon category, filter to relevant color/piece
    filt = ICON_FILTERS.get(piece)
    filtered: List[str] = []
    for t in titles:
        if "Category:Chess_piece_icons" in PIECE_TO_CATEGORIES[piece]:
            # apply filter only when filename is icon-like (heuristic)
            if "Farm-Fresh" in t or "icon" in t.lower() or "chess" in t.lower():
                if filt and not filt.search(t):
                    continue
        filtered.append(t)
    titles = filtered

    downloaded = 0
    for t in titles:
        if downloaded >= args.n:
            break

        try:
            url, ext = get_file_download_url(t, png_width=args.png_width)
            name = sanitize_filename(t)
            out_path = out_dir / f"{downloaded:03d}_{name}.{ext}"

            if out_path.exists():
                continue

            download_file(url, out_path)
            downloaded += 1
            print(f"[{downloaded}/{args.n}] {t} -> {out_path.name}")
            time.sleep(args.sleep)

        except Exception as e:
            print(f"Skipping {t}: {e}")

    print(f"\nDone. Downloaded {downloaded} images to: {out_dir}")


if __name__ == "__main__":
    main()
