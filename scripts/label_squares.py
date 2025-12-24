import cv2
import sys
import shutil
from pathlib import Path
from tkinter import Tk, filedialog

# Make repo imports work
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))


# ============================
# Configuration
# ============================

# Label mapping (keys -> class names)
KEY_TO_CLASS = {
    ord("1"): "empty",

    ord("P"): "wP",
    ord("N"): "wN",
    ord("B"): "wB",
    ord("R"): "wR",
    ord("Q"): "wQ",
    ord("K"): "wK",

    ord("p"): "bP",
    ord("n"): "bN",
    ord("b"): "bB",
    ord("r"): "bR",
    ord("q"): "bQ",
    ord("k"): "bK",
}

HELP_TEXT = """
Label keys:
  1  -> empty

  White:  P N B R Q K
  Black:  p n b r q k

Other keys:
  u  -> undo
  s  -> skip square
  Esc -> quit
"""


# ============================
# Helpers
# ============================

def ensure_class_dirs(out_root: Path):
    for cls in set(KEY_TO_CLASS.values()):
        (out_root / cls).mkdir(parents=True, exist_ok=True)


def move_with_rename(src: Path, dst_dir: Path):
    dst = dst_dir / src.name
    if dst.exists():
        dst = dst_dir / f"{src.stem}_dup{src.suffix}"
    shutil.move(str(src), str(dst))
    return dst


# ============================
# Main labeling loop
# ============================

def label_folder(
    in_dir: Path,
    out_dir: Path,
):
    in_dir = in_dir.resolve()
    out_dir = out_dir.resolve()

    ensure_class_dirs(out_dir)

    # Collect images (expects *_a8.png .. *_h1.png)
    images = sorted(
        [p for p in in_dir.iterdir() if p.suffix.lower() in (".png", ".jpg") and "_" in p.stem]
    )

    if not images:
        print("No square images found.")
        return

    history = []  # for undo
    idx = 0

    print(HELP_TEXT)

    while idx < len(images):
        img_path = images[idx]

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read {img_path}")
            idx += 1
            continue

        # Enlarge for visibility
        disp = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Label square", disp)
        cv2.setWindowTitle("Label square", f"{img_path.name}  [{idx+1}/{len(images)}]")

        key = cv2.waitKey(0)

        # ESC → quit
        if key == 27:
            print("Exiting.")
            break

        # Undo
        if key == ord("u"):
            if history:
                last_src, last_dst = history.pop()
                shutil.move(str(last_dst), str(last_src))
                print(f"Undo: {last_src.name}")
                idx -= 1
            continue

        # Skip
        if key == ord("s"):
            idx += 1
            continue

        # Label
        if key in KEY_TO_CLASS:
            cls = KEY_TO_CLASS[key]
            dst = move_with_rename(img_path, out_dir / cls)
            history.append((img_path, dst))
            print(f"Labeled {img_path.name} -> {cls}")
            idx += 1
            continue

        print("Unknown key.")

    cv2.destroyAllWindows()


# ============================
# CLI
# ============================

if __name__ == "__main__":
    # If no arguments provided, show folder selection dialogs
    if len(sys.argv) == 1:
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        print("Select input folder containing squares to label...")
        in_dir_str = filedialog.askdirectory(
            title="Select input folder with squares to label",
            initialdir=REPO_ROOT / "data" / "squares"
        )
        
        if not in_dir_str:
            print("No input folder selected. Exiting.")
            sys.exit(0)
        
        print("Select output folder for labeled squares...")
        out_dir_str = filedialog.askdirectory(
            title="Select output folder for labeled squares",
            initialdir=REPO_ROOT / "data" / "squares"
        )
        
        if not out_dir_str:
            print("No output folder selected. Exiting.")
            sys.exit(0)
        
        in_dir = Path(in_dir_str)
        out_dir = Path(out_dir_str)
        
        print(f"Input:  {in_dir}")
        print(f"Output: {out_dir}")
        
        label_folder(in_dir, out_dir)
        
    elif len(sys.argv) == 3:
        in_dir = Path(sys.argv[1])
        out_dir = Path(sys.argv[2])
        
        if not in_dir.exists():
            raise FileNotFoundError(in_dir)
        
        label_folder(in_dir, out_dir)
        
    else:
        print("Usage: python scripts/label_squares.py [in_dir] [out_dir]")
        print("  If no arguments provided, folder selection dialogs will open.")
        print("")
        print("Example:")
        print("  python scripts/label_squares.py data/squares/to_label data/squares/train")
        raise SystemExit(1)
