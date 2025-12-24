import sys
from pathlib import Path
import cv2
from tkinter import Tk, filedialog

# Make repo imports work
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from scripts.detect_board import detect_board_quad_grid_aware, warp_quad
from scripts.extract import split_board_to_squares, draw_grid_overlay


def review_overlays(in_dir: Path, out_dir: Path = None):
    """
    Generate and display overlays for all images in a folder.
    Press any key to move to next image, ESC to quit.
    """
    in_dir = in_dir.resolve()
    
    if out_dir:
        out_dir = out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    
    images = sorted(
        p for p in in_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    
    if not images:
        print(f"No images found in {in_dir}")
        return
    
    print(f"Found {len(images)} images")
    print("Press any key to advance, ESC to quit")
    
    for idx, img_path in enumerate(images):
        print(f"\n[{idx+1}/{len(images)}] Processing: {img_path.name}")
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ❌ Failed to read")
            continue
        
        # Detect board
        quad = detect_board_quad_grid_aware(img)
        if quad is None:
            print(f"  ❌ Board not detected")
            cv2.imshow("Review - Board not detected", img)
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                break
            continue
        
        # Warp board
        warped = warp_quad(img, quad, out_size=640)
        
        # Extract squares and create overlay
        try:
            squares, sq_size = split_board_to_squares(warped, do_autocrop=True)
            overlay = draw_grid_overlay(warped, sq_size)
            
            # Optionally save overlay
            if out_dir:
                overlay_path = out_dir / f"{img_path.stem}_overlay.png"
                cv2.imwrite(str(overlay_path), overlay)
                print(f"  ✅ Saved: {overlay_path.name}")
            else:
                print(f"  ✅ Generated overlay")
            
            # Show original with detection and overlay
            debug_img = img.copy()
            cv2.drawContours(debug_img, [quad], -1, (0, 255, 0), 4)
            
            # Resize for display if too large
            max_h = 800
            if debug_img.shape[0] > max_h:
                scale = max_h / debug_img.shape[0]
                new_w = int(debug_img.shape[1] * scale)
                debug_img = cv2.resize(debug_img, (new_w, max_h))
            
            cv2.imshow("Review - Original with Detection", debug_img)
            cv2.imshow("Review - Grid Overlay", overlay)
            
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                print("\nQuitting...")
                break
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            cv2.imshow("Review - Error", warped)
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                break
    
    cv2.destroyAllWindows()
    print("\nDone reviewing.")


if __name__ == "__main__":
    # If no arguments, show folder selection dialog
    if len(sys.argv) == 1:
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        print("Select folder containing chess board images...")
        in_dir_str = filedialog.askdirectory(
            title="Select folder with chess board images",
            initialdir=REPO_ROOT / "data" / "raw"
        )
        
        if not in_dir_str:
            print("No folder selected. Exiting.")
            sys.exit(0)
        
        in_dir = Path(in_dir_str)
        
        # Ask if user wants to save overlays
        print("\nDo you want to save overlays? Press 'y' for yes, any other key for no...")
        root2 = Tk()
        root2.withdraw()
        
        response = input("Save overlays? (y/n): ").lower()
        out_dir = None
        
        if response == 'y':
            out_dir_str = filedialog.askdirectory(
                title="Select output folder for overlays",
                initialdir=REPO_ROOT / "debug_out"
            )
            if out_dir_str:
                out_dir = Path(out_dir_str)
        
        review_overlays(in_dir, out_dir)
        
    elif len(sys.argv) >= 2:
        in_dir = Path(sys.argv[1])
        out_dir = Path(sys.argv[2]) if len(sys.argv) >= 3 else None
        
        if not in_dir.exists():
            raise FileNotFoundError(in_dir)
        
        review_overlays(in_dir, out_dir)
        
    else:
        print("Usage: python scripts/review_overlays.py [in_dir] [out_dir]")
        print("  If no arguments provided, folder selection dialogs will open.")
        print("")
        print("Example:")
        print("  python scripts/review_overlays.py data/raw/batch_001")
        print("  python scripts/review_overlays.py data/raw/batch_001 debug_out/overlays")
        raise SystemExit(1)
