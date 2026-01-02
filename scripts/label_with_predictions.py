"""
Interactive labeling tool with model-predicted suggestions.

Shows each tile with the model's prediction and confidence.
Press Enter to accept, type a new label to override, or 's' to skip.
"""

import sys
from pathlib import Path
import cv2
import torch
import numpy as np

# Make repo imports work
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.inference.inference_service import load_model, preprocess_image, CLASSES

# Shorthand mappings (chess notation style)
# Lowercase = black, Uppercase = white
SHORTHAND = {
    'p': 'bP', 'n': 'bN', 'b': 'bB', 'r': 'bR', 'q': 'bQ', 'k': 'bK',  # black
    'P': 'wP', 'N': 'wN', 'B': 'wB', 'R': 'wR', 'Q': 'wQ', 'K': 'wK',  # white
    'e': 'empty', 'E': 'empty', '.': 'empty', '0': 'empty',            # empty
}


def predict_tile(model, tile_path: Path) -> tuple[str, float, list[tuple[str, float]]]:
    """
    Predict the piece on a tile.
    
    Returns:
        (predicted_class, confidence, all_probs)
    """
    img = cv2.imread(str(tile_path))
    if img is None:
        return None, 0.0, []
    
    # Preprocess
    img = preprocess_image(img)
    
    # Convert to tensor
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    
    # Normalize (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    # Resize to 80x80
    img_tensor = torch.nn.functional.interpolate(
        img_tensor.unsqueeze(0), size=(80, 80), mode='bilinear', align_corners=False
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    
    # Get all probabilities
    all_probs = [(CLASSES[i], probs[i].item()) for i in range(len(CLASSES))]
    all_probs.sort(key=lambda x: x[1], reverse=True)
    
    pred_idx = probs.argmax().item()
    pred_class = CLASSES[pred_idx]
    confidence = probs[pred_idx].item()
    
    return pred_class, confidence, all_probs


def show_tile_opencv(tile_path: Path, prediction: str, confidence: float, all_probs: list):
    """Show tile in an OpenCV window with prediction info."""
    img = cv2.imread(str(tile_path))
    if img is None:
        return
    
    # Scale up for visibility
    scale = 4
    h, w = img.shape[:2]
    display = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    
    # Add info panel on the right
    panel_width = 300
    panel = np.ones((h * scale, panel_width, 3), dtype=np.uint8) * 40
    
    # Draw prediction info
    y = 30
    cv2.putText(panel, f"Prediction: {prediction}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y += 30
    cv2.putText(panel, f"Confidence: {confidence:.1%}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y += 40
    
    cv2.putText(panel, "Top predictions:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y += 25
    
    for cls, prob in all_probs[:5]:
        color = (0, 255, 0) if cls == prediction else (150, 150, 150)
        cv2.putText(panel, f"  {cls}: {prob:.1%}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 22
    
    y += 20
    cv2.putText(panel, "Keys:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
    y += 22
    cv2.putText(panel, "  Enter: Accept", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    y += 20
    cv2.putText(panel, "  s: Skip", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    y += 20
    cv2.putText(panel, "  q: Quit", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    
    # Combine
    combined = np.hstack([display, panel])
    
    cv2.imshow("Label Tile", combined)


def label_tiles(input_dir: Path, output_dir: Path, model_name: str = None):
    """
    Interactive labeling with model suggestions.
    
    Args:
        input_dir: Directory containing tiles to label
        output_dir: Base directory for labeled output (will create class subdirs)
        model_name: Model name (default: chess_resnet18_best)
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    
    # Load model
    print(f"Loading model...")
    model = load_model(model_name)
    print(f"Model loaded")
    
    # Find all tiles
    extensions = ('.png', '.jpg', '.jpeg')
    tiles = sorted([
        p for p in input_dir.rglob('*')
        if p.suffix.lower() in extensions and 'overlay' not in p.name.lower()
    ])
    
    if not tiles:
        print(f"No tiles found in {input_dir}")
        return
    
    print(f"\nFound {len(tiles)} tiles to label")
    print("Valid labels: " + ", ".join(CLASSES))
    print("Shortcuts: p/n/b/r/q/k=black, P/N/B/R/Q/K=white, e/.=empty")
    print("\nControls:")
    print("  Enter     - Accept suggested label")
    print("  Shortcut  - p=bP, P=wP, n=bN, N=wN, etc.")
    print("  Full name - wK, bP, empty, etc.")
    print("  s         - Skip this tile")
    print("  q         - Quit\n")
    
    # Create output directories
    for cls in CLASSES:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)
    
    accepted = 0
    overridden = 0
    skipped = 0
    
    cv2.namedWindow("Label Tile", cv2.WINDOW_AUTOSIZE)
    
    for i, tile_path in enumerate(tiles):
        # Predict
        pred_class, confidence, all_probs = predict_tile(model, tile_path)
        
        if pred_class is None:
            print(f"  ❌ Failed to load: {tile_path.name}")
            skipped += 1
            continue
        
        # Show tile
        show_tile_opencv(tile_path, pred_class, confidence, all_probs)
        
        # Status line
        status = f"[{i+1}/{len(tiles)}] {tile_path.name} → {pred_class} ({confidence:.0%})"
        print(f"\r{status.ljust(80)}", end="", flush=True)
        
        # Wait for input
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == 13:  # Enter - accept
                label = pred_class
                break
            elif key == ord('s') or key == ord('S'):  # Skip
                label = None
                break
            elif key == ord('q') or key == ord('Q'):  # Quit
                print(f"\n\nQuitting...")
                print(f"  Accepted:  {accepted}")
                print(f"  Overridden: {overridden}")
                print(f"  Skipped:   {skipped}")
                cv2.destroyAllWindows()
                return
            elif key == 27:  # Escape - quit
                cv2.destroyAllWindows()
                return
        
        if label is None:
            print(f"\r[{i+1}/{len(tiles)}] {tile_path.name} → SKIPPED".ljust(80))
            skipped += 1
            continue
        
        # If we need to type a label, use console input
        # For now, just use Enter to accept or press a number for quick override
        # Let's make it simpler with keyboard shortcuts
        
        # Copy file to output
        dest = output_dir / label / tile_path.name
        
        # Avoid overwriting
        if dest.exists():
            stem = tile_path.stem
            suffix = tile_path.suffix
            counter = 1
            while dest.exists():
                dest = output_dir / label / f"{stem}_{counter}{suffix}"
                counter += 1
        
        # Copy the file
        import shutil
        shutil.copy2(tile_path, dest)
        
        if label == pred_class:
            accepted += 1
            result = "✓"
        else:
            overridden += 1
            result = f"→ {label}"
        
        print(f"\r[{i+1}/{len(tiles)}] {tile_path.name} → {pred_class} ({confidence:.0%}) {result}".ljust(80))
    
    cv2.destroyAllWindows()
    
    print(f"\n\nDone!")
    print(f"  Accepted:   {accepted}")
    print(f"  Overridden: {overridden}")
    print(f"  Skipped:    {skipped}")
    print(f"  Output:     {output_dir}")


def label_tiles_console(input_dir: Path, output_dir: Path, model_name: str = None):
    """
    Console-based interactive labeling with model suggestions.
    Better for typing override labels.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    
    # Load model
    print(f"Loading model...")
    model = load_model(model_name)
    print(f"Model loaded")
    
    # Find all tiles
    extensions = ('.png', '.jpg', '.jpeg')
    tiles = sorted([
        p for p in input_dir.rglob('*')
        if p.suffix.lower() in extensions and 'overlay' not in p.name.lower()
    ])
    
    if not tiles:
        print(f"No tiles found in {input_dir}")
        return
    
    print(f"\nFound {len(tiles)} tiles to label")
    print("Valid labels: " + ", ".join(CLASSES))
    print("Shortcuts: p/n/b/r/q/k=black, P/N/B/R/Q/K=white, e/.=empty")
    print("\nControls:")
    print("  Enter     - Accept suggested label")
    print("  Shortcut  - p=bP, P=wP, n=bN, N=wN, etc.")
    print("  Full name - wK, bP, empty, etc.")
    print("  s         - Skip this tile")
    print("  q         - Quit\n")
    
    # Create output directories
    for cls in CLASSES:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)
    
    accepted = 0
    overridden = 0
    skipped = 0
    
    window_name = "Tile Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    for i, tile_path in enumerate(tiles):
        # Predict
        pred_class, confidence, all_probs = predict_tile(model, tile_path)
        
        if pred_class is None:
            print(f"  ❌ Failed to load: {tile_path.name}")
            skipped += 1
            continue
        
        # Show tile in OpenCV window with prediction in title
        img = cv2.imread(str(tile_path))
        if img is not None:
            scale = 5
            h, w = img.shape[:2]
            display = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
            
            # Add prediction text overlay on the image
            cv2.putText(display, f"{pred_class} ({confidence:.0%})", (5, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window_name, display)
            cv2.setWindowTitle(window_name, f"{pred_class} ({confidence:.0%}) - {tile_path.name}")
            cv2.waitKey(1)  # Update display
        
        # Print prediction info
        print(f"\n[{i+1}/{len(tiles)}] {tile_path.name}")
        print(f"  Prediction: {pred_class} ({confidence:.1%})")
        print(f"  Top 3: {all_probs[0][0]}={all_probs[0][1]:.0%}, {all_probs[1][0]}={all_probs[1][1]:.0%}, {all_probs[2][0]}={all_probs[2][1]:.0%}")
        
        # Get input
        try:
            user_input = input(f"  Label [{pred_class}]: ").strip()
        except EOFError:
            break
        
        if user_input.lower() == 'q':
            print(f"\nQuitting...")
            break
        elif user_input.lower() == 's':
            print(f"  → Skipped")
            skipped += 1
            continue
        elif user_input == '':
            label = pred_class
        elif user_input in SHORTHAND:
            label = SHORTHAND[user_input]
        elif user_input in CLASSES:
            label = user_input
        else:
            print(f"  ⚠ Invalid label '{user_input}', skipping")
            skipped += 1
            continue
        
        # Copy file to output
        dest = output_dir / label / tile_path.name
        
        # Avoid overwriting
        if dest.exists():
            stem = tile_path.stem
            suffix = tile_path.suffix
            counter = 1
            while dest.exists():
                dest = output_dir / label / f"{stem}_{counter}{suffix}"
                counter += 1
        
        import shutil
        shutil.copy2(tile_path, dest)
        
        if label == pred_class:
            accepted += 1
            print(f"  → Accepted: {label}")
        else:
            overridden += 1
            print(f"  → Overridden: {pred_class} → {label}")
    
    cv2.destroyAllWindows()
    
    print(f"\n\nDone!")
    print(f"  Accepted:   {accepted}")
    print(f"  Overridden: {overridden}")
    print(f"  Skipped:    {skipped}")
    print(f"  Output:     {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive labeling with model predictions")
    parser.add_argument("input_dir", type=Path, help="Directory containing tiles to label")
    parser.add_argument("output_dir", type=Path, help="Output directory for labeled tiles")
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g. chess_resnet18_best)")
    parser.add_argument("--console", action="store_true", help="Use console mode (better for typing labels)")
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    if args.console:
        label_tiles_console(args.input_dir, args.output_dir, args.model)
    else:
        label_tiles(args.input_dir, args.output_dir, args.model)
