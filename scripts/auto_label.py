#!/usr/bin/env python3
"""
Auto-label tiles using model predictions.
High-confidence predictions go to a review folder.
Low-confidence ones are left for manual review.
"""

import argparse
import shutil
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.inference_service import load_model, preprocess_image

CLASSES = ['empty', 'wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']

IMG_SIZE = 64
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_tile(tile_path: Path, model) -> np.ndarray:
    """Predict class probabilities for a single tile."""
    # Load and preprocess image
    img_bgr = cv2.imread(str(tile_path))
    if img_bgr is None:
        return None
    
    # Apply same preprocessing as inference service
    img_bgr = preprocess_image(img_bgr)
    
    # Convert to PIL for transform
    tile_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(tile_rgb)
    img_tensor = transform(pil_img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
    
    return probs


def auto_label(input_dir: Path, output_dir: Path, uncertain_dir: Path, 
               confidence_threshold: float = 0.90, model_path: str = None):
    """
    Auto-label tiles based on model confidence.
    
    Args:
        input_dir: Directory with tiles to label
        output_dir: Directory for high-confidence labels (organized by class)
        uncertain_dir: Directory for low-confidence tiles (flat, for manual review)
        confidence_threshold: Minimum confidence to auto-label
        model_path: Optional path to model file
    """
    # Load model
    print("Loading model...")
    model = load_model(model_path)
    print(f"Confidence threshold: {confidence_threshold:.0%}")
    print()
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    uncertain_dir.mkdir(parents=True, exist_ok=True)
    
    for cls in CLASSES:
        (output_dir / cls).mkdir(exist_ok=True)
    
    # Find all tiles (skip overlays)
    extensions = {'.png', '.jpg', '.jpeg'}
    tiles = []
    for ext in extensions:
        tiles.extend(input_dir.rglob(f'*{ext}'))
    
    # Filter out overlays
    tiles = [t for t in tiles if 'overlay' not in t.name.lower()]
    tiles = sorted(tiles)
    
    print(f"Found {len(tiles)} tiles to process")
    print()
    
    # Process tiles
    auto_labeled = 0
    uncertain = 0
    stats = {cls: 0 for cls in CLASSES}
    uncertain_stats = {cls: 0 for cls in CLASSES}
    
    for i, tile_path in enumerate(tiles):
        # Get prediction
        probs = predict_tile(tile_path, model)
        
        if probs is None:
            print(f"  Warning: Could not load {tile_path.name}")
            continue
        
        # Get top prediction
        top_idx = probs.argmax()
        top_class = CLASSES[top_idx]
        top_conf = probs[top_idx]
        
        if top_conf >= confidence_threshold:
            # High confidence - auto-label
            dest = output_dir / top_class / tile_path.name
            
            # Handle duplicates
            if dest.exists():
                stem = tile_path.stem
                suffix = tile_path.suffix
                counter = 1
                while dest.exists():
                    dest = output_dir / top_class / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            shutil.copy2(tile_path, dest)
            auto_labeled += 1
            stats[top_class] += 1
        else:
            # Low confidence - needs manual review
            # Save with prediction info in filename for easier review
            conf_pct = int(top_conf * 100)
            new_name = f"{top_class}_{conf_pct}pct_{tile_path.name}"
            dest = uncertain_dir / new_name
            
            shutil.copy2(tile_path, dest)
            uncertain += 1
            uncertain_stats[top_class] += 1
        
        # Progress
        if (i + 1) % 100 == 0 or i == len(tiles) - 1:
            print(f"  Processed {i+1}/{len(tiles)} - Auto: {auto_labeled}, Uncertain: {uncertain}")
    
    print()
    print("=" * 50)
    print(f"Auto-labeled: {auto_labeled} ({auto_labeled/len(tiles)*100:.1f}%)")
    print(f"Uncertain:    {uncertain} ({uncertain/len(tiles)*100:.1f}%)")
    print()
    
    print("Auto-labeled by class:")
    for cls in CLASSES:
        if stats[cls] > 0:
            print(f"  {cls:6}: {stats[cls]:4}")
    
    if uncertain > 0:
        print()
        print("Uncertain by predicted class:")
        for cls in CLASSES:
            if uncertain_stats[cls] > 0:
                print(f"  {cls:6}: {uncertain_stats[cls]:4}")
    
    print()
    print(f"Auto-labeled tiles saved to: {output_dir}")
    print(f"Uncertain tiles saved to:    {uncertain_dir}")
    print()
    print("Review the auto-labeled folder, then merge with:")
    print(f"  Copy-Item -Path '{output_dir}\\*' -Destination 'data/labeled_squares' -Recurse -Force")


def main():
    parser = argparse.ArgumentParser(description='Auto-label tiles using model predictions')
    parser.add_argument('input_dir', type=Path, help='Directory with tiles to label')
    parser.add_argument('--output', '-o', type=Path, default=Path('data/auto_labeled'),
                        help='Output directory for auto-labeled tiles (default: data/auto_labeled)')
    parser.add_argument('--uncertain', '-u', type=Path, default=Path('data/uncertain'),
                        help='Output directory for uncertain tiles (default: data/uncertain)')
    parser.add_argument('--threshold', '-t', type=float, default=0.90,
                        help='Confidence threshold for auto-labeling (default: 0.90)')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to model file (default: auto-detect)')
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    auto_label(args.input_dir, args.output, args.uncertain, args.threshold, args.model)


if __name__ == '__main__':
    main()
