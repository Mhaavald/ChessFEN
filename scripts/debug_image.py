"""
Debug script to analyze a specific image and see per-square predictions.
Shows exactly what tiles are being fed to the model.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from pathlib import Path

# Import from inference service
from src.inference.inference_service import (
    load_model, predict_board, detect_board_quad_grid_aware, 
    warp_quad, CLASSES, transform, correct_board_orientation
)
# Import edge refinement
from scripts.detect_board import expand_quad_to_edges, refine_quad_by_tile_uniformity, refine_quad_for_digital_board
# Import the same tile extraction used by review_overlays.py
from scripts.extract import split_board_to_squares, draw_grid_overlay
import torch
from PIL import Image

def debug_image(image_path):
    """Analyze an image and show per-square predictions with confidence."""
    
    print(f"Loading image: {image_path}")
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print("ERROR: Could not load image")
        return
    
    print(f"Image shape: {img_bgr.shape}")
    
    # Load model
    model = load_model()
    print(f"Model loaded")
    
    # Detect board (use adaptive detection, no hardcoded crop_ratio)
    quad = detect_board_quad_grid_aware(img_bgr)
    if quad is None:
        print("ERROR: Could not detect board")
        return
    
    print(f"Board detected: {quad.shape}")
    print(f"Quad corners (before refinement):\n{quad}")
    
    # Try multiple refinement strategies and pick the best one
    # Strategy 1: Gradient expansion + tile uniformity (good for physical boards)
    quad_expanded = expand_quad_to_edges(img_bgr, quad, max_expand=50)
    quad_physical = refine_quad_by_tile_uniformity(img_bgr, quad_expanded, max_adjust=15, step=2)
    
    # Strategy 2: Digital/book refinement (good for printed diagrams and screenshots)
    quad_digital = refine_quad_for_digital_board(img_bgr, quad)
    
    # Choose based on which gives more square result (books/digital should be perfectly square)
    pts_p = quad_physical.reshape(-1, 2)
    pts_d = quad_digital.reshape(-1, 2)
    
    size_p = (pts_p[1][0] - pts_p[0][0], pts_p[3][1] - pts_p[0][1])
    size_d = (pts_d[1][0] - pts_d[0][0], pts_d[3][1] - pts_d[0][1])
    
    aspect_p = abs(size_p[0] - size_p[1]) / max(size_p)
    aspect_d = abs(size_d[0] - size_d[1]) / max(size_d)
    
    if aspect_d < aspect_p:
        quad_refined = quad_digital
        print(f"Using DIGITAL refinement (aspect diff: {aspect_d:.3f} vs {aspect_p:.3f})")
    else:
        quad_refined = quad_physical
        print(f"Using PHYSICAL refinement (aspect diff: {aspect_p:.3f} vs {aspect_d:.3f})")
    
    print(f"Quad corners (after refinement):\n{quad_refined}")
    
    # Save output directory
    output_dir = Path(image_path).parent / "debug_output"
    output_dir.mkdir(exist_ok=True)
    
    # Save detection overlay showing original and refined quads
    detection_img = img_bgr.copy()
    # Original quad in red
    pts_orig = quad.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(detection_img, [pts_orig], True, (0, 0, 255), 2)
    # Physical refinement in yellow
    pts_phys = quad_physical.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(detection_img, [pts_phys], True, (0, 255, 255), 2)
    # Digital refinement in cyan
    pts_dig = quad_digital.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(detection_img, [pts_dig], True, (255, 255, 0), 2)
    # Final refined quad in green
    pts_refined = quad_refined.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(detection_img, [pts_refined], True, (0, 255, 0), 3)
    # Draw corner points for refined
    for i, pt in enumerate(quad_refined.reshape(-1, 2)):
        cv2.circle(detection_img, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), -1)
        cv2.putText(detection_img, str(i), (int(pt[0])+10, int(pt[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    detection_path = output_dir / f"{Path(image_path).stem}_detection.png"
    cv2.imwrite(str(detection_path), detection_img)
    print(f"Saved detection overlay: {detection_path.name}")
    print("  Red = original, Yellow = physical, Cyan = digital, Green = chosen")
    
    # Use the REFINED quad for warping
    quad = quad_refined
    
    # Warp board (BEFORE orientation correction) - use 640 like review_overlays.py
    warped_raw = warp_quad(img_bgr, quad, out_size=640)
    
    # Correct orientation (h1 = light, a1 = dark)
    warped = correct_board_orientation(warped_raw)
    print("Board orientation corrected")
    
    # Use the SAME tile extraction as review_overlays.py (with autocrop)
    squares, sq_size = split_board_to_squares(warped, do_autocrop=True)
    print(f"Tile extraction: {sq_size}x{sq_size} pixels per square")
    
    # Create tiles subdirectory
    tiles_dir = output_dir / "prediction_tiles"
    tiles_dir.mkdir(exist_ok=True)
    
    # Save warped images
    raw_path = output_dir / f"{Path(image_path).stem}_warped_raw.png"
    cv2.imwrite(str(raw_path), warped_raw)
    print(f"Saved RAW warped (before orientation): {raw_path.name}")
    
    corrected_path = output_dir / f"{Path(image_path).stem}_warped_corrected.png"
    cv2.imwrite(str(corrected_path), warped)
    print(f"Saved CORRECTED warped (after orientation): {corrected_path.name}")
    
    # Save warped with grid overlay (using the same function as review_overlays.py)
    grid_overlay = draw_grid_overlay(warped, sq_size)
    grid_path = output_dir / f"{Path(image_path).stem}_warped_grid.png"
    cv2.imwrite(str(grid_path), grid_overlay)
    print(f"Saved warped with grid: {grid_path.name}")
    
    print(f"\nSaving individual tiles to: {tiles_dir}")
    print("="*60)
    print("PIECE PREDICTIONS (row 0 = rank 8, col 0 = file a)")
    print("="*60)
    
    files = "abcdefgh"
    board = []
    piece_symbols = {
        'empty': '.', 
        'wP': 'P', 'wN': 'N', 'wB': 'B', 'wR': 'R', 'wQ': 'Q', 'wK': 'K',
        'bP': 'p', 'bN': 'n', 'bB': 'b', 'bR': 'r', 'bQ': 'q', 'bK': 'k'
    }
    
    with torch.no_grad():
        for row in range(8):
            rank = 8 - row
            row_predictions = []
            for col in range(8):
                file = files[col]
                square = f"{file}{rank}"
                
                # Use the SAME tiles from split_board_to_squares (with autocrop)
                tile = squares[row][col]
                
                # Save the EXACT tile being used for prediction
                tile_path = tiles_dir / f"{square}.png"
                cv2.imwrite(str(tile_path), tile)
                
                # Convert to PIL for transform
                tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(tile_rgb)
                img_tensor = transform(pil_img).unsqueeze(0)
                
                # Predict
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_idx = probs.argmax().item()
                confidence = probs[0, pred_idx].item()
                
                piece = CLASSES[pred_idx]
                row_predictions.append((piece, confidence))
                
                # Show predictions
                if piece != "empty" or confidence < 0.9:
                    print(f"  {square}: {piece:12s} ({confidence*100:5.1f}%)")
                    
                    if confidence < 0.8:
                        top_probs, top_indices = torch.topk(probs[0], 3)
                        print(f"       Top 3: ", end="")
                        for p, i in zip(top_probs, top_indices):
                            print(f"{CLASSES[i]}({p.item()*100:.1f}%) ", end="")
                        print()
            
            board.append(row_predictions)
    
    print(f"\nTiles saved to: {tiles_dir}")
    print("Each tile is EXACTLY what was fed to the model")
    
    # Board visualization
    print("\n" + "="*60)
    print("BOARD VISUALIZATION")
    print("="*60)
    
    print("    a b c d e f g h")
    print("  +-----------------+")
    for row in range(8):
        rank = 8 - row
        print(f"{rank} | ", end="")
        for col in range(8):
            piece, conf = board[row][col]
            symbol = piece_symbols.get(piece, '?')
            if conf < 0.7:
                print(f"({symbol})", end="")
            else:
                print(f" {symbol} ", end="")
        print(f"| {rank}")
    print("  +-----------------+")
    print("    a b c d e f g h")
    
    # Generate FEN
    fen_rows = []
    for row in range(8):
        fen_row = ""
        empty_count = 0
        for col in range(8):
            piece, _ = board[row][col]
            if piece == "empty":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += piece_symbols.get(piece, '?')
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    
    fen = "/".join(fen_rows)
    print(f"\nFEN: {fen}")
    
    print(f"\n" + "="*60)
    print("DEBUG FILES CREATED:")
    print("="*60)
    print(f"  {raw_path.name} - Warped board BEFORE orientation fix")
    print(f"  {corrected_path.name} - Warped board AFTER orientation fix")
    print(f"  {grid_path.name} - Warped with grid and labels")
    print(f"  prediction_tiles/*.png - Each tile as fed to the model")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_image.py <image_path>")
        sys.exit(1)
    
    debug_image(sys.argv[1])
