#!/usr/bin/env python
"""Test the queen disambiguation fix with feedback image."""

import sys
sys.path.insert(0, 'src/inference')

import cv2
from inference_service import predict_board, load_model

# Load model
model = load_model()
print('Model loaded')

# Load image
img = cv2.imread('data/user_feedback/dcd29323.png')
print(f'Image shape: {img.shape}')

# Run prediction
result = predict_board(img, model)
print(f'FEN: {result["fen"]}')
print(f'King fixes: {result.get("king_fixes", [])}')
print(f'Queen fixes: {result.get("queen_fixes", [])}')

# Check if d2 is now wQ
# d2 is row 6 (8-2=6), col 3 (d=3)
d2_piece = result['board'][6][3]
d2_conf = result['confidences'][6][3]
print(f'd2 piece: {d2_piece}, confidence: {d2_conf:.3f}')

# Expected correction
expected_fen = "5rk1/1b3p1p/pp3p2/3n1N2/1P6/P1qB1PP1/3Q3P/4R1K1"
if result["fen"] == expected_fen:
    print("SUCCESS: FEN matches expected!")
else:
    print(f"MISMATCH: Expected {expected_fen}")
