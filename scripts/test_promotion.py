#!/usr/bin/env python
"""Test 3532ff6e - case with TWO valid black queens (promotion position)."""

import sys
sys.path.insert(0, 'src/inference')

import cv2
from inference_service import predict_board, load_model

model = load_model()
print('Model loaded')

# 3532ff6e - has TWO valid black queens (c3 and e7)
img = cv2.imread('data/user_feedback/3532ff6e.png')
result = predict_board(img, model)
print(f'FEN: {result["fen"]}')
print(f'Queen fixes: {result.get("queen_fixes", [])}')

# Expected FEN: 3r2k1/p3q2p/1p2P2P/6p1/5b2/1Bq5/PPP5/1K3R2
# Note: This has TWO black queens (e7 and c3) - valid promotion position
expected = '3r2k1/p3q2p/1p2P2P/6p1/5b2/1Bq5/PPP5/1K3R2'
if result['fen'] == expected:
    print('SUCCESS: FEN matches expected!')
else:
    print(f'Expected: {expected}')
    print(f'Got:      {result["fen"]}')
