#!/usr/bin/env python
"""Test all feedback cases to verify fixes don't break anything."""

import sys
sys.path.insert(0, 'src/inference')

import os
import json
import cv2
from inference_service import predict_board, load_model

# Load model
model = load_model()
print('Model loaded\n')

# Get all feedback files
feedback_dir = 'data/user_feedback'
feedback_files = [f for f in os.listdir(feedback_dir) if f.endswith('.json')]

results = {
    'fully_fixed': [],
    'partially_fixed': [],
    'not_fixed': [],
    'new_errors': []
}

for fb_file in sorted(feedback_files):
    fb_id = fb_file.replace('.json', '')
    fb_path = os.path.join(feedback_dir, fb_file)
    img_path = os.path.join(feedback_dir, f"{fb_id}.png")
    
    if not os.path.exists(img_path):
        print(f"[{fb_id}] No image found, skipping")
        continue
    
    with open(fb_path) as f:
        feedback = json.load(f)
    
    corrected_squares = feedback.get('corrected_squares', {})
    expected_fen = feedback.get('corrected_fen', '')
    
    if not corrected_squares:
        print(f"[{fb_id}] No corrections specified, skipping")
        continue
    
    # Run prediction
    img = cv2.imread(img_path)
    if img is None:
        print(f"[{fb_id}] Failed to load image, skipping")
        continue
        
    result = predict_board(img, model)
    if result is None:
        print(f"[{fb_id}] Prediction failed, skipping")
        continue
        
    predicted_fen = result['fen']
    
    # Check if the corrected squares are now correct
    board = result.get('board')
    if board is None:
        print(f"[{fb_id}] ❌ No board returned")
        results['not_fixed'].append(fb_id)
        continue
        
    fixed_squares = []
    still_wrong = []
    
    for square, expected_piece in corrected_squares.items():
        col = ord(square[0]) - ord('a')
        row = 8 - int(square[1])
        actual_piece = board[row][col]
        
        if actual_piece == expected_piece:
            fixed_squares.append(square)
        else:
            still_wrong.append(f"{square}: expected {expected_piece}, got {actual_piece}")
    
    # Check for new errors (pieces that were correct in original but wrong now)
    king_fixes = result.get('king_fixes', [])
    queen_fixes = result.get('queen_fixes', [])
    
    status = "OK" if predicted_fen == expected_fen else "FAIL"
    
    print(f"[{fb_id}] {status}")
    if king_fixes:
        print(f"  King fixes: {king_fixes}")
    if queen_fixes:
        print(f"  Queen fixes: {queen_fixes}")
    if fixed_squares:
        print(f"  Fixed: {fixed_squares}")
    if still_wrong:
        print(f"  Still wrong: {still_wrong}")
    
    if predicted_fen == expected_fen:
        results['fully_fixed'].append(fb_id)
    elif fixed_squares:
        results['partially_fixed'].append(fb_id)
    else:
        results['not_fixed'].append(fb_id)

print("\n" + "="*50)
print(f"Fully fixed: {len(results['fully_fixed'])} / {len(feedback_files)}")
print(f"Partially fixed: {len(results['partially_fixed'])}")
print(f"Not fixed: {len(results['not_fixed'])}")
if results['not_fixed']:
    print(f"  IDs: {results['not_fixed']}")
