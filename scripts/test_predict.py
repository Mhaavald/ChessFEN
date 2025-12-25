"""Quick test of prediction with orientation correction."""
import sys
import cv2
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src" / "inference"))

from inference_service import load_model, predict_board

img_path = REPO_ROOT / "data" / "raw" / "batch_006" / "Skjermbilde 2025-12-23 005618.png"
print(f"Loading: {img_path}")

img = cv2.imread(str(img_path))
print(f"Image shape: {img.shape}")

model = load_model()
print("Model loaded")

result = predict_board(img, model, debug=False)
print(f"\nSuccess: {result['success']}")

if result['success']:
    print(f"FEN: {result['fen']}")
    print("\nBoard:")
    for row in result['board']:
        print("  ", " ".join(f"{p:6s}" for p in row))
else:
    print(f"Error: {result['error']}")
