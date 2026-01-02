"""Debug prediction for a specific square."""
import cv2
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'inference'))
from inference_service import load_model, predict_board, preprocess_image, transform, CLASSES, warp_quad
from detect_board import detect_board_quad_grid_aware

model = load_model()
img = cv2.imread('data/raw/batch_007/20251231_125233569_iOS.png')
print(f'Image shape: {img.shape}')

# Preprocess
img = preprocess_image(img)

# Detect and warp
quad = detect_board_quad_grid_aware(img, crop_ratio=1.0)
warped = warp_quad(img, quad, out_size=512)
tile_size = 512 // 8

# f2 is row 6 (rank 2), col 5 (file f)
row, col = 6, 5
square_name = "f2"

y1, y2 = row * tile_size, (row + 1) * tile_size
x1, x2 = col * tile_size, (col + 1) * tile_size
tile = warped[y1:y2, x1:x2]

# Save tile for inspection
cv2.imwrite(f'data/debug/{square_name}_tile.png', tile)
print(f'Saved tile: data/debug/{square_name}_tile.png')

# Run prediction
from PIL import Image
tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(tile_rgb)
img_tensor = transform(pil_img).unsqueeze(0)

with torch.no_grad():
    outputs = model(img_tensor)
    probs = torch.softmax(outputs, dim=1)[0]

print(f'\nPrediction for {square_name}:')
print('-' * 40)

# Sort by probability
sorted_probs = sorted(enumerate(probs.tolist()), key=lambda x: -x[1])
for idx, prob in sorted_probs:
    class_name = CLASSES[idx]
    bar = '#' * int(prob * 50)
    print(f'{class_name:6s}: {prob*100:5.1f}% {bar}')
