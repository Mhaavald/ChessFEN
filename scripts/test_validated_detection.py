"""Test the validated detection on a specific image."""
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'inference'))
from inference_service import load_model, predict_board

model = load_model()
img = cv2.imread('data/raw/batch_007/20251231_125333956_iOS.png')
result = predict_board(img, model, debug=True)

print('Success:', result['success'])
print('FEN:', result['fen'])
print('Avg conf:', round(result.get('avg_confidence', 0), 4))

info = result.get('detection_info', {})
for c in info.get('tried_candidates', []):
    print(f"Cand {c['index']}: grid={c['grid_score']:.3f} conf={c['avg_confidence']:.3f} valid={c['is_valid']} reason={c['reason']}")

# Save the warped image to verify
if 'debug_images' in result and 'warped' in result['debug_images']:
    from PIL import Image
    warped = result['debug_images']['warped']
    Image.fromarray(warped).save('data/debug/20251231_125333956_validated_warped.png')
    print('Saved: data/debug/20251231_125333956_validated_warped.png')
