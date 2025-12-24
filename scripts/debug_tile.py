"""
Debug a specific tile on a chess board image.
Shows the extracted tile and prediction probabilities.

Usage:
    python scripts/debug_tile.py <image_path> <square>
    
Example:
    python scripts/debug_tile.py data/raw/batch_004/image.jpg b5
"""

import sys
from pathlib import Path
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from tkinter import Tk, filedialog

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))
from detect_board import detect_board_quad_grid_aware, warp_quad

IMG_SIZE = 64

CLASSES = ['empty', 'wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']


class ChessCNN(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def create_resnet18(num_classes=13):
    """Create ResNet-18 model matching training script structure."""
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


def parse_square(square: str) -> tuple:
    """Convert chess notation (e.g., 'b5') to row, col indices."""
    if len(square) != 2:
        raise ValueError(f"Invalid square: {square}")
    
    file = square[0].lower()
    rank = square[1]
    
    if file not in 'abcdefgh' or rank not in '12345678':
        raise ValueError(f"Invalid square: {square}")
    
    col = ord(file) - ord('a')  # a=0, b=1, ..., h=7
    row = 8 - int(rank)  # 8=0, 7=1, ..., 1=7
    
    return row, col


def main():
    # Parse arguments
    if len(sys.argv) < 2:
        # Open file dialog
        Tk().withdraw()
        image_path = filedialog.askopenfilename(
            title="Select chess board image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not image_path:
            print("No file selected")
            return
        square = input("Enter square to debug (e.g., b5): ").strip()
    elif len(sys.argv) == 2:
        image_path = sys.argv[1]
        square = input("Enter square to debug (e.g., b5): ").strip()
    else:
        image_path = sys.argv[1]
        square = sys.argv[2]
    
    # Validate square
    try:
        row, col = parse_square(square)
    except ValueError as e:
        print(e)
        return
    
    # Load model
    model_path = Path(__file__).parent.parent / "models" / "chess_cnn_best.pth"
    device = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Detect model type
    model_type = checkpoint.get('model_type', 'cnn')
    if model_type == 'resnet18':
        model = create_resnet18(num_classes=13)
        print(f"Loaded ResNet-18 model")
    else:
        model = ChessCNN(num_classes=13)
        print(f"Loaded CNN model")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load image and detect board
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    quad = detect_board_quad_grid_aware(img, crop_ratio=0.80)
    if quad is None:
        print("Could not detect board")
        return
    
    warped = warp_quad(img, quad, out_size=640)
    tile_size = 640 // 8
    
    # Extract tile
    y1, y2 = row * tile_size, (row + 1) * tile_size
    x1, x2 = col * tile_size, (col + 1) * tile_size
    tile = warped[y1:y2, x1:x2]
    tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
    
    # Predict
    with torch.no_grad():
        inp = transform(tile_rgb).unsqueeze(0)
        out = model(inp)
        probs = torch.softmax(out, dim=1)[0]
        pred = probs.argmax().item()
    
    # Print results
    print(f"\n{'='*40}")
    print(f"Square: {square.upper()}")
    print(f"Predicted: {CLASSES[pred]} ({probs[pred]:.2%})")
    print(f"{'='*40}")
    print("\nAll predictions:")
    for i in probs.argsort(descending=True):
        bar = '█' * int(probs[i] * 30)
        print(f"  {CLASSES[i]:>5}: {probs[i]:6.2%} {bar}")
    
    # Show tile
    display = cv2.resize(tile, (300, 300), interpolation=cv2.INTER_NEAREST)
    cv2.putText(display, f"{square.upper()}: {CLASSES[pred]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow(f'Tile {square.upper()}', display)
    
    # Also show the full warped board with grid
    board_display = warped.copy()
    for i in range(9):
        cv2.line(board_display, (i * tile_size, 0), (i * tile_size, 640), (0, 255, 0), 1)
        cv2.line(board_display, (0, i * tile_size), (640, i * tile_size), (0, 255, 0), 1)
    # Highlight selected square
    cv2.rectangle(board_display, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.imshow('Board (selected square highlighted)', board_display)
    
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
