"""
Predict chess positions from images using the trained classifier.
Outputs FEN notation.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import cv2
from tkinter import Tk, filedialog

# Make repo imports work
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from scripts.detect_board import detect_board_quad_grid_aware, warp_quad


# ============================
# Model (must match training)
# ============================

CLASSES = [
    "empty",
    "wP", "wN", "wB", "wR", "wQ", "wK",
    "bP", "bN", "bB", "bR", "bQ", "bK",
]

# FEN piece mapping
PIECE_TO_FEN = {
    "empty": "",
    "wP": "P", "wN": "N", "wB": "B", "wR": "R", "wQ": "Q", "wK": "K",
    "bP": "p", "bN": "n", "bB": "b", "bR": "r", "bQ": "q", "bK": "k",
}


class ChessCNN(nn.Module):
    """Simple CNN for chess piece classification."""
    
    def __init__(self, num_classes=13):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_resnet18(num_classes=13):
    """Create ResNet-18 model matching training script structure."""
    model = models.resnet18(weights=None)
    
    # Replace final FC layer (must match training script exactly)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    
    return model


# ============================
# Prediction
# ============================

def load_model(model_path: Path, device: torch.device) -> nn.Module:
    """Load trained model from checkpoint.
    
    Automatically detects model type from checkpoint.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Detect model type from checkpoint
    model_type = checkpoint.get('model_type', 'cnn')  # Default to CNN for old checkpoints
    
    if model_type == 'resnet18':
        model = create_resnet18(num_classes=len(CLASSES))
        print(f"Loading ResNet-18 model from {model_path}")
    else:
        model = ChessCNN(num_classes=len(CLASSES))
        print(f"Loading CNN model from {model_path}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def predict_square(model, tile_bgr, transform, device) -> tuple:
    """Predict class for a single tile. Returns (class_name, confidence)."""
    # Convert BGR to RGB PIL Image
    tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(tile_rgb)
    
    # Transform and predict
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)
    
    class_name = CLASSES[predicted.item()]
    return class_name, confidence.item()


def board_to_fen(board: list) -> str:
    """Convert 8x8 board array to FEN position string."""
    fen_rows = []
    
    for row in board:
        fen_row = ""
        empty_count = 0
        
        for piece in row:
            fen_char = PIECE_TO_FEN[piece]
            if fen_char == "":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += fen_char
        
        if empty_count > 0:
            fen_row += str(empty_count)
        
        fen_rows.append(fen_row)
    
    return "/".join(fen_rows)


def predict_board(img_bgr, model, transform, device, debug=False):
    """
    Detect board, extract tiles, predict each square.
    Returns (fen_string, board_array, warped_image) or (None, None, None) if failed.
    """
    # Detect board
    quad = detect_board_quad_grid_aware(img_bgr, crop_ratio=0.80)
    if quad is None:
        return None, None, None
    
    # Warp to 800x800 (100px per tile)
    warped = warp_quad(img_bgr, quad, out_size=800)
    tile_size = 100
    
    # Predict each square
    board = []
    confidences = []
    
    for row in range(8):
        board_row = []
        conf_row = []
        for col in range(8):
            tile = warped[row*tile_size:(row+1)*tile_size, 
                         col*tile_size:(col+1)*tile_size]
            
            class_name, conf = predict_square(model, tile, transform, device)
            board_row.append(class_name)
            conf_row.append(conf)
            
            if debug:
                square = chr(ord('a') + col) + str(8 - row)
                print(f"  {square}: {class_name} ({conf:.2%})")
        
        board.append(board_row)
        confidences.append(conf_row)
    
    fen = board_to_fen(board)
    return fen, board, warped


def print_board(board: list):
    """Print board in a readable format."""
    print("\n  a  b  c  d  e  f  g  h")
    print(" ┌──┬──┬──┬──┬──┬──┬──┬──┐")
    
    for row_idx, row in enumerate(board):
        rank = 8 - row_idx
        row_str = f"{rank}│"
        for piece in row:
            if piece == "empty":
                row_str += "  │"
            else:
                # Use unicode chess symbols
                symbols = {
                    "wK": "♔", "wQ": "♕", "wR": "♖", "wB": "♗", "wN": "♘", "wP": "♙",
                    "bK": "♚", "bQ": "♛", "bR": "♜", "bB": "♝", "bN": "♞", "bP": "♟",
                }
                row_str += f"{symbols.get(piece, piece):>2}│"
        print(row_str)
        
        if row_idx < 7:
            print(" ├──┼──┼──┼──┼──┼──┼──┼──┤")
    
    print(" └──┴──┴──┴──┴──┴──┴──┴──┘")


def main(image_path: str = None, model_path: str = "models/chess_cnn_best.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train the model first with: python scripts/train_classifier.py")
        return
    
    print(f"Loading model: {model_path}")
    model = load_model(model_path, device)
    
    # Image transform (must match training)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Get image path
    if image_path is None:
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        image_path = filedialog.askopenfilename(
            title="Select a chess board image",
            initialdir=REPO_ROOT / "data" / "raw",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if not image_path:
            print("No file selected.")
            return
    
    print(f"\nProcessing: {image_path}")
    
    # Load and process image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to read image: {image_path}")
        return
    
    # Predict
    fen, board, warped = predict_board(img, model, transform, device, debug=False)
    
    if fen is None:
        print("❌ Board not detected")
        return
    
    print("\n✅ Board detected and classified!")
    print_board(board)
    print(f"\nFEN: {fen}")
    
    # Show result
    cv2.imshow("Warped Board", warped)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(image_path=sys.argv[1])
    else:
        main()
