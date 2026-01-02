"""
Train a CNN classifier for chess piece recognition.
Uses labeled squares from data/labeled_squares/
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
import json
from datetime import datetime

# Make repo imports work
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))


# ============================
# Configuration
# ============================

CLASSES = [
    "empty",
    "wP", "wN", "wB", "wR", "wQ", "wK",
    "bP", "bN", "bB", "bR", "bQ", "bK",
]

CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls for i, cls in enumerate(CLASSES)}

# Training hyperparameters
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0005  # Lower LR for fine-tuning
VAL_SPLIT = 0.15
CONVERGENCE_THRESHOLD = 0.001  # Min improvement to count as progress (0.1%)
CONVERGENCE_PATIENCE = 5  # Stop after N epochs with < threshold improvement


# ============================
# Dataset
# ============================

class ChessSquareDataset(Dataset):
    """Dataset for labeled chess squares."""
    
    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # Collect all images from class subdirectories
        for cls_name in CLASSES:
            cls_dir = self.root_dir / cls_name
            if cls_dir.exists():
                for img_path in cls_dir.glob("*.png"):
                    self.samples.append((img_path, CLASS_TO_IDX[cls_name]))
                for img_path in cls_dir.glob("*.jpg"):
                    self.samples.append((img_path, CLASS_TO_IDX[cls_name]))
        
        print(f"Loaded {len(self.samples)} samples from {root_dir}")
        
        # Print class distribution
        counts = {}
        for _, label in self.samples:
            cls = IDX_TO_CLASS[label]
            counts[cls] = counts.get(cls, 0) + 1
        print("Class distribution:")
        for cls in CLASSES:
            print(f"  {cls}: {counts.get(cls, 0)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================
# Model
# ============================

class ChessCNN(nn.Module):
    """Simple CNN for chess piece classification."""
    
    def __init__(self, num_classes=13):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 3: 16x16 -> 8x8
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


def create_resnet18(num_classes=13, pretrained=True):
    """Create ResNet-18 model with pretrained weights for transfer learning."""
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)
    
    # Freeze early layers (optional - can unfreeze for fine-tuning)
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
    
    # Replace final fully connected layer
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    
    return model


def create_resnet34(num_classes=13, pretrained=True):
    """Create ResNet-34 model with pretrained weights for transfer learning."""
    if pretrained:
        weights = models.ResNet34_Weights.IMAGENET1K_V1
        model = models.resnet34(weights=weights)
    else:
        model = models.resnet34(weights=None)
    
    # Freeze early layers
    for param in list(model.parameters())[:-30]:
        param.requires_grad = False
    
    # Replace final fully connected layer
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    
    return model


# ============================
# Training
# ============================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return total_loss / total, correct / total


def main(
    data_dir: str = "data/labeled_squares",
    output_dir: str = "models",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    model_type: str = "cnn",
    convergence_threshold: float = CONVERGENCE_THRESHOLD,
    convergence_patience: int = CONVERGENCE_PATIENCE,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model type: {model_type}")
    
    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load dataset
    full_dataset = ChessSquareDataset(data_dir, transform=train_transform)
    
    if len(full_dataset) == 0:
        print("No samples found! Check your data directory.")
        return
    
    # Split into train and validation
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Override transform for validation set
    # (random_split shares the same transform, so we need a workaround)
    val_dataset.dataset = ChessSquareDataset(data_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\nTrain samples: {train_size}")
    print(f"Val samples: {val_size}")
    
    # Model selection
    if model_type == "resnet18":
        model = create_resnet18(num_classes=len(CLASSES), pretrained=True).to(device)
        model_name = "chess_resnet18"
    elif model_type == "resnet34":
        model = create_resnet34(num_classes=len(CLASSES), pretrained=True).to(device)
        model_name = "chess_resnet34"
    else:
        model = ChessCNN(num_classes=len(CLASSES)).to(device)
        model_name = "chess_cnn"
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    prev_val_acc = 0.0
    stagnant_epochs = 0  # Epochs with improvement below threshold
    
    print(f"\nTraining for up to {epochs} epochs...")
    print(f"  Learning rate: {lr}")
    print(f"  Convergence: stop if improvement < {convergence_threshold*100:.1f}% for {convergence_patience} epochs")
    print("-" * 70)
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        scheduler.step(val_loss)
        
        # Calculate improvement
        improvement = val_acc - prev_val_acc
        prev_val_acc = val_acc
        
        # Build status string
        status_parts = []
        if improvement > convergence_threshold:
            status_parts.append(f"+{improvement*100:.2f}%")
            stagnant_epochs = 0
        elif epoch > 1:  # Don't count first epoch
            stagnant_epochs += 1
            status_parts.append(f"stagnant {stagnant_epochs}/{convergence_patience}")
        
        # Show if LR was reduced
        if current_lr < lr:
            status_parts.append(f"lr={current_lr:.6f}")
        
        status = f" [{', '.join(status_parts)}]" if status_parts else ""
        
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"Val: {val_loss:.4f}/{val_acc:.4f}{status}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            model_path = output_dir / f"{model_name}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'classes': CLASSES,
                'model_type': model_type,
            }, model_path)
            print(f"  ✓ Saved best model (val_acc: {val_acc:.4f})")
        
        # Check for convergence
        if stagnant_epochs >= convergence_patience:
            print(f"\n✓ Converged: improvement < {convergence_threshold*100:.1f}% for {convergence_patience} epochs")
            break
    
    print("-" * 60)
    print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    
    # Save final model
    final_path = output_dir / f"{model_name}_final.pth"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'val_acc': val_acc,
        'classes': CLASSES,
        'model_type': model_type,
    }, final_path)
    print(f"Saved final model to: {final_path}")
    
    # Save training info
    info = {
        'timestamp': datetime.now().isoformat(),
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'img_size': IMG_SIZE,
        'train_samples': train_size,
        'val_samples': val_size,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'classes': CLASSES,
    }
    
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\nTraining complete! Models saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train chess piece classifier")
    parser.add_argument("--data", default="data/labeled_squares", help="Path to labeled squares")
    parser.add_argument("--output", default="models", help="Output directory for models")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Max number of epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--model", choices=["cnn", "resnet18", "resnet34"], default="resnet18",
                        help="Model architecture: cnn (custom), resnet18, resnet34")
    parser.add_argument("--all", action="store_true", help="Train both CNN and ResNet18 models")
    parser.add_argument("--conv-threshold", type=float, default=CONVERGENCE_THRESHOLD,
                        help="Min improvement to count as progress (default: 0.001 = 0.1%%)")
    parser.add_argument("--conv-patience", type=int, default=CONVERGENCE_PATIENCE,
                        help="Stop after N epochs with < threshold improvement (default: 5)")
    
    args = parser.parse_args()
    
    if args.all:
        # Train both models
        print("="*60)
        print("Training CNN model")
        print("="*60)
        main(
            data_dir=args.data,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            model_type="cnn",
            convergence_threshold=args.conv_threshold,
            convergence_patience=args.conv_patience,
        )
        
        print("\n")
        print("="*60)
        print("Training ResNet18 model")
        print("="*60)
        main(
            data_dir=args.data,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            model_type="resnet18",
            convergence_threshold=args.conv_threshold,
            convergence_patience=args.conv_patience,
        )
    else:
        main(
            data_dir=args.data,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            model_type=args.model,
            convergence_threshold=args.conv_threshold,
            convergence_patience=args.conv_patience,
        )
