"""
Generate augmented data for underrepresented piece classes.
Uses various transformations to create new training samples.
"""

import cv2
import numpy as np
from pathlib import Path
import random


def augment_image(img: np.ndarray, idx: int) -> np.ndarray:
    """Apply augmentation based on index for variety."""
    h, w = img.shape[:2]
    result = img.copy()
    
    # Different augmentation combinations based on idx
    aug_type = idx % 10  # Increased from 8 to 10
    
    if aug_type == 0:
        # Slight rotation
        angle = random.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    elif aug_type == 1:
        # Brightness adjustment
        factor = random.uniform(0.7, 1.3)
        result = np.clip(result * factor, 0, 255).astype(np.uint8)
    
    elif aug_type == 2:
        # Mild contrast adjustment
        factor = random.uniform(0.8, 1.2)
        mean = np.mean(result)
        result = np.clip((result - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    elif aug_type == 3:
        # Small translation
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    elif aug_type == 4:
        # Slight zoom
        scale = random.uniform(0.9, 1.1)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(result, (new_w, new_h))
        
        if scale > 1:
            # Crop center
            y0 = (new_h - h) // 2
            x0 = (new_w - w) // 2
            result = resized[y0:y0+h, x0:x0+w]
        else:
            # Pad
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            result = cv2.copyMakeBorder(resized, pad_y, h-new_h-pad_y, 
                                        pad_x, w-new_w-pad_x, cv2.BORDER_REFLECT)
    
    elif aug_type == 5:
        # Gaussian noise
        noise = np.random.normal(0, 10, result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    elif aug_type == 6:
        # Slight blur
        result = cv2.GaussianBlur(result, (3, 3), 0)
    
    elif aug_type == 7:
        # Color jitter (hue/saturation)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    elif aug_type == 8:
        # HIGH CONTRAST - darks darker, lights lighter
        # This helps distinguish empty dark squares from dark pieces
        alpha = random.uniform(1.3, 1.8)  # contrast factor
        beta = random.uniform(-40, -20)   # slight darkening
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
    
    elif aug_type == 9:
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Enhances local contrast
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return result


def augment_class(data_dir: Path, class_name: str, target_count: int):
    """Augment images in a class folder to reach target count."""
    class_dir = data_dir / class_name
    
    # Get existing images
    images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
    current_count = len(images)
    
    if current_count >= target_count:
        print(f"{class_name}: {current_count} samples (no augmentation needed)")
        return 0
    
    needed = target_count - current_count
    print(f"{class_name}: {current_count} -> {target_count} (generating {needed} augmented samples)")
    
    generated = 0
    aug_idx = 0
    
    while generated < needed:
        # Pick a random source image
        src_path = random.choice(images)
        img = cv2.imread(str(src_path))
        
        if img is None:
            continue
        
        # Apply augmentation
        aug_img = augment_image(img, aug_idx)
        
        # Save with unique name
        aug_name = f"{src_path.stem}_aug{generated:03d}.png"
        aug_path = class_dir / aug_name
        cv2.imwrite(str(aug_path), aug_img)
        
        generated += 1
        aug_idx += 1
    
    return generated


def main(data_dir: str = "data/labeled_squares", target_min: int = 50):
    """
    Augment underrepresented classes to have at least target_min samples.
    """
    data_dir = Path(data_dir)
    
    classes = [
        "empty",
        "wP", "wN", "wB", "wR", "wQ", "wK",
        "bP", "bN", "bB", "bR", "bQ", "bK",
    ]
    
    print(f"Augmenting classes to minimum {target_min} samples each\n")
    
    total_generated = 0
    
    for cls in classes:
        # Don't augment empty - already have plenty
        if cls == "empty":
            class_dir = data_dir / cls
            count = len(list(class_dir.glob("*.png"))) + len(list(class_dir.glob("*.jpg")))
            print(f"{cls}: {count} samples (skipping - enough data)")
            continue
        
        generated = augment_class(data_dir, cls, target_min)
        total_generated += generated
    
    print(f"\nTotal augmented images generated: {total_generated}")
    
    # Print final distribution
    print("\nFinal class distribution:")
    for cls in classes:
        class_dir = data_dir / cls
        count = len(list(class_dir.glob("*.png"))) + len(list(class_dir.glob("*.jpg")))
        print(f"  {cls}: {count}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Augment chess piece training data")
    parser.add_argument("--data", default="data/labeled_squares", help="Data directory")
    parser.add_argument("--target", type=int, default=50, help="Minimum samples per class")
    
    args = parser.parse_args()
    main(data_dir=args.data, target_min=args.target)
