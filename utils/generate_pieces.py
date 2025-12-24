from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import math

OUT_DIR = Path("tmp/wQ")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 79


def draw_queen(draw, cx, cy, size, fill_color, outline_color):
    """Draw a simple queen shape - crown with 5 points and base"""
    s = size // 2
    
    # Crown points (5 points)
    points = []
    num_points = 5
    for i in range(num_points):
        angle = math.pi / 2 + (i * 2 * math.pi / num_points)
        # Outer point
        px = cx + int(s * 0.8 * math.cos(angle))
        py = cy - int(s * 0.6 * math.sin(angle)) - s // 4
        points.append((px, py))
        # Inner point (valley)
        angle2 = angle + math.pi / num_points
        px2 = cx + int(s * 0.4 * math.cos(angle2))
        py2 = cy - int(s * 0.3 * math.sin(angle2)) - s // 4
        points.append((px2, py2))
    
    # Draw crown
    draw.polygon(points, fill=fill_color, outline=outline_color)
    
    # Draw base (ellipse)
    base_top = cy + s // 4
    base_bottom = cy + s // 2
    draw.ellipse([cx - s//2, base_top, cx + s//2, base_bottom], 
                 fill=fill_color, outline=outline_color)
    
    # Draw neck connecting crown to base
    neck_width = s // 3
    draw.rectangle([cx - neck_width, cy - s//8, cx + neck_width, base_top + 5],
                   fill=fill_color, outline=outline_color)
    
    # Draw circles on crown points
    for i in range(0, len(points), 2):
        px, py = points[i]
        r = 3
        draw.ellipse([px-r, py-r, px+r, py+r], fill=fill_color, outline=outline_color)


# Generate variations
count = 0
variations = [
    # (bg_color, fill_color, outline_color, size_factor)
    ("white", "white", "black", 0.8),
    ("white", "white", "black", 0.7),
    ("white", "white", "black", 0.9),
    ("#d4d4d4", "white", "black", 0.8),  # gray bg
    ("#f0d9b5", "white", "black", 0.8),  # light square color
    ("#b58863", "white", "black", 0.8),  # dark square color
    ("white", "#fffff0", "black", 0.8),  # ivory fill
    ("white", "white", "#333333", 0.8),  # dark gray outline
    ("#e8e8e8", "white", "black", 0.75),
    ("#d4d4d4", "#f8f8f8", "#222222", 0.85),
]

for i, (bg, fill, outline, size_factor) in enumerate(variations):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg)
    draw = ImageDraw.Draw(img)
    
    piece_size = int(IMG_SIZE * size_factor)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    
    draw_queen(draw, cx, cy, piece_size, fill, outline)
    
    out = OUT_DIR / f"wQ_gen_{i:03d}.png"
    img.save(out)
    count += 1

print(f"Generated {count} white queen images")
