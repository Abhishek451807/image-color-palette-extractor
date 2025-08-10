# image_palette_extractor_github.py

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys
import os

def rgb_to_hex(color):
    """Convert RGB to HEX."""
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def extract_colors(image_path, num_colors=5):
    """Extract dominant colors from image."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize for faster processing
    resized_img = cv2.resize(img, (600, 400))
    reshaped_img = resized_img.reshape(-1, 3)

    # KMeans clustering
    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(reshaped_img)
    colors = kmeans.cluster_centers_
    labels = np.bincount(kmeans.labels_)

    # Sort colors by frequency
    sorted_colors = [color for _, color in sorted(zip(labels, colors), reverse=True)]
    return sorted_colors

def create_github_palette(colors, save_path="palette_with_codes.png"):
    """Create a palette image with HEX codes for GitHub."""
    num_colors = len(colors)
    swatch_height = 100
    text_height = 40
    width_per_color = 200

    # Create a blank image
    palette_img = np.ones((swatch_height + text_height, num_colors * width_per_color, 3), dtype=np.uint8) * 255

    for i, color in enumerate(colors):
        rgb = tuple(map(int, color))
        hex_code = rgb_to_hex(color)

        # Draw the swatch
        palette_img[:swatch_height, i * width_per_color:(i + 1) * width_per_color] = rgb

        # Add the HEX code text
        cv2.putText(
            palette_img, hex_code,
            (i * width_per_color + 20, swatch_height + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA
        )

    # Save and show the image
    cv2.imwrite(save_path, cv2.cvtColor(palette_img, cv2.COLOR_RGB2BGR))
    print(f"✅ GitHub-ready palette saved as {save_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python image_palette_extractor_github.py <image_path> [num_colors]")
        sys.exit(1)

    image_path = sys.argv[1]
    num_colors = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    if not os.path.exists(image_path):
        print(f"❌ Error: File '{image_path}' not found.")
        sys.exit(1)

    print("🎨 Extracting colors...")
    colors = extract_colors(image_path, num_colors)

    print("\nDominant Colors:")
    for i, color in enumerate(colors):
        print(f"Color {i+1}: RGB={tuple(map(int, color))} HEX={rgb_to_hex(color)}")

    create_github_palette(colors)
