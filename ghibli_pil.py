import os
from PIL import Image, ImageEnhance

def apply_ghibli_filter(image_path, output_path):
    img = Image.open(image_path)
    # Make colors pastel like Ghibli
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(0.7)  # reduce saturation
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.1)  # increase brightness
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)  # increase contrast
    img.save(output_path)

if __name__ == "__main__":
    images_dir = "images"
    output_dir = "images_ghibli"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and filename != "birthday-video.mp4":
            content_path = os.path.join(images_dir, filename)
            output_path = os.path.join(output_dir, filename)
            print(f"Processing {filename}")
            apply_ghibli_filter(content_path, output_path)
            print(f"Saved to {output_path}")