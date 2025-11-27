import cv2
import os

def cartoonize(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read {image_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    cv2.imwrite(output_path, cartoon)

if __name__ == "__main__":
    images_dir = "images"
    output_dir = "images_ghibli"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and filename != "birthday-video.mp4":
            content_path = os.path.join(images_dir, filename)
            output_path = os.path.join(output_dir, filename)
            print(f"Processing {filename}")
            cartoonize(content_path, output_path)
            print(f"Saved to {output_path}")