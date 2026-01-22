import cv2
import PlateExtraction
import os

print("Testing all images in CarPictures...")
image_files = os.listdir('CarPictures')

for img_file in image_files:
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        print(f"\n--- Testing {img_file} ---")
        image_path = os.path.join('CarPictures', img_file)
        image = cv2.imread(image_path)
        
        if image is not None:
            print(f"Image loaded: {image.shape}")
            plate = PlateExtraction.extraction(image)
            
            if plate is not None:
                print("SUCCESS: License plate detected!")
                output_name = f"plate_{img_file}"
                cv2.imwrite(output_name, plate)
                print(f"Saved as '{output_name}'")
            else:
                print("No license plate found")
        else:
            print("Failed to load image")
