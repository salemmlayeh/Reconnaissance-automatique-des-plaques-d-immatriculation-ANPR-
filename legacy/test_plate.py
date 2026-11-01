import cv2
import PlateExtraction

print("Testing plate extraction...")

# Try to load a car image
try:
    image = cv2.imread('CarPictures/001.jpg')
    if image is None:
        print("ERROR: Could not load CarPictures/001.jpg")
        print("Make sure the file exists in the CarPictures folder")
    else:
        print("Image loaded successfully!")
        print(f"Image size: {image.shape}")
        
        # Use the extraction function
        plate = PlateExtraction.extraction(image)
        
        if plate is not None:
            print("SUCCESS: License plate detected!")
            cv2.imwrite('detected_plate.jpg', plate)
            print("Saved as 'detected_plate.jpg'")
        else:
            print("No license plate found in the image")
            
except Exception as e:
    print(f"Error: {e}")
