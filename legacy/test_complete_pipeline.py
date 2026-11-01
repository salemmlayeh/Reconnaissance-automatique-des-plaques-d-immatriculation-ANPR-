import cv2
import PlateExtraction
import pytesseract

# Set tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

print("=== COMPLETE ANPR PIPELINE TEST ===")

# Test with original car image
image = cv2.imread('CarPictures/001.jpg')
print(f"1. Original image loaded: {image.shape}")

# Step 1: Plate Detection
plate = PlateExtraction.extraction(image)
print(f"2. Plate detected: {plate is not None}")

if plate is not None:
    # Step 2: Character Recognition
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(gray, config=custom_config).strip()
    
    print(f"3. License Plate Number: {text}")
    
    # Display results
    cv2.imshow('Original Car Image', image)
    cv2.imshow('Detected License Plate', plate)
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("3. No plate detected")
