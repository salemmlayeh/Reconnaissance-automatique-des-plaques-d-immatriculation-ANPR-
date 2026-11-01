import cv2
import pytesseract
import os

# Try to find Tesseract path automatically
def find_tesseract_path():
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

# Set tesseract path
tesseract_path = find_tesseract_path()
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print(f"Tesseract found at: {tesseract_path}")
else:
    print("Tesseract not found. Please install it first.")
    exit(1)

print("Testing OCR with detected plates...")

# Test with our first plate
plate = cv2.imread('plate_001.jpg')
if plate is not None:
    print("Plate loaded, running OCR...")
    
    # Convert to grayscale for better OCR
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    
    # Try different OCR configurations
    try:
        # Basic OCR
        text = pytesseract.image_to_string(gray)
        print(f"Basic OCR result: '{text.strip()}'")
        
        # OCR with custom config for license plates
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text_clean = pytesseract.image_to_string(gray, config=custom_config)
        print(f"Clean OCR result: '{text_clean.strip()}'")
        
    except Exception as e:
        print(f"OCR error: {e}")
else:
    print("Could not load plate image")
