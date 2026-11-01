import cv2
import pytesseract
import os

# Set tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

print("Testing OCR on all detected plates...")
plate_files = [f for f in os.listdir('.') if f.startswith('plate_') and f.endswith(('.jpg', '.jpeg', '.png'))]

for plate_file in plate_files:
    print(f"\n--- {plate_file} ---")
    plate = cv2.imread(plate_file)
    
    if plate is not None:
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        
        # OCR with license plate optimized config
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(gray, config=custom_config).strip()
        
        print(f"Detected: '{text}'")
        
        # Show the plate and text
        cv2.imshow(f'Plate: {text}', plate)
        cv2.waitKey(500)  # Show for 0.5 seconds
    else:
        print("Could not load plate")

cv2.destroyAllWindows()
print("\nOCR testing completed!")
