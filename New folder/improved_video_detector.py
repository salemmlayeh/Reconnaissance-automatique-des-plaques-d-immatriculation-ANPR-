import cv2
import PlateExtraction
import pytesseract
import time
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

print("🎥 IMPROVED VIDEO LICENSE PLATE DETECTOR")
print("=" * 50)

# Use camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open camera")
    exit()

print("✅ Camera opened successfully")
print("Controls: Q=Quit, P=Pause, S=Save frame")

frame_count = 0
detection_count = 0
paused = False
unique_plates = []

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Process every 8th frame
        if frame_count % 8 == 0:
            plate = PlateExtraction.extraction(frame)
            
            if plate is not None:
                # Preprocess the plate for better OCR
                gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                
                # Apply multiple preprocessing techniques
                # 1. Increase contrast
                contrast = cv2.convertScaleAbs(gray_plate, alpha=1.5, beta=0)
                # 2. Apply threshold
                _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # 3. Remove noise
                denoised = cv2.medianBlur(thresh, 3)
                
                # Try multiple OCR configurations
                configs = [
                    r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    r'--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                ]
                
                best_text = ""
                for config in configs:
                    text = pytesseract.image_to_string(denoised, config=config).strip()
                    # Prefer longer, more plausible text
                    if len(text) > len(best_text) and 3 <= len(text) <= 12:
                        best_text = text
                
                if best_text and len(best_text) >= 3:
                    detection_count += 1
                    
                    # Only accept if it looks like a license plate (mix of letters and numbers)
                    has_letters = any(c.isalpha() for c in best_text)
                    has_numbers = any(c.isdigit() for c in best_text)
                    
                    if has_letters and has_numbers:  # More likely to be a real plate
                        if best_text not in unique_plates:
                            unique_plates.append(best_text)
                            filename = f"plate_{best_text}.jpg"
                            cv2.imwrite(filename, plate)
                            # Also save the preprocessed version
                            cv2.imwrite(f"processed_{filename}", denoised)
                            print(f"✅ CONFIDENT: {best_text} (saved)")
                            
                            # Draw confident detection
                            cv2.rectangle(display_frame, (50, 30), (w-50, 100), (0, 255, 0), -1)
                            cv2.putText(display_frame, f"PLATE: {best_text}", (60, 80), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                        else:
                            # Show but don't save duplicates
                            cv2.putText(display_frame, f"Known: {best_text}", (w//2-100, 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    else:
                        # Show low confidence detections in yellow
                        cv2.putText(display_frame, f"Maybe: {best_text}", (w//2-100, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        print(f"⚠️  LOW CONFIDENCE: {best_text}")
        
        # Display info
        cv2.putText(display_frame, f"Frame: {frame_count}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Detections: {detection_count}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Unique: {len(unique_plates)}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(display_frame, "Scanning... Point camera at license plates", 
                   (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Improved Plate Detector', display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
        print("⏸️  Paused" if paused else "▶️  Resumed")

cap.release()
cv2.destroyAllWindows()

print(f"\\n📊 FINAL RESULTS:")
print(f"Frames: {frame_count}, Detections: {detection_count}")
print(f"High-confidence plates: {len(unique_plates)}")
if unique_plates:
    print("Confirmed plates:", unique_plates)
