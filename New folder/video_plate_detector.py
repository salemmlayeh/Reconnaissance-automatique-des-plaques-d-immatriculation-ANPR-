import cv2
import PlateExtraction
import pytesseract
import time
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

print("🎥 VIDEO LICENSE PLATE DETECTOR")
print("=" * 50)
print("Options:")
print("1. Use laptop camera (default)")
print("2. Use video file (if available)")
print("=" * 50)

# Ask for video source choice
choice = input("Enter choice (1 or 2, press Enter for camera): ").strip()

if choice == "2":
    # Look for video files
    video_files = [f for f in os.listdir('.') if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if video_files:
        print("Available video files:")
        for i, f in enumerate(video_files):
            print(f"  {i+1}. {f}")
        vid_choice = input("Enter number or filename: ").strip()
        if vid_choice.isdigit() and 1 <= int(vid_choice) <= len(video_files):
            video_path = video_files[int(vid_choice)-1]
        else:
            video_path = vid_choice
    else:
        print("No video files found. Using camera instead.")
        video_path = 0
else:
    video_path = 0  # Default to camera

# Initialize video capture
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Could not open video source")
    exit()

print("✅ Video source opened successfully")
print("Controls:")
print("• Press 'Q' to quit")
print("• Press 'P' to pause/resume")
print("• Press 'S' to save current frame")
print("• Press 'D' to save detected plate")
print("=" * 50)

frame_count = 0
detection_count = 0
paused = False
unique_plates = []

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("❌ End of video or cannot read frame")
            break
        
        frame_count += 1
        display_frame = frame.copy()
        current_time = time.time()
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Process every 10th frame for performance
        if frame_count % 10 == 0:
            plate = PlateExtraction.extraction(frame)
            
            if plate is not None:
                # OCR on detected plate
                gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                text = pytesseract.image_to_string(gray_plate, config=custom_config).strip()
                
                if text and len(text) >= 4:
                    detection_count += 1
                    
                    # Add to unique plates if new
                    if text not in unique_plates:
                        unique_plates.append(text)
                        # Auto-save new unique plates
                        filename = f"detected_plate_{text}.jpg"
                        cv2.imwrite(filename, plate)
                        print(f"✅ NEW PLATE: {text} (saved as {filename})")
                    
                    # Draw detection info on frame
                    cv2.rectangle(display_frame, (50, 30), (w-50, 100), (0, 255, 0), -1)
                    cv2.putText(display_frame, f"PLATE: {text}", (60, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                    
                    # Draw scanning area
                    cv2.rectangle(display_frame, (w//4, h//3), (3*w//4, 2*h//3), (0, 255, 0), 2)
        
        # Display frame info
        status = "PAUSED" if paused else "LIVE"
        cv2.putText(display_frame, f"Status: {status}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Frames: {frame_count}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Detections: {detection_count}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Unique: {len(unique_plates)}", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Controls help
        cv2.putText(display_frame, "Q:Quit P:Pause S:SaveFrame D:SavePlate", 
                   (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display frame
    cv2.imshow('Video Plate Detector', display_frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
        print("⏸️  Paused" if paused else "▶️  Resumed")
    elif key == ord('s'):
        # Save current frame
        filename = f"frame_{frame_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Frame saved: {filename}")
    elif key == ord('d') and not paused:
        # Manually save current detection
        plate = PlateExtraction.extraction(frame)
        if plate is not None:
            filename = f"manual_detect_{frame_count}.jpg"
            cv2.imwrite(filename, plate)
            print(f"💾 Plate saved: {filename}")

# Cleanup
cap.release()
cv2.destroyAllWindows()

print(f"\\n📊 SESSION SUMMARY:")
print(f"Total frames processed: {frame_count}")
print(f"License plates detected: {detection_count}")
print(f"Unique plates found: {len(unique_plates)}")
if unique_plates:
    print("Detected plates:", ", ".join(unique_plates))
print("🎉 Video detection completed!")
