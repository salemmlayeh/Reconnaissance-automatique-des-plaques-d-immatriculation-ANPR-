import cv2
import pytesseract
import os
import sys

# Try to import PlateExtraction
try:
    import PlateExtraction
    print("✅ PlateExtraction module loaded")
except ImportError:
    print("❌ ERROR: PlateExtraction.py not found!")
    print("Make sure PlateExtraction.py is in the same directory")
    sys.exit(1)

# Set tesseract path (adjust for your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

print("=" * 50)
print("LICENSE PLATE DETECTION TEST")
print("=" * 50)

# Check if CarPictures folder exists
if not os.path.exists('CarPictures'):
    print("❌ ERROR: CarPictures folder not found!")
    print("Creating CarPictures folder...")
    os.makedirs('CarPictures', exist_ok=True)
    print("✅ Created CarPictures folder")
    print("Please add images to CarPictures folder first")
    sys.exit(1)

# List available images
images = []
for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
    images.extend([f for f in os.listdir('CarPictures') if f.lower().endswith(ext)])

if not images:
    print("❌ No images found in CarPictures folder!")
    print("Please add some images to CarPictures folder")
    sys.exit(1)

print(f"\n📁 Found {len(images)} image(s) in CarPictures folder:")
for i, img in enumerate(sorted(images), 1):
    print(f"  {i:2d}. {img}")

# Get user input
print("\n" + "-" * 50)
choice = input(f"Enter image number (1-{len(images)}) or filename: ").strip()

if choice.isdigit():
    index = int(choice) - 1
    if 0 <= index < len(images):
        image_path = os.path.join('CarPictures', images[index])
    else:
        print(f"❌ Invalid number. Using first image: {images[0]}")
        image_path = os.path.join('CarPictures', images[0])
else:
    # User entered filename
    if os.path.exists(os.path.join('CarPictures', choice)):
        image_path = os.path.join('CarPictures', choice)
    elif os.path.exists(choice):
        image_path = choice
    else:
        print(f"❌ File not found. Using first image: {images[0]}")
        image_path = os.path.join('CarPictures', images[0])

print(f"\n🔍 Testing with: {image_path}")

# Load and process image
image = cv2.imread(image_path)
if image is None:
    print(f"❌ ERROR: Could not load {image_path}")
    print("Supported formats: JPG, PNG, BMP, TIFF")
    sys.exit(1)

print(f"✅ Image loaded successfully")
print(f"   Dimensions: {image.shape[1]}x{image.shape[0]} pixels")
print(f"   Channels: {image.shape[2]}")

# Plate detection
print("\n🔎 Detecting license plate...")
plate = PlateExtraction.extraction(image)

if plate is not None:
    print("✅ License plate detected!")
    print(f"   Plate size: {plate.shape[1]}x{plate.shape[0]} pixels")
    
    # Character recognition
    print("\n📝 Reading characters with OCR...")
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    
    # Optional: Enhance image for better OCR
    # gray = cv2.equalizeHist(gray)
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(gray, config=custom_config).strip()
    
    # Clean up OCR result
    text = ''.join([c for c in text if c.isalnum()])
    
    if text:
        print(f"✅ License Plate Number: {text}")
    else:
        print("⚠️  No text detected in plate")
        text = "UNKNOWN"
    
    # Save results
    output_name = f"detected_plate_{os.path.basename(image_path)}"
    cv2.imwrite(output_name, plate)
    print(f"💾 Plate saved as: {output_name}")
    
    # Save result to text file
    with open('plate_results.txt', 'a') as f:
        f.write(f"{image_path}: {text}\n")
    
    # Display results
    print("\n👀 Displaying results...")
    print("   Press 'q' to close windows")
    print("   Press 's' to save current display")
    
    # Resize for display if too large
    max_display_size = 800
    h, w = image.shape[:2]
    if max(h, w) > max_display_size:
        scale = max_display_size / max(h, w)
        display_img = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        display_img = image
    
    # Draw text on image
    cv2.putText(display_img, f"Plate: {text}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Original Image with Detection', display_img)
    cv2.imshow('Detected License Plate', plate)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'screenshot_{os.path.basename(image_path)}', display_img)
            print("💾 Screenshot saved!")
    
    cv2.destroyAllWindows()
    
else:
    print("❌ No license plate detected in the image")
    print("\n💡 Tips for better detection:")
    print("   1. Ensure plate is clearly visible")
    print("   2. Try different lighting conditions")
    print("   3. Plate should be reasonably sized in image")

print("\n" + "=" * 50)
print("Test completed!")
print("=" * 50)
