import cv2
import PlateExtraction

print("Testing OCR functionality...")

# First, let's check if OpticalCharacterRecognition exists
try:
    import OpticalCharacterRecognition
    print("OCR module found!")
    
    # Check what functions are available in OCR
    print("Available functions in OCR module:", [f for f in dir(OpticalCharacterRecognition) if not f.startswith('_')])
    
    # Test with our first detected plate
    plate = cv2.imread('plate_001.jpg')
    if plate is not None:
        print("Loaded plate_001.jpg, running OCR...")
        
        # Try common OCR function names
        if hasattr(OpticalCharacterRecognition, 'ocr'):
            text = OpticalCharacterRecognition.ocr(plate)
            print(f"OCR result: {text}")
        elif hasattr(OpticalCharacterRecognition, 'read_text'):
            text = OpticalCharacterRecognition.read_text(plate)
            print(f"OCR result: {text}")
        else:
            print("No standard OCR function found")
            
    else:
        print("Could not load plate_001.jpg")
        
except ImportError as e:
    print(f"OCR module import error: {e}")
except Exception as e:
    print(f"Error: {e}")
