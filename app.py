import easyocr
import cv2
import pandas as pd
import os
import numpy as np
import unicodedata
from datetime import datetime

# -----------------------------
# Setup folders
# -----------------------------
if not os.path.exists("plates"):
    os.makedirs("plates")

csv_file = "plate_log.csv"

if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=["Plate Number", "Timestamp", "Image Path", "Confidence"])
    df.to_csv(csv_file, index=False)

# -----------------------------
# Load OCR + Cascade
# -----------------------------
print("Initializing EasyOCR...")
reader = easyocr.Reader(['en', 'hi'], gpu=False)

cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
print(f"Loading cascade from: {cascade_path}")
plate_cascade = cv2.CascadeClassifier(cascade_path)

if plate_cascade.empty():
    print("Error loading cascade classifier. Please ensure OpenCV is correctly installed.")
    exit(1)

detected_plates = set()

print("✅ Starting webcam....")
print("❌ Press 'q' on the keyboard to exit")

# -----------------------------
# LIVE LOOP
# -----------------------------
# Initialize VideoCapture (0 is usually the built-in webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in plates:
        # --- 1. Pad the bounding box slightly ---
        # Often the cascade crops too tightly causing letters on edges to get cut off or misread.
        pad_y = int(h * 0.15)
        pad_x = int(w * 0.15)
        y1, y2 = max(0, y - pad_y), min(frame.shape[0], y + h + pad_y)
        x1, x2 = max(0, x - pad_x), min(frame.shape[1], x + w + pad_x)
        plate_img = frame[y1:y2, x1:x2]
        
        # --- 2. Aggressive Preprocessing for OCR ---
        # Scale up significantly
        plate_img_resized = cv2.resize(plate_img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        # Convert to Grayscale
        plate_img_gray = cv2.cvtColor(plate_img_resized, cv2.COLOR_BGR2GRAY)
        # Denoise before thresholding
        plate_img_blur = cv2.GaussianBlur(plate_img_gray, (5, 5), 0)
        # Extreme contrast: Otsu's Thresholding to make text purely black and white
        _, plate_img_thresh = cv2.threshold(plate_img_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Force easyocr to only return alphanumeric characters
        result = reader.readtext(plate_img_thresh, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        if result:
            # Drawing a rectangle around the detected plate
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            
            # Combine all extracted text chunks
            plate_text = "".join([res[1] for res in result])
            # Translate Devanagari numerals to standard numerals
            translation_table = str.maketrans("०१२३४५६७८९", "0123456789")
            plate_text = plate_text.translate(translation_table)
            
            # Convert any styled alphabets/numerals to normal ASCII equivalents
            plate_text = unicodedata.normalize('NFKD', plate_text).encode('ascii', 'ignore').decode('ascii')
            
            plate_text = ''.join(filter(str.isalnum, plate_text)).upper().replace(" ", "")
            confidence = result[0][-1]

            cv2.putText(frame, f"{plate_text} ({confidence:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            if plate_text not in detected_plates and len(plate_text) >= 6:
                detected_plates.add(plate_text)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Ensure the 'plates' directory exists before saving (handled by setup loop above)
                img_path = f"plates/{plate_text}_{int(datetime.now().timestamp())}.jpg"
                
                cv2.imwrite(img_path, plate_img)

                new_row = pd.DataFrame([{
                    "Plate Number": plate_text,
                    "Timestamp": timestamp,
                    "Image Path": img_path,
                    "Confidence": confidence
                }])

                new_row.to_csv(csv_file, mode='a', header=False, index=False)

                print(f"🚗 Detected: {plate_text} (Conf: {confidence:.2f})")

    # Display the live feed
    cv2.imshow('License Plate Detection', frame)

    # Exit condition: pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Application finished.")