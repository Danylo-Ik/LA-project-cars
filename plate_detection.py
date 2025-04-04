import numpy as np
import cv2 as cv
from ultralytics import YOLO

def main():
    """Process a live video stream for vehicle and license plate detection."""
    cap = cv.VideoCapture(0)  # Use 0 for webcam, or replace with video file path

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    frame_count = 0  # Frame counter for performance optimization

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_count += 1

        # Process every 2nd frame for better performance
        if frame_count % 2 == 0:
            vehicles = detect_vehicles(frame)

            for (x1, y1, x2, y2) in vehicles:
                # Draw bounding box around detected vehicle
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(frame, "Vehicle", (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                vehicle_roi = frame[y1:y2, x1:x2]  # Extract vehicle region
                plate_candidates, _ = detect_plate(vehicle_roi)

                for (px, py, pw, ph) in plate_candidates:
                    # Convert ROI plate coordinates to original image coordinates
                    plate_x1, plate_y1 = x1 + px, y1 + py
                    plate_x2, plate_y2 = plate_x1 + pw, plate_y1 + ph

                    # Draw bounding box around detected plate
                    cv.rectangle(frame, (plate_x1, plate_y1), (plate_x2, plate_y2), (0, 0, 255), 2)
                    cv.putText(frame, "Plate", (plate_x1, plate_y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Display processed frame with overlays
            cv.imshow("Detected Vehicles & Plates", frame)

        # Press 'q' to exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def detect_vehicles(frame):
    """Run YOLOv8 to detect vehicles and return bounding boxes."""
    results = model(frame)  # Run YOLOv8 model (returns a list)

    vehicles = []
    for result in results:
        boxes = result.boxes  # Access detected objects
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
            cls = int(box.cls[0])  # Get class ID

            if cls in [2, 3, 5, 7]:  # Class IDs for car, truck, bus, motorcycle
                vehicles.append((x1, y1, x2, y2))

    return vehicles

def detect_plate(vehicle_roi):
    """Detect potential license plates within a vehicle ROI using edge detection & contours."""
    gray = cv.cvtColor(vehicle_roi, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(16,16))
    gray = clahe.apply(gray)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blur, 75, 375)

    # Morphological closing to fill gaps
    closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Find contours
    contours, _ = cv.findContours(closed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    plate_candidates = []
    for contour in contours:
        perimeter = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.05 * perimeter, True)

        x, y, w, h = cv.boundingRect(approx)
        aspect_ratio = float(w) / h
        area = w * h
        solidity = cv.contourArea(contour) / (w * h)

        if 3.5 < aspect_ratio < 5.5 and area > 1000 and solidity > 0.5:
            plate_candidates.append((x, y, w, h))

    return plate_candidates, canny

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")  # Load YOLO model
    main()
