import numpy as np
import cv2 as cv
from ultralytics import YOLO


def main(image_path):
    """Process a single image for license plate detection."""
    image = cv.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    vehicles = detect_vehicles(image)

    for (x1, y1, x2, y2) in vehicles:
        vehicle_roi = image[y1:y2, x1:x2]
        plate_candidates, canny = detect_plate(vehicle_roi)

        for (px, py, pw, ph) in plate_candidates:
            plate_x1, plate_y1 = x1 + px, y1 + py
            plate_x2, plate_y2 = plate_x1 + pw, plate_y1 + ph

            cv.rectangle(image, (plate_x1, plate_y1),
                         (plate_x2, plate_y2), (0, 0, 255), 2)
            cv.putText(image, "Plate", (plate_x1, plate_y1 - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv.imshow("Detected Vehicles", image)

    cv.waitKey(0)
    cv.destroyAllWindows()


def detect_vehicles(frame):
    """Run YOLOv8 to detect vehicles and return bounding boxes."""
    results = car_model(frame)

    vehicles = []
    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf.item())

            if cls in [2, 3, 5, 7] and conf > 0.75:
                vehicles.append((x1, y1, x2, y2))

    return vehicles


def detect_plate(vehicle_roi):
    """Detect potential license plates within a vehicle ROI using edge detection & contours."""
    gray = cv.cvtColor(vehicle_roi, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
    gray = clahe.apply(gray)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blur, 75, 375)

    closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))

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
    image_path = 'test images/cars parking.jpg'
    car_model = YOLO('yolov8n.pt')
    main(image_path)
