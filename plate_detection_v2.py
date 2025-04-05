import numpy as np
import cv2 as cv
from ultralytics import YOLO
from decomposition import denoise_image

def main(image_path):
    """Process a single image for license plate detection."""
    # Read the image
    image = cv.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    # vehicles = detect_vehicles(image)
    plates = plate_model(image)[0]

    # for (x1, y1, x2, y2) in vehicles:
    #     #draw bounding box around detected vehicle
    #     cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv.putText(image, "Vehicle", (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    plate_roi = []
    for plate in plates.boxes:
        x1, y1, x2, y2 = map(int, plate.xyxy[0])
        # pad the bounding box
        x1 -= 10
        y1 -= 10
        x2 += 10
        y2 += 10
        plate_roi.append((x1, y1, x2, y2))
    
    for plate in plate_roi:
        corners = detect_corners(image[plate[1]:plate[3], plate[0]:plate[2]])
        if corners is not None:
            # show the detected corners
            for corner in corners:
                corner_x, corner_y = corner
                cv.circle(image, (corner_x + plate[0], corner_y + plate[1]), 3, (255, 0, 0), -1)

    # Display the results  
    cv.imshow("Detected Plates", image)

    cv.waitKey(0)
    cv.destroyAllWindows()


def detect_vehicles(frame):
    """Run YOLOv8 to detect vehicles and return bounding boxes."""
    # set confidence threshold
    results = car_model(frame)

    vehicles = []
    for result in results:
        boxes = result.boxes  # Access detected objects
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
            cls = int(box.cls[0])  # Get class id
            conf = float(box.conf.item())

            if cls in [2, 3, 5, 7] and conf > 0.75:  # Class id for car, truck, bus, motorcycle
                vehicles.append((x1, y1, x2, y2))

    return vehicles


def detect_corners(vehicle_roi):
    """Detect potential license plates within a vehicle ROI using edge detection & contours."""
    binary = cv.cvtColor(vehicle_roi, cv.COLOR_BGR2GRAY)
    binary = cv.GaussianBlur(binary, (5, 5), 0)
    # clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(16,16))
    # binary = clahe.apply(binary)
    binary = cv.Canny(binary, 75, 375)
    # binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    binary = cv.dilate(binary, None, iterations=1)
    # binary = cv.morphologyEx(binary, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cv.imshow("Contours", binary)

    best_corners = None
    max_area = 0

    for contour in contours:
        perimeter = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.05 * perimeter, True)

        if len(approx) == 4:  # Plate should have 4 corners
            area = cv.contourArea(approx)
            if area > max_area:
                max_area = area
                best_corners = approx

    if best_corners is None:
        return None  # No valid plate found

    # Convert to list of (x, y) tuples
    corners = [tuple(point[0]) for point in best_corners]

    # Sort the corners in a standard order: (top-left, top-right, bottom-left, bottom-right)
    corners = sorted(corners, key=lambda p: (p[1], p[0]))  # Sort by y first, then x
    if corners[0][0] > corners[1][0]:  # Ensure TL is first
        corners[0], corners[1] = corners[1], corners[0]
    if corners[2][0] > corners[3][0]:  # Ensure BL is before BR
        corners[2], corners[3] = corners[3], corners[2]

    return corners


if __name__ == "__main__":
    image_path = 'test images/red car.jpg'
    car_model = YOLO('yolov8n.pt')
    plate_model = YOLO('license_plate_detector.pt')
    main(image_path)