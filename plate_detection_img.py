import numpy as np
import cv2 as cv
from ultralytics import YOLO
from decomposition import denoise_image
from ocr_module import binarize, connected_components, extract_characters, recognize_character
from data_process import load_dataset

def main(image_path):
    """Process a single image for license plate detection and save character collage."""
    import os

    os.makedirs("temp", exist_ok=True)

    image = cv.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    plates = plate_model(image)[0]
    plate_roi = []

    for plate in plates.boxes:
        x1, y1, x2, y2 = map(int, plate.xyxy[0])
        x1 -= 10
        y1 -= 10
        x2 += 10
        y2 += 10
        plate_roi.append((x1, y1, x2, y2))

    i = 0
    for plate in plate_roi:
        corners = detect_corners(image[plate[1]:plate[3], plate[0]:plate[2]])
        if corners is not None:
            src = np.array([[pt[0] + plate[0], pt[1] + plate[1]] for pt in corners], dtype=np.float32)
            for corner in src:
                cv.circle(image, (int(corner[0]), int(corner[1])), 3, (0, 255, 0), -1)

            dst = np.array([[0, 0], [600, 0], [0, 200], [600, 200]], dtype=np.float32)
            M = get_perspective_transform_matrix(src, dst)
            warped = cv.warpPerspective(image, M, (600, 200))
            warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)

            k = int(min(warped.shape) * 0.15)
            warped = denoise_image(warped, k)

            cv.imwrite(f"temp/warped_plate_{i}.jpg", warped)

        binarized_plate = binarize(warped)
        cv.imwrite(f"temp/binarized_plate_{i}.jpg", binarized_plate * 255)

        components = connected_components(binarized_plate)
        recognized_str = ""
        chars = extract_characters(binarized_plate, components, padding=2, target_size=(28, 28))

        collage_chars = []

        for _, char_img in enumerate(chars):
            char_img_vis = (char_img * 255).astype(np.uint8)
            collage_chars.append(char_img_vis)

            recognized = recognize_character(char_img, dataset)
            recognized_str += recognized

        if collage_chars:
            collage = np.hstack(collage_chars)
            collage_resized = cv.resize(collage, (collage.shape[1] * 2, collage.shape[0] * 2), interpolation=cv.INTER_NEAREST)
            cv.imwrite(f"temp/segmented_characters_{i}.jpg", collage_resized)

        cv.putText(image, recognized_str, (plate[0], plate[1] - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv.rectangle(image, (plate[0], plate[1]), (plate[2], plate[3]), (255, 0, 0), 2)

        i += 1

    cv.imshow("Detected Plates", image)
    cv.waitKey(0)
    cv.destroyAllWindows()



def detect_vehicles(frame):
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


def detect_corners(vehicle_roi):
    """Detect potential license plates within a vehicle ROI using edge detection & contours."""
    binary = cv.cvtColor(vehicle_roi, cv.COLOR_BGR2GRAY)
    binary = cv.GaussianBlur(binary, (5, 5), 0)
    # cv.imwrite("temp/blurred.jpg", binary)

    k = int(min(binary.shape) * 0.25)
    print(f"Using level {k} for SVD denoising")
    binary = denoise_image(binary, k)
    cv.imwrite("temp/denoised.jpg", binary)

    binary = cv.Canny(binary, 75, 375)
    # binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    binary = cv.dilate(binary, None, iterations=1)
    # binary = cv.morphologyEx(binary, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cv.imwrite("temp/edges.jpg", binary)
    contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    best_corners = None
    max_area = 0

    for contour in contours:
        perimeter = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.05 * perimeter, True)

        if len(approx) == 4:
            area = cv.contourArea(approx)
            if area > max_area:
                max_area = area
                best_corners = approx

    if best_corners is None:
        return None

    corners = [tuple(point[0]) for point in best_corners]

    # sort the corners in a standard order (top left, top right, bottom left, bottom right)
    corners = sorted(corners, key=lambda p: (p[1], p[0]))
    if corners[0][0] > corners[1][0]:
        corners[0], corners[1] = corners[1], corners[0]
    if corners[2][0] > corners[3][0]:
        corners[2], corners[3] = corners[3], corners[2]

    return corners

def get_perspective_transform_matrix(source, destination):
    """Get the perspective transform matrix from source to destination points."""
    A = []
    b = []
    for (x, y), (u, v) in zip(source, destination):
        A.append([x, y, 1, 0, 0, 0, -x * u, -y * u])
        A.append([0, 0, 0, x, y, 1, -x * v, -y * v])
        b.append(u)
        b.append(v)
    A = np.array(A, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    x = np.linalg.lstsq(A, b, rcond=None)[0]

    perspective_matrix = np.array([
        [x[0], x[1], x[2]],
        [x[3], x[4], x[5]],
        [x[6], x[7], 1.0]
    ], dtype=np.float32)

    return perspective_matrix

if __name__ == "__main__":
    image_path = 'test images/red car.jpg'
    # car_model = YOLO('yolov8n.pt')
    plate_model = YOLO('license_plate_detector.pt')
    dataset = load_dataset()
    main(image_path)
