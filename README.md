## 🚗 Automatic Number Plate Recognition

This project implements vehicle recognition, number plate extraction, and speed measurement using a hybrid approach combining machine learning and classical image processing.

<img src="https://github.com/user-attachments/assets/ed7baf91-d617-407a-8015-1c9ef350f466" alt="image" width="66%"/>



## 📁 Files
```
├── test images/              # Input test images
├── processed_corners/        # Output images with detected plate corners
├── temp/                     # Intermediate processing outputs
├── plate_detection_img.py    # Plate detection from images
├── plate_detection_video.py  # Plate detection from videos
├── decomposition.py          # SVD-based denoising
├── edge_detection.py         # Edge detection logic
├── license_plate_detector.pt # YOLOv8 model for detecting number plates
├── yolov8n.pt                # YOLOv8 model for detecting vehicles
```

## Features

1. **License Plate Detection**:
   - Detects license plates in images and videos using YOLOv8.
   - Pre-trained YOLOv8 models are stored in `license_plate_detector.pt` and `yolov8n.pt`.

2. **Image Processing**:
   - **Denoising**: Implemented in [`decomposition.py`](decomposition.py) using Singular Value Decomposition (SVD).
   - **Edge Detection**: Algorithms for edge detection are implemented in [`edge_detection.py`](edge_detection.py).

3. **Processed Outputs**:
   - Detected plates are saved in the `processed_corners/` directory.
   - Intermediate processing results (blurred, denoised, and edge-detected images) are stored in the `temp/` directory.

## 🛠 Setup

<pre>
git clone https://github.com/yourusername/your-repo.git
cd your-repo
pip install -r requirements.txt
</pre>

## 📸 Usage

### 1. Detect License Plates in Images

Run the `plate_detection_img.py` script to detect license plates in images from the `test images/` directory.

### 2. Detect License Plates in Videos (Not Finished)
Run the `plate_detection_video.py` script to detect license plates in video streams.

## 🚧 To Do

- [ ] Improve plate localization with perspective correction
- [ ] Implement OCR for extracting text from license plates
- [ ] Implement speed estimation
- [ ] Finish video stream support
- [ ] Сombine everything into a working system
