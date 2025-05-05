
# A processing platform for pedestrian surveillance with integrated object detection, tracking, and pose estimation

## Overview
The goal of this project is to build an efficient and robust video processing system to enhance intelligence and safety in surveillance applications. The system should not only achieve detection and tracking of people in complex real-world scenarios with high accuracy but also analyze the pose of the people in various aspects. Specifically, the system will provide a comprehensive and useful set of features. These include 1. accurate bounding boxes for recognized objects such as persons and cars, clearly identifying the position of the object in the video frame. 2. accurate track drawing to go to a complete presentation of the movement path. 3.reasonable speed estimation. 4. reliable fall detection to find potential safety risks in a timely manner.5. effective intuitive information including counts and alerts.


## Features

* **Interactive Web Interface**: User-friendly interface via Streamlit.
* **Object Detection**: YOLOv11 for real-time detection of persons and cars.
* **Object Tracking**: SORT algorithm: Kalman Filters and Hungarian algorithm
* **Fall Detection**: Pose estimation module to identify person falls.
* **Speed Estimation**: Calculates speed of person and cars.


## Installation


### Setting up the Environment

```
pip install streamlit opencv-python torch torchvision numpy filterpy scipy lap ultralytics
```

## Usage

Run the application:

```
streamlit run client.py
```

## How to Use the Interface

* **Upload a Video**: Select and upload a video file (`mp4`, `avi`, or `mov`).
* **Click to process video**: Click the "Process video" button to perform detection and tracking.
* **View Results**: The processed video will display.
* **Download processed video**: Option to download the final processed video.

## Project Structure

```
project/
├── client.py            # Streamlit interface and main application logic
├── detect.py            # Object detection module (YOLO)
├── fall.py              # Fall detection using pose estimation
├── util.py              # Utility functions for drawing and calculation
├── SORT.py              # SORT tracking algorithm
├── coco.class           # Class names for YOLO
├── yolo11s.pt           # YOLO model for object detection
├── yolo11s-pose.pt      # YOLO pose model for fall detection
├── test_video           # Test video for testing
├── processed_video      # Processed video
└── README.md
```

## License
This project is for educational use only under the University coursework policy.
