# Face Recognition System

A Python-based face recognition system that can identify faces from images and real-time video feed using face_recognition library.

## Project Structure
```
project/
├── face_recognition_system.py  # Main code
├── known_faces/               # Directory with known face images
│   ├── person1.jpg
│   ├── person2.jpg
│   └── ...
├── face_database.csv         # CSV with face information
└── test_images/             # Images to test recognition on
```

## Requirements
- Python 3.6+
- OpenCV (cv2)
- face_recognition
- numpy
- pandas

## Prerequisites
### Windows Users
Before installing the required packages, you need to install Visual Studio Build Tools with C++ support:

1. Download Visual Studio Build Tools from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run the installer
3. In the installer, select "Desktop development with C++"
4. Click Install and wait for the installation to complete
5. Restart your computer after installation

This step is required because the face_recognition library depends on dlib, which needs to be compiled from source on Windows. The compilation process requires Visual Studio's C++ build tools.

## Installation
1. Clone this repository
2. Install the required packages:
   ```bash
   pip install face_recognition opencv-python numpy pandas
   ```

## Usage
1. Add known face images to the `known_faces/` directory
2. Update face information in `face_database.csv`
3. Run the system:
   ```python
   python face_recognition_system.py
   ```

### Features
- Face recognition from images
- Real-time face recognition from webcam
- Face database management
- Match percentage calculation
- CSV-based face information storage

## Note
Make sure to have good quality, well-lit face images in the `known_faces/` directory for better recognition accuracy.
