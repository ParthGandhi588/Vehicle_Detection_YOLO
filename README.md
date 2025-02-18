# Vehicle Detection System

A real-time vehicle detection and distance estimation system using YOLOv8 and OpenCV. This system can detect cars, buses, and trucks in video streams while estimating their distance from the camera.

## Features

- Real-time vehicle detection using YOLOv8
- GPU acceleration support with CUDA
- Distance estimation for detected vehicles
- Support for multiple vehicle types (cars, buses, trucks)
- Batch processing for improved performance
- Visual output with bounding boxes and labels
- Distance measurements display
- Video output saving capability

## Requirements

- Python 3.8 or higher
- NVIDIA GPU (optional, for CUDA acceleration)
- See `requirements.txt` for Python package dependencies

## Installation

1. Clone this repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the YOLOv8 model weights (this will be done automatically on first run)

## Usage

1. Update the video path in `main()` function of `vehicle.py`:
```python
video_path = 'path/to/your/video.mp4'
```

2. Run the detection script:
```bash
python vehicle.py
```

The script will:
- Process the input video
- Display the detection results in real-time
- Save the processed video as 'output_detection.mp4'
- Press 'q' to quit the application

## Configuration

You can modify the following parameters in the `VehicleDetector` class:

- `KNOWN_WIDTH`: Average vehicle width in meters (default: 1.8m)
- `FOCAL_LENGTH`: Camera focal length (needs calibration)
- `vehicle_classes`: Dictionary mapping class IDs to vehicle types
- Detection confidence threshold (default: 0.5)

## Class Structure

### VehicleDetector

Main class that handles all detection and processing functionality:

- `__init__()`: Initializes the YOLO model and sets up GPU if available
- `estimate_distance()`: Calculates distance based on pixel width
- `process_video()`: Processes video frames and handles detection/visualization

## Output

The system provides:
- Real-time visual display of detections
- Bounding boxes around detected vehicles
- Vehicle type and confidence score labels
- Estimated distance measurements
- Processed video saved as MP4 file

## Performance Notes

- GPU acceleration is automatically enabled if CUDA is available
- Batch processing is implemented for improved performance
- Adjust batch_size parameter based on your GPU memory

## Limitations

- Distance estimation requires camera calibration for accuracy
- Performance depends on hardware capabilities
- Distance estimation assumes flat terrain
- Fixed average vehicle width for distance calculations

## Future Improvements

- Add camera calibration module
- Implement tracking for continuous vehicle monitoring
- Add support for more vehicle types
- Improve distance estimation accuracy
- Add speed estimation capability