# Face Detector

A robust real-time face detection application built with Python and OpenCV. This project uses a pre-trained Deep Neural Network (ResNet SSD) for accurate face detection and includes features like face blurring and object tracking.

## Features

- **Multiple Operation Modes**:
  - **Image**: Detect faces in static images.
  - **Video**: Process video files with tracking for performance optimization.
  - **Webcam**: Real-time face detection from your camera.
- **Privacy Mode**: Option to automatically blur detected faces.
- **High Performance**: Uses an SSD (Single Shot Detector) model with a ResNet-10 architecture.
- **Tracking**: Integrates a correlation tracker (via dlib or OpenCV) to maintain face identities and improve processing speed by reducing the frequency of full DNN inference.
- **Configurable**: Easy-to-adjust confidence thresholds and settings.

## Prerequisites

- **Python**: >= 3.11
- **uv**: An extremely fast Python package installer and resolver.

## Installation

This project manages dependencies using `uv`.

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd face_detector
    ```

2.  **Install dependencies:**

    ```bash
    uv sync
    ```

    Or manually install requirements if not using the lock file:

    ```bash
    uv pip install opencv-python numpy imutils
    ```

3.  **Model Files**:
    Ensure you have the required Caffe model files in a `models/` directory:
    - `deploy.prototxt.txt`
    - `res10_300x300_ssd_iter_140000.caffemodel`

    _Note: These files are standard part of OpenCV's face detection module examples._

## Usage

Run the main application using `uv`:

```bash
uv run main.py
```

Or if you have your virtual environment activated:

```bash
python main.py
```

### Interactive Menu

Upon running, you will be prompted to choose a mode:

```text
=== MULTIPLE FACE DETECTOR ===
Select mode:
1 - Image
2 - Video
3 - Webcam Live Stream
Enter choice [1/2/3]:
```

1.  **Image Mode**:
    - Prompts for the path to an input image.
    - Saves the result with detected faces (and optional blur) to `output/images/`.

2.  **Video Mode**:
    - Prompts for the path to an input video file.
    - Processes the video, tracking faces between detections to maximize FPS.
    - Saves the processed video to `output/videos/`.

3.  **Webcam Mode**:
    - Opens your default camera (Index 0).
    - Performs real-time detection and tracking.
    - Press `q` to quit the live view.

### Configuration

You can adjust key settings in `src/config.py`:

- `CONFIDENCE_THRESHOLD`: Minimum confidence for a detection to be considered valid (Default: 0.5).
- `BLUR_FACES`: Default state for face blurring (can also be toggled at runtime).
- `SKIP_FRAMES`: Number of frames to track before running a full detection again (Optimization).

## Directory Structure

```
face_detector/
├── main.py              # Entry point of the application
├── pyproject.toml       # Project configuration and dependencies
├── uv.lock              # Dependency lock file
├── README.md            # Project documentation
├── models/              # Pre-trained Caffe model files
│   ├── deploy.prototxt.txt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── src/
│   ├── config.py        # Configuration constants
│   ├── detector.py      # FaceDetector class (DNN logic)
│   ├── processor.py     # Logic for processing images/video/webcam
│   ├── tracker.py       # Object tracking logic
│   └── utils.py         # Helper functions (drawing, FPS, etc.)
└── output/              # Generated results
    ├── images/
    └── videos/
```

## Credits

- Face detection model provided by OpenCV (ResNet-10 SSD).
