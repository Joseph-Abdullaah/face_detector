# -------------------------
# Face Detection Settings
# -------------------------
CONFIDENCE_THRESHOLD = 0.5   # Minimum confidence to accept detection
BLUR_FACES = True            # Blur faces ON/OFF
TRACKING_ENABLED = True      # Enable tracking (True = use tracker)

# -------------------------
# Model Paths
# -------------------------
MODEL_PATHS = {
    "prototxt": "models/deploy.prototxt.txt",
    "caffemodel": "models/res10_300x300_ssd_iter_140000.caffemodel"
}

# -------------------------
# Colors
# -------------------------
COLORS = {
    "face_box": (0, 255, 0),
    "text": (0, 255, 0),
    "fps": (0, 255, 255)
}

# -------------------------
# Other Settings
# -------------------------
SKIP_FRAMES = 6  # Detect every 6 frames in video/webcam
