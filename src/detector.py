import cv2
import os
import time


class FaceDetector:
    def __init__(self, model_prototxt, model_weights, confidence_threshold=0.5):
        """
        Initialize the detector with model paths and confidence threshold.
        """
        self.model_prototxt = model_prototxt
        self.model_weights = model_weights
        self.conf_threshold = confidence_threshold
        self.net = None

        # Load the model
        self._load_model()

    # --------------------------------------------------------
    def _load_model(self):
        """
        Load the DNN model from disk.
        """
        # Validate model paths
        if not os.path.exists(self.model_prototxt):
            raise FileNotFoundError(f"Prototxt not found at: {self.model_prototxt}")

        if not os.path.exists(self.model_weights):
            raise FileNotFoundError(f"CaffeModel not found at: {self.model_weights}")

        # Load model
        self.net = cv2.dnn.readNetFromCaffe(self.model_prototxt, self.model_weights)

        # Set backend and target (CPU by default)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # --------------------------------------------------------
    def _preprocess(self, frame):
        """
        Prepare frame as input blob for DNN.
        """
        (h, w) = frame.shape[:2]

        # Create blob for DNN
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
            swapRB=False,
            crop=False
        )

        return blob, w, h

    # --------------------------------------------------------
    def detect(self, frame):
        """
        Perform detection on a single frame.
        Returns:
          - detections list
          - count
          - detection time in ms
        """
        blob, w, h = self._preprocess(frame)
        self.net.setInput(blob)

        start_time = time.time()
        detections = self.net.forward()
        end_time = time.time()

        detection_time_ms = (end_time - start_time) * 1000.0

        # Process raw detections
        final_faces = self._postprocess(detections, w, h)

        result = {
            "detections": final_faces,
            "count": len(final_faces),
            "detection_time_ms": detection_time_ms
        }

        return result

    # --------------------------------------------------------
    def _postprocess(self, detections, frame_width, frame_height):
        """
        Parse raw DNN detections and filter valid faces.
        """
        results = []
        detections_count = detections.shape[2]

        for i in range(detections_count):
            confidence = detections[0, 0, i, 2]

            # Skip low-confidence detections
            if confidence < self.conf_threshold:
                continue

            # Extract normalized box
            x1_norm = detections[0, 0, i, 3]
            y1_norm = detections[0, 0, i, 4]
            x2_norm = detections[0, 0, i, 5]
            y2_norm = detections[0, 0, i, 6]

            # Scale to pixel coordinates
            box = self._scale_box(
                (x1_norm, y1_norm, x2_norm, y2_norm),
                frame_width,
                frame_height
            )

            results.append({
                "box": box,
                "confidence": float(confidence)
            })

        return results

    # --------------------------------------------------------
    def _scale_box(self, box, frame_width, frame_height):
        """
        Convert normalized box to pixel coordinates.
        """
        (x1, y1, x2, y2) = box

        x1 = int(x1 * frame_width)
        y1 = int(y1 * frame_height)
        x2 = int(x2 * frame_width)
        y2 = int(y2 * frame_height)

        # Clip to boundaries
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width - 1))
        y2 = max(0, min(y2, frame_height - 1))

        return (x1, y1, x2, y2)
