import cv2
import time


# ---------------------------------------------------------
# COLOR UTILITIES
# ---------------------------------------------------------

def get_color(name: str):
    """Return BGR color tuple based on color name."""
    colors = {
        "green": (0, 255, 0),
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "blue": (255, 0, 0),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }
    return colors.get(name.lower(), (0, 255, 0))  # default green


# ---------------------------------------------------------
# DRAWING UTILITIES
# ---------------------------------------------------------

def draw_box(frame, box, color=(0, 255, 0), thickness=2):
    """Draw a rectangle box on a frame."""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_label(frame, text, position, color=(0, 255, 0)):
    """Draw a text label on the frame."""
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA
    )


def put_face_count(frame, count, color=(0, 255, 0)):
    """Draw the number of faces detected."""
    draw_label(frame, f"Faces: {count}", (10, 30), color)


def put_fps(frame, fps, color=(0, 255, 0)):
    """Draw FPS information."""
    draw_label(frame, f"FPS: {int(fps)}", (10, 60), color)


def put_detection_time(frame, ms, color=(0, 255, 0)):
    """Draw detection time per frame."""
    draw_label(frame, f"Detect: {int(ms)}ms", (10, 90), color)


# ---------------------------------------------------------
# FACE BLUR UTILITY
# ---------------------------------------------------------

def blur_faces(frame, face_boxes):
    """Blur only face regions in the frame."""
    for (x1, y1, x2, y2) in face_boxes:
        face_region = frame[y1:y2, x1:x2]
        if face_region.size != 0:
            blurred = cv2.GaussianBlur(face_region, (35, 35), 30)
            frame[y1:y2, x1:x2] = blurred
    return frame


# ---------------------------------------------------------
# BOUNDARY UTILITY
# ---------------------------------------------------------

def clip_box_to_frame(box, width, height):
    """Ensure box coordinates stay inside the frame."""
    x1, y1, x2, y2 = box

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    return (x1, y1, x2, y2)


# ---------------------------------------------------------
# TIME MEASUREMENT UTILITIES
# ---------------------------------------------------------

def start_timer():
    """Start timing an event."""
    return time.time()


def end_timer(start_time):
    """Return time difference in milliseconds."""
    return (time.time() - start_time) * 1000


def calculate_fps(start_time):
    """Compute FPS based on start time of frame."""
    dt = time.time() - start_time
    if dt == 0:
        return 0
    return 1 / dt
