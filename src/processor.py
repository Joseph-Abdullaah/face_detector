import cv2
import os
import time
from src.detector import FaceDetector
from src.tracker import FaceTrackerManager
from src import utils
from src import config

# ----------------------------
# IMAGE PROCESSING
# ----------------------------
def process_image(image_path, detector, blur_faces=config.BLUR_FACES, output_dir="output/images"):
    """
    Process a single image:
    - Detect faces
    - Blur faces if enabled
    - Draw boxes & labels
    - Save output
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return

    h, w = image.shape[:2]

    # Detect faces
    start = utils.start_timer()
    result = detector.detect(image)
    detection_time = utils.end_timer(start)

    face_boxes = [utils.clip_box_to_frame(det["box"], w, h) for det in result["detections"]]

    # Blur faces if enabled
    if blur_faces and face_boxes:
        image = utils.blur_faces(image, face_boxes)

    # Draw boxes
    for box in face_boxes:
        utils.draw_box(image, box, color=utils.get_color("green"))

    # Overlay labels
    utils.put_face_count(image, result["count"])
    utils.put_detection_time(image, detection_time)
    fps = utils.calculate_fps(start)
    utils.put_fps(image, fps)

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)
    print(f"[INFO] Processed image saved at: {output_path}")


# ----------------------------
# VIDEO PROCESSING
# ----------------------------
def process_video(video_path, detector, blur_faces=config.BLUR_FACES, skip_frames=6, output_dir="output/videos"):
    """
    Process a video file:
    - Detect faces every skip_frames
    - Track faces in between
    - Draw boxes, blur faces, overlay labels
    - Save output video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    # Video writer setup
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(video_path)
    output_path = os.path.join(output_dir, filename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps_input, (width, height))

    tracker_manager = FaceTrackerManager()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w = frame.shape[:2]

        # Detect faces every skip_frames
        if frame_count % skip_frames == 1 or tracker_manager.get_active_tracks() == 0:
            start = utils.start_timer()
            result = detector.detect(frame)
            detection_time = utils.end_timer(start)

            face_boxes = [utils.clip_box_to_frame(det["box"], w, h) for det in result["detections"]]

            # Reset trackers
            tracker_manager.clear_all()
            for box in face_boxes:
                tracker_manager.add_tracker(frame, box)

        else:
            # Update trackers
            tracked_faces = tracker_manager.update_trackers(frame)
            face_boxes = [box for box, _id in tracked_faces]
            detection_time = 0

        # Blur faces if enabled
        if blur_faces and face_boxes:
            frame = utils.blur_faces(frame, face_boxes)

        # Draw boxes
        for box in face_boxes:
            utils.draw_box(frame, box, color=utils.get_color("green"))

        # Overlay labels
        utils.put_face_count(frame, len(face_boxes))
        utils.put_detection_time(frame, detection_time)
        fps = utils.calculate_fps(time.time() - 1 / fps_input)  # approximate
        utils.put_fps(frame, fps)

        out.write(frame)

    cap.release()
    out.release()
    print(f"[INFO] Processed video saved at: {output_path}")


# ----------------------------
# WEBCAM PROCESSING
# ----------------------------
def process_webcam(detector, blur_faces=config.BLUR_FACES, skip_frames=6, camera_index=0):
    """
    Process live webcam stream:
    - Detect faces every skip_frames
    - Track faces in between
    - Draw boxes, blur faces, overlay labels
    - Display live feed
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam: {camera_index}")
        return

    tracker_manager = FaceTrackerManager()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w = frame.shape[:2]

        # Detect faces every skip_frames
        if frame_count % skip_frames == 1 or tracker_manager.get_active_tracks() == 0:
            start = utils.start_timer()
            result = detector.detect(frame)
            detection_time = utils.end_timer(start)

            face_boxes = [utils.clip_box_to_frame(det["box"], w, h) for det in result["detections"]]

            # Reset trackers
            tracker_manager.clear_all()
            for box in face_boxes:
                tracker_manager.add_tracker(frame, box)

        else:
            # Update trackers
            tracked_faces = tracker_manager.update_trackers(frame)
            face_boxes = [box for box, _id in tracked_faces]
            detection_time = 0

        # Blur faces if enabled
        if blur_faces and face_boxes:
            frame = utils.blur_faces(frame, face_boxes)

        # Draw boxes
        for box in face_boxes:
            utils.draw_box(frame, box, color=utils.get_color("green"))

        # Overlay labels
        utils.put_face_count(frame, len(face_boxes))
        utils.put_detection_time(frame, detection_time)
        fps = utils.calculate_fps(time.time() - 1 / 30)  # approximate
        utils.put_fps(frame, fps)

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
