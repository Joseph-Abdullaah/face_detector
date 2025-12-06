import os
from src.detector import FaceDetector
from src.processor import process_image, process_video, process_webcam
from src import config

def main():
    # Load detector
    detector = FaceDetector(
        model_prototxt=config.MODEL_PATHS["prototxt"],
        model_weights=config.MODEL_PATHS["caffemodel"],
        confidence_threshold=config.CONFIDENCE_THRESHOLD
    )

    print("=== MULTIPLE FACE DETECTOR ===")
    print("Select mode:")
    print("1 - Image")
    print("2 - Video")
    print("3 - Webcam Live Stream")
    choice = input("Enter choice [1/2/3]: ")

    blur_input = input("Blur faces? (y/n) [default=y]: ").lower()
    blur_faces = True if blur_input != "n" else False

    if choice == "1":
        image_path = input("Enter path to image: ")
        if not os.path.isfile(image_path):
            print("[ERROR] File does not exist!")
            return
        process_image(image_path, detector, blur_faces)

    elif choice == "2":
        video_path = input("Enter path to video: ")
        if not os.path.isfile(video_path):
            print("[ERROR] File does not exist!")
            return
        process_video(video_path, detector, blur_faces, skip_frames=config.SKIP_FRAMES)

    elif choice == "3":
        cam_index = input("Enter webcam index [default=0]: ")
        cam_index = int(cam_index) if cam_index.isdigit() else 0
        process_webcam(detector, blur_faces, skip_frames=config.SKIP_FRAMES, camera_index=cam_index)

    else:
        print("[ERROR] Invalid choice!")


if __name__ == "__main__":
    main()
