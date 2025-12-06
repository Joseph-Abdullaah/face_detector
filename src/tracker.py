import cv2

class FaceTrackerManager:
    """
    Manages multiple face trackers (one tracker per detected face).
    Helps follow faces across frames without running detection every frame.
    """

    def __init__(self, tracker_type="CSRT"):
        self.tracker_type = tracker_type
        self.trackers = []   # list of (tracker_object, face_id)
        self.next_face_id = 0

    def _create_tracker(self):
        """Create a new tracker based on the tracker type."""
        if self.tracker_type == "CSRT":
            return cv2.TrackerCSRT_create()
        elif self.tracker_type == "KCF":
            return cv2.TrackerKCF_create()
        else:
            raise ValueError("Unsupported tracker type!")

    def add_tracker(self, frame, box):
        """
        Add a new tracker for a detected face.
        box must be (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        tracker = self._create_tracker()
        tracker.init(frame, (x1, y1, w, h))

        self.trackers.append((tracker, self.next_face_id))
        self.next_face_id += 1

    def update_trackers(self, frame):
        """
        Update all trackers with the new frame.
        Returns a list of tracked boxes and their IDs.
        Automatically removes trackers that fail.
        """
        updated_boxes = []
        working_trackers = []

        for tracker, face_id in self.trackers:
            success, box = tracker.update(frame)

            if success:
                x, y, w, h = box
                x2 = x + w
                y2 = y + h
                updated_boxes.append(((int(x), int(y), int(x2), int(y2)), face_id))
                working_trackers.append((tracker, face_id))
            # if tracker fails → ignore and drop it

        # keep only working trackers
        self.trackers = working_trackers

        return updated_boxes

    def clear_all(self):
        """Remove all trackers."""
        self.trackers = []

    def get_active_tracks(self):
        """Return number of active trackers."""
        return len(self.trackers)
