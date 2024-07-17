import cv2
import pickle
import numpy as np
from utils import draw_keypoints


def load_ideal_keypoints(filename='ideal_keypoints.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)


ideal_keypoints = load_ideal_keypoints()


def show_overlay_and_capture():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        overlay_frame = frame.copy()
        draw_keypoints(overlay_frame, ideal_keypoints)
        combined_frame = cv2.addWeighted(frame, 0.5, overlay_frame, 0.5, 0)
        cv2.imshow('Overlay', combined_frame)

        if cv2.waitKey(10) & 0xFF == ord('n'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_overlay_and_capture()
