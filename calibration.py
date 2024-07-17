import cv2
import mediapipe as mp
import pickle

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def capture_video_and_save_keypoints(filename='ideal_keypoints.pkl'):
    cap = cv2.VideoCapture(0)
    frames = []
    while len(frames) < 600:  # 20 seconds at 30fps
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Calibration', frame)  # Show the webcam feed
        frames.append(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    if frames:
        last_frame = frames[-1]
        image = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks:
            keypoints = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            with open(filename, 'wb') as f:
                pickle.dump(keypoints, f)
            print("Keypoints saved successfully.")
        else:
            print("No landmarks detected in the final frame.")
    else:
        print("No frames captured.")


if __name__ == "__main__":
    capture_video_and_save_keypoints()
