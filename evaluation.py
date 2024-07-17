import cv2
import mediapipe as mp
import time
from utils import load_ideal_keypoints, generate_feedback, check_feet_alignment, check_shoulder_width, check_foot_angles, check_knee_bend

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

ideal_keypoints = load_ideal_keypoints()

def evaluate_stance():
    cap = cv2.VideoCapture(0)
    hold_time = 5  # Hold each position for at least 5 seconds
    start_time = None
    steps = [
        ("Toe-Heel Alignment", check_feet_alignment),
        ("Shoulder Width", check_shoulder_width),
        ("Foot Angles", check_foot_angles),
        ("Knee Bend", check_knee_bend)
    ]
    current_step = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks:
            current_keypoints = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            step_name, check_function = steps[current_step]
            if check_function(current_keypoints):
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time >= hold_time:
                    current_step += 1
                    start_time = None
                    if current_step >= len(steps):
                        break
            else:
                start_time = None
            feedback = generate_feedback(ideal_keypoints, current_keypoints)
            cv2.putText(frame, f'Current Step: {step_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            for i, message in enumerate(feedback):
                cv2.putText(frame, message, (10, 60 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('Evaluation', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    evaluate_stance()
