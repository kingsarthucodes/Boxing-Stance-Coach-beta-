import cv2
import mediapipe as mp
import time
import pickle
from utils import draw_keypoints_with_lines, generate_feedback, load_ideal_keypoints, check_feet_alignment, \
    check_foot_angles, check_knee_bend, check_hands_and_chin, draw_foot_direction_lines, draw_foot_position_box, \
    draw_foot_rotation_arrows, check_alignment  # Import the new function

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def main():
    print("Starting Virtual Boxing Coach...")

    # Step 1: Show welcome message
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Boxing Coach', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Boxing Coach', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    start_time = time.time()
    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, "Welcome to Boxing Coach!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)
        cv2.imshow('Boxing Coach', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    time.sleep(2)  # Pause for 2 seconds between messages

    # Step 2: Instruct user to get into boxing stance
    start_time = time.time()
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, "Get into your boxing stance", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)
        cv2.imshow('Boxing Coach', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Step 3: Load ideal keypoints and show overlay
    ideal_keypoints = load_ideal_keypoints()
    print(f"Loaded ideal keypoints: {ideal_keypoints}")  # Debugging line
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay_frame = frame.copy()
        draw_keypoints_with_lines(overlay_frame, ideal_keypoints)
        combined_frame = cv2.addWeighted(frame, 0.5, overlay_frame, 0.5, 0)
        cv2.putText(combined_frame, "Align yourself with the overlay", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2, cv2.LINE_AA)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks:
            current_keypoints = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            draw_keypoints_with_lines(frame, current_keypoints)  # Draw user's pose
            if check_alignment(current_keypoints, ideal_keypoints, threshold=0.1):  # Adjust threshold as needed
                break
            else:
                print("Alignment not correct yet")  # Debugging line
        else:
            print("No pose landmarks detected")  # Debugging line
        cv2.imshow('Boxing Coach', combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Step 4: Evaluate stance
    steps = [
        ("1. Feet Alignment", draw_foot_position_box,
         "Align your front foot toe with your back foot heel and keep feet a bit wider than shoulder-width apart."),
        ("2. Foot Positioning", check_foot_angles,
         "Point your front foot 20-30 degrees and your back foot 50-90 degrees."),
        ("3. Knee Bend", check_knee_bend, "Bend knees a bit by sticking out your butt."),
        ("4. Hands and Chin", check_hands_and_chin, "Raise your hands and tuck your chin.")
    ]
    current_step = 0
    hold_time = 3  # Hold each position for at least 3 seconds
    step_start_time = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks:
            current_keypoints = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            draw_keypoints_with_lines(frame, current_keypoints)  # Draw current keypoints and lines on the frame

            step_name, check_function, step_feedback = steps[current_step]
            correct, feedback = check_function(frame, current_keypoints)
            if correct:
                if step_start_time is None:
                    step_start_time = time.time()
                elapsed_time = time.time() - step_start_time
                remaining_time = hold_time - int(elapsed_time)
                if remaining_time > 0:
                    feedback_text = f"{feedback}. Hold for {remaining_time} seconds..."
                    cv2.putText(frame, feedback_text, (4, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                                cv2.LINE_AA)
                else:
                    current_step += 1
                    step_start_time = None
                    if current_step >= len(steps):
                        cv2.putText(frame, "Congrats! You're in a perfect boxing stance!", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.imshow('Boxing Coach', frame)
                        cv2.waitKey(5000)
                        break
            else:
                step_start_time = None
                cv2.putText(frame, feedback, (4, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, f'Current Step: {step_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, step_feedback, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('Boxing Coach', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
