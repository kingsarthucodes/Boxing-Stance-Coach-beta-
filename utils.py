import cv2
import mediapipe as mp
import math
import pickle

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS


def calculate_angle(a, b):
    delta_x = b[0] - a[0]
    delta_y = b[1] - a[1]
    angle = math.degrees(math.atan2(delta_y, delta_x))
    return angle

def check_alignment(current_keypoints, ideal_keypoints, threshold=0.1):
    for i in range(len(current_keypoints)):
        current = current_keypoints[i]
        ideal = ideal_keypoints[i]
        distance = ((current[0] - ideal[0]) ** 2 + (current[1] - ideal[1]) ** 2 + (current[2] - ideal[2]) ** 2) ** 0.5
        if distance > threshold:
            return False
    return True

def check_feet_alignment(current_keypoints):
    left_toe = current_keypoints[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
    right_heel = current_keypoints[mp_pose.PoseLandmark.RIGHT_HEEL.value]
    left_shoulder = current_keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = current_keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Calculate the angle of the line between left toe and right heel
    foot_line_angle = calculate_angle(right_heel, left_toe)
    correct_alignment = abs(foot_line_angle) < 15 or abs(foot_line_angle - 180) < 15

    # Calculate the distance between the feet
    foot_distance = math.sqrt((left_toe[0] - right_heel[0]) ** 2 + (left_toe[1] - right_heel[1]) ** 2)
    shoulder_width = math.sqrt((left_shoulder[0] - right_shoulder[0]) ** 2 + (left_shoulder[1] - right_shoulder[1]) ** 2)

    # Ensure feet are slightly more than shoulder-width apart (10-20% more)
    correct_distance = 1.1 * shoulder_width <= foot_distance <= 1.2 * shoulder_width

    return correct_alignment and correct_distance


def draw_keypoints_with_lines(frame, keypoints):
    h, w, _ = frame.shape
    for i, keypoint in enumerate(keypoints):
        try:
            x, y = int(keypoint[0] * w), int(keypoint[1] * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        except ValueError as e:
            print(f"Error processing keypoint {i}: {keypoint}, error: {e}")
            continue

    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        try:
            start_point = (int(keypoints[start_idx][0] * w), int(keypoints[start_idx][1] * h))
            end_point = (int(keypoints[end_idx][0] * w), int(keypoints[end_idx][1] * h))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        except ValueError as e:
            print(f"Error processing connection {connection}: {e}")
            continue


def calculate_angle(point1, point2):
    angle = math.degrees(math.atan2(point2[1] - point1[1], point2[0] - point1[0]))
    return angle

def draw_foot_position_box(frame, current_keypoints):
    h, w, _ = frame.shape
    left_toe = (int(current_keypoints[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value][0] * w),
                int(current_keypoints[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value][1] * h))
    right_heel = (int(current_keypoints[mp_pose.PoseLandmark.RIGHT_HEEL.value][0] * w),
                  int(current_keypoints[mp_pose.PoseLandmark.RIGHT_HEEL.value][1] * h))
    left_shoulder = (int(current_keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0] * w),
                     int(current_keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1] * h))
    right_shoulder = (int(current_keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0] * w),
                      int(current_keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1] * h))

    # Calculate the alignment angle
    foot_line_angle = calculate_angle(right_heel, left_toe)
    correct_alignment = abs(foot_line_angle) < 15 or abs(foot_line_angle - 180) < 15

    # Draw the line between the left foot and right heel
    line_color = (0, 255, 0) if correct_alignment else (0, 0, 255)
    cv2.line(frame, left_toe, right_heel, line_color, 2)

    # Calculate the distance between the feet and shoulder width
    foot_distance = math.sqrt((left_toe[0] - right_heel[0]) ** 2 + (left_toe[1] - right_heel[1]) ** 2)
    shoulder_width = math.sqrt(
        (left_shoulder[0] - right_shoulder[0]) ** 2 + (left_shoulder[1] - right_shoulder[1]) ** 2)
    correct_distance = 1.2 * shoulder_width <= foot_distance <= 1.4 * shoulder_width  # Adjusted threshold

    # Draw visual cues for foot distance
    distance_color = (0, 255, 0) if correct_distance else (0, 0, 255)
    cv2.line(frame, left_shoulder, right_shoulder, (255, 0, 0), 2)  # Draw shoulder line
    cv2.line(frame, left_toe, right_heel, distance_color, 2)  # Draw foot distance line

    # Adjusted angled lines from shoulders to the ground
    angle_offset = 0.8  # Adjust this value to change the angle
    left_shoulder_ground = (int(left_shoulder[0] + shoulder_width * angle_offset), h)  # Adjusted for outward pointing
    right_shoulder_ground = (int(right_shoulder[0] - shoulder_width * angle_offset), h)  # Adjusted for outward pointing

    cv2.line(frame, left_shoulder, left_shoulder_ground, (0, 255, 0), 1)
    cv2.line(frame, right_shoulder, right_shoulder_ground, (0, 255, 0), 1)

    # Draw lines from feet to the corresponding shoulder line
    cv2.line(frame, left_toe, (left_shoulder_ground[0], left_toe[1]), distance_color, 1)
    cv2.line(frame, right_heel, (right_shoulder_ground[0], right_heel[1]), distance_color, 1)

    # Determine the feedback
    if not correct_alignment:
        feedback = "Adjust Feet"
    elif not correct_distance:
        feedback = "Widen feet a bit more" if foot_distance < 1.2 * shoulder_width else "Stance too wide"
    else:
        feedback = "Correct Alignment and Distance"

    return correct_alignment and correct_distance, feedback

def generate_feedback(current_keypoints):
    left_toe = current_keypoints[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
    right_heel = current_keypoints[mp_pose.PoseLandmark.RIGHT_HEEL.value]
    left_shoulder = current_keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = current_keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Calculate the angle of the line between left toe and right heel
    foot_line_angle = calculate_angle(right_heel, left_toe)
    correct_alignment = abs(foot_line_angle) < 10 or abs(foot_line_angle - 180) < 10

    # Calculate the distance between the feet and shoulders
    foot_distance = math.sqrt((left_toe[0] - right_heel[0]) ** 2 + (left_toe[1] - right_heel[1]) ** 2)
    shoulder_width = math.sqrt(
        (left_shoulder[0] - right_shoulder[0]) ** 2 + (left_shoulder[1] - right_shoulder[1]) ** 2)

    # Ensure feet are slightly more than shoulder-width apart (20-30% more)
    correct_distance = 1.2 * shoulder_width <= foot_distance <= 1.3 * shoulder_width

    feedback = []
    if not correct_alignment:
        feedback.append("Align your front foot toe with your back foot heel.")
    if not correct_distance:
        feedback.append("Keep feet slightly more than shoulder-width apart.")

    return feedback


def load_ideal_keypoints():
    with open("ideal_keypoints.pkl", "rb") as f:
        ideal_keypoints = pickle.load(f)
    return ideal_keypoints


def calculate_angle_2d(point1, point2, point3):
    angle = math.degrees(math.atan2(point3[1] - point2[1], point3[0] - point2[0]) - math.atan2(point1[1] - point2[1],
                                                                                               point1[0] - point2[0]))
    return abs(angle) if abs(angle) <= 180 else 360 - abs(angle)


def check_foot_angles(frame, keypoints):
    h, w, _ = frame.shape
    left_heel = (int(keypoints[mp_pose.PoseLandmark.LEFT_HEEL.value][0] * w),
                 int(keypoints[mp_pose.PoseLandmark.LEFT_HEEL.value][1] * h))
    left_toe = (int(keypoints[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value][0] * w),
                int(keypoints[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value][1] * h))
    right_heel = (int(keypoints[mp_pose.PoseLandmark.RIGHT_HEEL.value][0] * w),
                  int(keypoints[mp_pose.PoseLandmark.RIGHT_HEEL.value][1] * h))
    right_toe = (int(keypoints[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value][0] * w),
                 int(keypoints[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value][1] * h))

    front_foot_angle = calculate_angle_2d(left_heel, left_toe, [left_toe[0] + 0.1, left_toe[1]])
    back_foot_angle = calculate_angle_2d(right_heel, right_toe, [right_toe[0] + 0.1, right_toe[1]])

    correct_front_foot = 140 <= front_foot_angle <= 160
    correct_back_foot = 50 <= back_foot_angle <= 90

    feedback = []
    if not correct_front_foot:
        if front_foot_angle < 150:
            feedback.append("Rotate front foot right")
            cv2.arrowedLine(frame, left_toe, (left_toe[0] + 50, left_toe[1]), (0, 0, 255), 5, tipLength=0.5)
        else:
            feedback.append("Rotate front foot left")
            cv2.arrowedLine(frame, left_toe, (left_toe[0] - 50, left_toe[1]), (0, 0, 255), 5, tipLength=0.5)

    if not correct_back_foot:
        if back_foot_angle < 50:
            feedback.append("Rotate back foot right")
            cv2.arrowedLine(frame, right_toe, (right_toe[0] + 50, right_toe[1]), (0, 0, 255), 5, tipLength=0.5)
        else:
            feedback.append("Rotate back foot left")
            cv2.arrowedLine(frame, right_toe, (right_toe[0] - 50, right_toe[1]), (0, 0, 255), 5, tipLength=0.5)

    return correct_front_foot and correct_back_foot, " ".join(feedback)


def check_knee_bend(frame, keypoints):
    h, w, _ = frame.shape
    left_knee = (int(keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value][0] * w),
                 int(keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value][1] * h))
    right_knee = (int(keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value][0] * w),
                  int(keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value][1] * h))
    left_hip = (int(keypoints[mp_pose.PoseLandmark.LEFT_HIP.value][0] * w),
                int(keypoints[mp_pose.PoseLandmark.LEFT_HIP.value][1] * h))
    right_hip = (int(keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value][0] * w),
                 int(keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value][1] * h))

    left_ankle = (int(keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value][0] * w),
                  int(keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value][1] * h))
    right_ankle = (int(keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value][0] * w),
                   int(keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value][1] * h))

    left_knee_angle = calculate_angle_2d(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle_2d(right_hip, right_knee, right_ankle)

    # Adjusted threshold for front knee (left knee)
    correct_left_knee = 140 <= left_knee_angle <= 170
    correct_right_knee = 150 <= right_knee_angle <= 170

    feedback = []
    if not correct_left_knee:
        feedback.append("Bend your left knee more (make sure your facing forward)")
        cv2.arrowedLine(frame, left_knee, (left_knee[0], left_knee[1] + 50), (0, 0, 255), 5, tipLength=0.5)
    if not correct_right_knee:
        feedback.append("Bend your right knee more (make sure your facing forward)")
        cv2.arrowedLine(frame, right_knee, (right_knee[0], right_knee[1] + 50), (0, 0, 255), 5, tipLength=0.5)

    return correct_left_knee and correct_right_knee, " ".join(feedback)


def check_hands_and_chin(frame, keypoints):
    h, w, _ = frame.shape
    left_hand = (int(keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value][0] * w),
                 int(keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value][1] * h))
    right_hand = (int(keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value][0] * w),
                  int(keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value][1] * h))
    chin = (int(keypoints[mp_pose.PoseLandmark.NOSE.value][0] * w),
            int(keypoints[mp_pose.PoseLandmark.NOSE.value][1] * h))
    left_elbow = (int(keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value][0] * w),
                  int(keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value][1] * h))
    right_elbow = (int(keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value][0] * w),
                   int(keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value][1] * h))
    left_shoulder = (int(keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0] * w),
                     int(keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1] * h))
    right_shoulder = (int(keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0] * w),
                      int(keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1] * h))

    # Criteria for hands being raised above the chin with a buffer
    correct_left_hand = left_hand[1] < chin[1] + 50
    correct_right_hand = right_hand[1] < chin[1] + 50

    # Criteria for elbows being tucked closer to the torso
    correct_left_elbow = abs(left_elbow[0] - left_shoulder[0]) < 50
    correct_right_elbow = abs(right_elbow[0] - right_shoulder[0]) < 50

    feedback = []
    if not correct_left_hand:
        feedback.append("Raise your left hand")
        cv2.arrowedLine(frame, left_hand, (left_hand[0], left_hand[1] - 50), (0, 0, 255), 5, tipLength=0.5)
    if not correct_right_hand:
        feedback.append("Raise your right hand")
        cv2.arrowedLine(frame, right_hand, (right_hand[0], right_hand[1] - 50), (0, 0, 255), 5, tipLength=0.5)
    if not correct_left_elbow:
        feedback.append("Tuck your left elbow closer to your body")
        cv2.arrowedLine(frame, left_elbow, (left_elbow[0] + 50, left_elbow[1]), (0, 0, 255), 5, tipLength=0.5)
    if not correct_right_elbow:
        feedback.append("Tuck your right elbow closer to your body")
        cv2.arrowedLine(frame, right_elbow, (right_elbow[0] - 50, right_elbow[1]), (0, 0, 255), 5, tipLength=0.5)

    return (correct_left_hand and correct_right_hand and correct_left_elbow and correct_right_elbow), " ".join(feedback)

def draw_foot_direction_lines(frame, current_keypoints):
    h, w, _ = frame.shape
    left_foot = (int(current_keypoints[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value][0] * w),
                 int(current_keypoints[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value][1] * h))
    right_foot = (int(current_keypoints[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value][0] * w),
                  int(current_keypoints[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value][1] * h))
    left_ankle = (int(current_keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value][0] * w),
                  int(current_keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value][1] * h))
    right_ankle = (int(current_keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value][0] * w),
                   int(current_keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value][1] * h))

    # Calculate the angles
    front_foot_angle = calculate_angle(left_ankle, left_foot)
    back_foot_angle = calculate_angle(right_ankle, right_foot)

    # Draw the direction lines
    front_direction = (left_foot[0] + int(50 * math.cos(math.radians(front_foot_angle))),
                       left_foot[1] + int(50 * math.sin(math.radians(front_foot_angle))))
    back_direction = (right_foot[0] + int(50 * math.cos(math.radians(back_foot_angle))),
                      right_foot[1] + int(50 * math.sin(math.radians(back_foot_angle))))

    cv2.line(frame, left_foot, front_direction, (255, 0, 0), 2)
    cv2.line(frame, right_foot, back_direction, (255, 0, 0), 2)



def draw_foot_rotation_arrows(frame, current_keypoints):
    h, w, _ = frame.shape
    left_foot = (int(current_keypoints[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value][0] * w),
                 int(current_keypoints[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value][1] * h))
    right_foot = (int(current_keypoints[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value][0] * w),
                  int(current_keypoints[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value][1] * h))
    left_ankle = (int(current_keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value][0] * w),
                  int(current_keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value][1] * h))
    right_ankle = (int(current_keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value][0] * w),
                   int(current_keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value][1] * h))

    front_foot_angle = calculate_angle(left_ankle, left_foot)
    back_foot_angle = calculate_angle(right_ankle, right_foot)

    front_direction = (left_foot[0] + int(50 * math.cos(math.radians(front_foot_angle))),
                       left_foot[1] + int(50 * math.sin(math.radians(front_foot_angle))))
    back_direction = (right_foot[0] + int(50 * math.cos(math.radians(back_foot_angle))),
                      right_foot[1] + int(50 * math.sin(math.radians(back_foot_angle))))

    cv2.arrowedLine(frame, left_foot, front_direction, (255, 0, 0), 2)
    cv2.arrowedLine(frame, right_foot, back_direction, (255, 0, 0), 2)
