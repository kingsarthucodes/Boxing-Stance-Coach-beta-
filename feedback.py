from utils import check_feet_alignment, check_shoulder_width, check_foot_angles, check_knee_bend

def generate_feedback(current_keypoints):
    feedback = []
    if not check_feet_alignment(current_keypoints):
        feedback.append("Align your front foot toe with your back foot heel.")
    if not check_shoulder_width(current_keypoints):
        feedback.append("Make sure your feet are shoulder-width apart.")
    if not check_foot_angles(current_keypoints):
        feedback.append("Adjust your foot angles: back foot at 30-60 degrees, front foot at 20 degrees.")
    if not check_knee_bend(current_keypoints):
        feedback.append("Slightly bend your knees.")
    return feedback
