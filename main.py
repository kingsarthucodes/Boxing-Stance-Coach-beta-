import calibration
import overlay
import evaluation


def main():
    print("Starting Virtual Boxing Coach...")
    print("Step 1: Calibrate the system by capturing your ideal stance.")
    calibration.capture_video_and_save_keypoints()

    print("Step 2: Show overlay to align yourself with the ideal stance.")
    overlay.show_overlay_and_capture()

    print("Step 3: Evaluate your stance and provide feedback.")
    evaluation.evaluate_stance()


if __name__ == "__main__":
    main()
