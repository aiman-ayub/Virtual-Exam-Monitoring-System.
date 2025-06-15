import cv2
import mediapipe as mp
import pandas as pd
import os
from datetime import datetime
import numpy as np

# Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=2,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

# Setup folders
os.makedirs("logs", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)
log_path = "logs/violations.csv"
if not os.path.exists(log_path):
    pd.DataFrame(columns=["timestamp", "violation", "screenshot"]).to_csv(log_path, index=False)

# Iris and face landmarks
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
LEFT_IRIS = 468
RIGHT_IRIS = 473
NOSE_TIP = 1
CHIN = 152
LEFT_EAR = 234
RIGHT_EAR = 454

cap = cv2.VideoCapture(0)
print("[INFO] Monitoring started. Press ESC to stop.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    frame_height, frame_width, _ = frame.shape
    face_count = 0
    violation = None

    if results.multi_face_landmarks:
        face_count = len(results.multi_face_landmarks)

        for face_landmarks in results.multi_face_landmarks[:1]:
            lm = face_landmarks.landmark

            # Eye Gaze Detection
            left_outer = int(lm[LEFT_EYE[0]].x * frame_width)
            left_inner = int(lm[LEFT_EYE[1]].x * frame_width)
            left_iris_x = int(lm[LEFT_IRIS].x * frame_width)
            left_iris_y = int(lm[LEFT_IRIS].y * frame_height)
            left_eye_width = left_inner - left_outer
            left_ratio = (left_iris_x - left_outer) / (left_eye_width + 1e-6)

            right_outer = int(lm[RIGHT_EYE[0]].x * frame_width)
            right_inner = int(lm[RIGHT_EYE[1]].x * frame_width)
            right_iris_x = int(lm[RIGHT_IRIS].x * frame_width)
            right_iris_y = int(lm[RIGHT_IRIS].y * frame_height)
            right_eye_width = right_inner - right_outer
            right_ratio = (right_iris_x - right_outer) / (right_eye_width + 1e-6)

            iris_avg_ratio = (left_ratio + right_ratio) / 2

            # Vertical gaze detection
            eye_top = (lm[159].y + lm[386].y) / 2
            eye_bottom = (lm[145].y + lm[374].y) / 2
            eye_center = (eye_top + eye_bottom) / 2
            iris_center = (lm[LEFT_IRIS].y + lm[RIGHT_IRIS].y) / 2
            vertical_ratio = (iris_center - eye_top) / (eye_bottom - eye_top + 1e-6)

            # Gaze classification
            gaze = "Focused"
            if iris_avg_ratio < 0.35:
                gaze = "Looking Left"
                violation = "Distraction - Looking Left"
            elif iris_avg_ratio > 0.65:
                gaze = "Looking Right"
                violation = "Distraction - Looking Right"
            elif vertical_ratio < 0.35:
                gaze = "Looking Up"
                violation = "Distraction - Looking Up"
            elif vertical_ratio > 0.65:
                gaze = "Looking Down"
                violation = "Distraction - Looking Down"

            # Head Pose Estimation
            def calc_angle(p1, p2):
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                return np.degrees(np.arctan2(dy, dx))

            nose = (int(lm[NOSE_TIP].x * frame_width), int(lm[NOSE_TIP].y * frame_height))
            chin = (int(lm[CHIN].x * frame_width), int(lm[CHIN].y * frame_height))
            left_ear = (int(lm[LEFT_EAR].x * frame_width), int(lm[LEFT_EAR].y * frame_height))
            right_ear = (int(lm[RIGHT_EAR].x * frame_width), int(lm[RIGHT_EAR].y * frame_height))

            vertical_angle = calc_angle(nose, chin)
            horizontal_angle = calc_angle(left_ear, right_ear)

            head_direction = "Head Centered"
            if vertical_angle < 80:
                head_direction = "Head Tilted Down"
                violation = violation or "Head Tilted Down"
            elif vertical_angle > 100:
                head_direction = "Head Tilted Up"
                violation = violation or "Head Tilted Up"
            elif horizontal_angle < 160:
                head_direction = "Head Turned Right"
                violation = violation or "Head Turned Right"
            elif horizontal_angle > 200:
                head_direction = "Head Turned Left"
                violation = violation or "Head Turned Left"

            # Display gaze and head pose info
            cv2.putText(frame, f"Gaze: {gaze}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Head: {head_direction}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # If any distraction (not focused and head not centered), capture violation
            if gaze != "Focused" or head_direction != "Head Centered":
                violation = f"{gaze}, {head_direction}"
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                screenshot_name = f"screenshots/distraction_{timestamp}.jpg"
                cv2.imwrite(screenshot_name, frame)
                log_entry = pd.DataFrame([[timestamp, violation, screenshot_name]])
                log_entry.to_csv(log_path, mode='a', header=False, index=False)
                cv2.putText(frame, f"Violation: {violation}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    else:
        # No face detected = Absent
        violation = "Student Absent"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        screenshot_name = f"screenshots/absent_{timestamp}.jpg"
        cv2.imwrite(screenshot_name, frame)
        log_entry = pd.DataFrame([[timestamp, violation, screenshot_name]])
        log_entry.to_csv(log_path, mode='a', header=False, index=False)
        cv2.putText(frame, "Violation: Absent", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Check for multiple people
    if face_count > 1:
        violation = "Multiple Faces Detected"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        screenshot_name = f"screenshots/multiple_{timestamp}.jpg"
        cv2.imwrite(screenshot_name, frame)
        log_entry = pd.DataFrame([[timestamp, violation, screenshot_name]])
        log_entry.to_csv(log_path, mode='a', header=False, index=False)
        cv2.putText(frame, "Violation: Multiple People", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display face count
    cv2.putText(frame, f"Faces Detected: {face_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('AI Exam Monitoring System', frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to stop
        print("[INFO] Monitoring stopped.")
        break

cap.release()
cv2.destroyAllWindows()
