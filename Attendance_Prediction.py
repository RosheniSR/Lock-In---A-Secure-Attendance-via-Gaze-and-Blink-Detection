import cv2
import dlib
import numpy as np
import time
from tensorflow.keras.models import load_model
from joblib import load as joblib_load

# -----------------------------
# Load Haar cascades and Dlib model
# -----------------------------
face_cascade = cv2.CascadeClassifier(r"C:\Users\roshe\Documents\PROJECTS\Gaze Detection\models\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"C:\Users\roshe\Documents\PROJECTS\Gaze Detection\models\haarcascade_eye.xml")

predictor_path = r"C:\Users\roshe\Documents\PROJECTS\Gaze Detection\models\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# -----------------------------
# Load your two trained ML models
# -----------------------------
# 1. Fine tuned MobileNetV2 for looking / not looking
#    (expects full face color image resized to 224x224)
looking_model_path = r"C:\Users\roshe\Documents\PROJECTS\Gaze Detection\models\fine_tuned_mobilenetv2.h5"
looking_model = load_model(looking_model_path)

# 2. Blink SVM for open/closed eye classification
#    (expects a flattened, normalized grayscale image of size 64x64, i.e. 4096 features)
blink_model_path = r"C:\Users\roshe\Documents\PROJECTS\Gaze Detection\models\blink_svm.pkl"
blink_model = joblib_load(blink_model_path)

# -----------------------------
# Global parameters for attendance logic and fonts
# -----------------------------
ATTENDANCE_DURATION = 10       # seconds required to mark attendance (must be "good" for 10 sec)
ATTENDANCE_MESSAGE_DURATION = 2  # seconds to display the attendance message
MAX_CLOSED_FRAMES = 10         # if eyes are closed for > this many consecutive frames, reset the timer
EYE_INPUT_SIZE = (64, 64)      # expected input size for the blink SVM (64x64 = 4096 features)

# Font scales for on-screen text
FONT_SCALE = 1.2
ATTENDANCE_FONT_SCALE = 2.0
FONT_THICKNESS = 2

def detect_face_eyes(video_source=0):
    """
    Detects face, eyes, and landmarks; then runs the looking model and blink model.
    - The entire face region (in color) is passed to the looking model.
    - The eye regions (cropped from the grayscale image, expanded upward to include eyebrows)
      are resized to 64x64 and passed to the blink SVM.
      
    Prediction logic (inverted if necessary):
      * For the looking model, a probability <= 0.5 indicates "looking."
      * For the blink model, a prediction of 1 indicates "closed" and 0 indicates "open."
      
    If a face is detected, the system checks if the subject is looking. If yes,
    it then checks the eyes. When both eyes are predicted as open (blink SVM returns 0)
    for a continuous period (ATTENDANCE_DURATION seconds) with only brief closures allowed,
    attendance is marked.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    # Variables for attendance logic
    good_start_time = None       
    attendance_marked = False    # Whether attendance has been marked already
    attendance_marked_time = None  
    closed_eye_consecutive = 0   # Count of consecutive frames with eyes closed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -----------------------------
        # 1. Face detection using Haar cascade
        # -----------------------------
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) > 0:
            # For simplicity, select the largest detected face.
            (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), FONT_THICKNESS)

            # -----------------------------
            # 2. Looking detection using the entire face crop
            # -----------------------------
            face_roi = frame[y:y + h, x:x + w]
            try:
                face_roi_resized = cv2.resize(face_roi, (224, 224))
            except Exception as e:
                face_roi_resized = face_roi
            face_roi_norm = face_roi_resized.astype("float32") / 255.0
            face_input = np.expand_dims(face_roi_norm, axis=0)
            looking_pred = looking_model.predict(face_input)
            # Inverted logic: probability <= 0.5 means "looking"
            if looking_pred[0][0] <= 0.5:
                looking = True
                looking_text = "Looking: Yes"
            else:
                looking = False
                looking_text = "Looking: No"
            cv2.putText(frame, looking_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 0, 0), FONT_THICKNESS)

            # -----------------------------
            # 3. Eye detection using Haar cascade within the face ROI
            # -----------------------------
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), FONT_THICKNESS)

            # -----------------------------
            # 4. Get accurate eye coordinates using dlib’s landmarks
            # -----------------------------
            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, dlib_rect)

            # Extract landmarks for left (36-41) and right (42-47) eyes.
            left_eye_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)])
            right_eye_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)])

            for (ex_pt, ey_pt) in left_eye_pts:
                cv2.circle(frame, (ex_pt, ey_pt), 2, (0, 0, 255), -1)
            for (ex_pt, ey_pt) in right_eye_pts:
                cv2.circle(frame, (ex_pt, ey_pt), 2, (0, 0, 255), -1)

            # Compute bounding boxes for each eye.
            lx, ly, lw, lh = cv2.boundingRect(left_eye_pts)
            rx, ry, rw, rh = cv2.boundingRect(right_eye_pts)

            # Expand upward to include eyebrows (approx. 30% of the eye height)
            eyebrow_offset_left = int(0.3 * lh)
            eyebrow_offset_right = int(0.3 * rh)
            lx_new, ly_new = lx, max(ly - eyebrow_offset_left, 0)
            lw_new, lh_new = lw, lh + eyebrow_offset_left
            rx_new, ry_new = rx, max(ry - eyebrow_offset_right, 0)
            rw_new, rh_new = rw, rh + eyebrow_offset_right

            cv2.rectangle(frame, (lx_new, ly_new), (lx_new + lw_new, ly_new + lh_new), (0, 255, 255), FONT_THICKNESS)
            cv2.rectangle(frame, (rx_new, ry_new), (rx_new + rw_new, ry_new + rh_new), (0, 255, 255), FONT_THICKNESS)

            # -----------------------------
            # 5. Blink detection using the cropped eye images (grayscale with eyebrows)
            # -----------------------------
            left_eye_roi = gray[ly_new:ly_new + lh_new, lx_new:lx_new + lw_new]
            right_eye_roi = gray[ry_new:ry_new + rh_new, rx_new:rx_new + rw_new]

            eyes_open = False
            eye_state = "Unknown"
            if left_eye_roi.size != 0 and right_eye_roi.size != 0:
                try:
                    left_eye_resized = cv2.resize(left_eye_roi, EYE_INPUT_SIZE)
                    right_eye_resized = cv2.resize(right_eye_roi, EYE_INPUT_SIZE)
                except Exception as e:
                    left_eye_resized = None
                    right_eye_resized = None

                if left_eye_resized is not None and right_eye_resized is not None:
                    left_eye_input = left_eye_resized.flatten().astype("float32") / 255.0
                    right_eye_input = right_eye_resized.flatten().astype("float32") / 255.0

                    left_pred = blink_model.predict([left_eye_input])[0]
                    right_pred = blink_model.predict([right_eye_input])[0]

                    # Inverted logic: prediction of 1 indicates closed; 0 indicates open.
                    if left_pred == 1 and right_pred == 1:
                        eye_state = "Closed"
                        eyes_open = False
                    else:
                        eye_state = "Open"
                        eyes_open = True

                    cv2.putText(frame, f"Eye: {eye_state}", (x, y + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), FONT_THICKNESS)
            else:
                eyes_open = False

            # -----------------------------
            # 6. Attendance logic:
            #    A frame is "good" if the subject is looking and the eyes are open.
            #    If these conditions hold continuously for ATTENDANCE_DURATION seconds,
            #    mark attendance. Reset the timer if eyes close too long.
            # -----------------------------
            if looking:
                if eyes_open:
                    closed_eye_consecutive = 0
                    if good_start_time is None:
                        good_start_time = current_time
                else:
                    closed_eye_consecutive += 1
                    if closed_eye_consecutive > MAX_CLOSED_FRAMES:
                        good_start_time = None
            else:
                good_start_time = None
                closed_eye_consecutive = 0

            # If criteria are met for 10 seconds and attendance not yet marked.
            if good_start_time is not None:
                elapsed = current_time - good_start_time
                cv2.putText(frame, f"Good for {elapsed:.1f}s", (x, y - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), FONT_THICKNESS)
                if elapsed >= ATTENDANCE_DURATION and not attendance_marked:
                    attendance_marked = True
                    attendance_marked_time = current_time
            else:
                cv2.putText(frame, "Reset Timer", (x, y - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)
        else:
            good_start_time = None
            closed_eye_consecutive = 0

        # -----------------------------
        # 7. Display the Attendance Marked message at the top-right (if within display duration)
        # -----------------------------
        if attendance_marked_time is not None:
            if current_time - attendance_marked_time <= ATTENDANCE_MESSAGE_DURATION:
                # Calculate position at top right
                text = "Attendance Marked"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, ATTENDANCE_FONT_SCALE, FONT_THICKNESS)
                # Place text with some margin from top-right corner
                pos = (frame.shape[1] - text_width - 20, text_height + 20)
                cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, ATTENDANCE_FONT_SCALE, (0, 255, 0), FONT_THICKNESS)
            else:
                # After 2 seconds, clear the attendance display time.
                attendance_marked_time = None

        cv2.imshow("Face & Eye Detection with Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    # Pass a video file path or 0 for webcam.
    detect_face_eyes(0)