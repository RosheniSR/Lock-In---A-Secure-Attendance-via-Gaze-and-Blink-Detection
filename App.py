import cv2
import dlib
import numpy as np
import time
import csv
import os
from threading import Thread, Lock
from datetime import datetime
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from multiprocessing import process, Process, Queue
import time

from tensorflow.keras.models import load_model
from joblib import load as joblib_load

app = Flask(_name_)

# -----------------------------
# Global Variables and Locks
# -----------------------------
attendance_lock = Lock()      # To avoid race conditions when writing to CSV
attendance_marked = False      # Global flag for attendance marking
current_attendance = {}        # To temporarily store the user record (name, date, ...)
CSV_FILE = 'attendance.csv'
camera_process = None
frame_queue = None

# -----------------------------
# Load Haar Cascades and Dlib model
# -----------------------------
face_cascade = cv2.CascadeClassifier("/Users/rafaelzieganpalg/Projects/SRP_Lab/Scopus_Proj/models/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("/Users/rafaelzieganpalg/Projects/SRP_Lab/Scopus_Proj/models/haarcascade_eye.xml")

predictor_path = "/Users/rafaelzieganpalg/Projects/SRP_Lab/Scopus_Proj/models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# -----------------------------
# Load your two trained ML models
# -----------------------------
looking_model_path = "/Users/rafaelzieganpalg/Projects/SRP_Lab/Scopus_Proj/models/fine_tuned_mobilenetv2.h5"
looking_model = load_model(looking_model_path)

blink_model_path = "/Users/rafaelzieganpalg/Projects/SRP_Lab/Scopus_Proj/models/blink_svm.pkl"
blink_model = joblib_load(blink_model_path)
camera_active = True
attendance_logs = []


# -----------------------------
# Global parameters for attendance logic and fonts
# -----------------------------
ATTENDANCE_DURATION = 10       # seconds required to mark attendance (must be "good" for 10 sec)
ATTENDANCE_MESSAGE_DURATION = 2  # seconds to display the attendance message
MAX_CLOSED_FRAMES = 10         # if eyes closed for > this many consecutive frames, reset the timer
EYE_INPUT_SIZE = (64, 64)      # expected input size for the blink SVM

FONT_SCALE = 1.2
ATTENDANCE_FONT_SCALE = 2.0
FONT_THICKNESS = 2

# -----------------------------
# Camera capture
# -----------------------------
video_path = '/Users/rafaelzieganpalg/Projects/SRP_Lab/Scopus_Proj/Sample_ADHD.mov'
camera = cv2.VideoCapture(video_path, cv2.CAP_AVFOUNDATION)

if not camera.isOpened():
    raise RuntimeError("Could not start camera.")

# -----------------------------
# Generator: Process video frames and perform detection

# -----------------------------
def generate_frames():
    global attendance_marked, current_attendance, camera_active

    good_start_time = None
    closed_eye_consecutive = 0
    attendance_marked_time = None

    while camera_active:
        success, frame = camera.read()
        if not success:
            break

        current_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Face detection using Haar cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) > 0:
            (x, y, w, h) = max(faces, key=lambda r: r[2]*r[3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), FONT_THICKNESS)
            
            # 2. Looking detection: use full face region
            face_roi = frame[y:y+h, x:x+w]
            try:
                face_roi_resized = cv2.resize(face_roi, (224, 224))
            except Exception as e:
                face_roi_resized = face_roi
            face_roi_norm = face_roi_resized.astype("float32")/255.0
            face_input = np.expand_dims(face_roi_norm, axis=0)
            looking_pred = looking_model.predict(face_input)
            # Inverted logic: <= 0.5 means "looking"
            if looking_pred[0][0] <= 0.5:
                looking = True
                looking_text = "Looking: Yes"
            else:
                looking = False
                looking_text = "Looking: No"
            cv2.putText(frame, looking_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 0, 0), FONT_THICKNESS)

            # 3. Eye detection within face ROI
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), FONT_THICKNESS)

            # 4. Landmarks via dlib for more accurate eye region
            dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
            shape = predictor(gray, dlib_rect)
            left_eye_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)])
            right_eye_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)])
            for (ex_pt, ey_pt) in left_eye_pts:
                cv2.circle(frame, (ex_pt, ey_pt), 2, (0, 0, 255), -1)
            for (ex_pt, ey_pt) in right_eye_pts:
                cv2.circle(frame, (ex_pt, ey_pt), 2, (0, 0, 255), -1)

            lx, ly, lw, lh = cv2.boundingRect(left_eye_pts)
            rx, ry, rw, rh = cv2.boundingRect(right_eye_pts)
            eyebrow_offset_left = int(0.3 * lh)
            eyebrow_offset_right = int(0.3 * rh)
            lx_new, ly_new = lx, max(ly - eyebrow_offset_left, 0)
            lw_new, lh_new = lw, lh + eyebrow_offset_left
            rx_new, ry_new = rx, max(ry - eyebrow_offset_right, 0)
            rw_new, rh_new = rw, rh + eyebrow_offset_right
            cv2.rectangle(frame, (lx_new, ly_new), (lx_new+lw_new, ly_new+lh_new), (0, 255, 255), FONT_THICKNESS)
            cv2.rectangle(frame, (rx_new, ry_new), (rx_new+rw_new, ry_new+rh_new), (0, 255, 255), FONT_THICKNESS)

            # 5. Blink detection using cropped eye images
            left_eye_roi = gray[ly_new:ly_new+lh_new, lx_new:lx_new+lw_new]
            right_eye_roi = gray[ry_new:ry_new+rh_new, rx_new:rx_new+rw_new]
            eyes_open = False
            if left_eye_roi.size != 0 and right_eye_roi.size != 0:
                try:
                    left_eye_resized = cv2.resize(left_eye_roi, EYE_INPUT_SIZE)
                    right_eye_resized = cv2.resize(right_eye_roi, EYE_INPUT_SIZE)
                except Exception as e:
                    left_eye_resized = right_eye_resized = None

                if left_eye_resized is not None and right_eye_resized is not None:
                    left_eye_input = left_eye_resized.flatten().astype("float32")/255.0
                    right_eye_input = right_eye_resized.flatten().astype("float32")/255.0
                    left_pred = blink_model.predict([left_eye_input])[0]
                    right_pred = blink_model.predict([right_eye_input])[0]
                    # Inverted logic: 1 indicates closed
                    if left_pred == 1 and right_pred == 1:
                        eyes_open = False
                        eye_state = "Closed"
                    else:
                        eyes_open = True
                        eye_state = "Open"
                    cv2.putText(frame, f"Eye: {eye_state}", (x, y+h+20),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), FONT_THICKNESS)
            else:
                eyes_open = False

            # 6. Attendance logic: good if looking and eyes_open continuously
            if looking:
                if eyes_open:
                    closed_eye_consecutive = 0
                    if good_start_time is None:
                        good_start_time = current_time
                else:
                    closed_eye_consecutive += 1
                    if closed_eye_consecutive > MAX_CLOSED_FRAMES:
                        good_start_time = None
                # Display elapsed time on screen
                if good_start_time is not None:
                    elapsed = current_time - good_start_time
                    cv2.putText(frame, f"Good for {elapsed:.1f}s", (x, y-40),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0,255,0), FONT_THICKNESS)
                    # Check if attendance condition met
                    if elapsed >= ATTENDANCE_DURATION and not attendance_marked:
                        attendance_marked = True
                        attendance_marked_time = current_time
                        # Update CSV with attendance "Yes"
                        update_csv_attendance("Yes")
                else:
                    cv2.putText(frame, "Reset Timer", (x, y-40),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0,0,255), FONT_THICKNESS)
            else:
                good_start_time = None
                closed_eye_consecutive = 0

            # 7. Display Attendance Marked Message
            if attendance_marked and attendance_marked_time is not None:
                if current_time - attendance_marked_time <= ATTENDANCE_MESSAGE_DURATION:
                    # Add to logs only once
                    if not attendance_logs or attendance_logs[-1] != "Attendance Marked":
                        attendance_logs.append("Attendance Marked")
                        update_csv_attendance("Yes")

                    # Draw attendance marked text
                    text = "Attendance Marked"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, ATTENDANCE_FONT_SCALE, FONT_THICKNESS)
                    pos = (frame.shape[1]-tw-20, th+20)
                    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, ATTENDANCE_FONT_SCALE, (0,255,0), FONT_THICKNESS)
                else:
                    attendance_marked_time = None  # Only reset after time duration ends

        else:
            good_start_time = None
            closed_eye_consecutive = 0

        # Overlay a verifying/loading text on the bottom-center
        cv2.putText(frame, "Verifying...", (20, frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
def update_csv_attendance(attendance_value):
    with attendance_lock:
        if not os.path.isfile(CSV_FILE):
            print("CSV file not found.")
            return

        # Read all rows
        with open(CSV_FILE, 'r', newline='') as f:
            rows = list(csv.reader(f))

        if len(rows) < 2:
            print("CSV has no data rows to update.")
            return

        header = rows[0]
        last_row = rows[-1]

        # Ensure 'Attendance' column exists
        if "Attendance" not in header:
            print("'Attendance' column missing in header.")
            return

        # Update the last row's attendance value
        attendance_idx = header.index("Attendance")
        if len(last_row) <= attendance_idx:
            # Pad the row if it's too short
            last_row.extend([""] * (attendance_idx - len(last_row) + 1))

        last_row[attendance_idx] = attendance_value
        rows[-1] = last_row

        # Write everything back
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)


# -----------------------------
# Flask Routes
# -----------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    today = datetime.utcnow().strftime('%Y-%m-%d')
    global current_attendance, attendance_marked
    if request.method == 'POST':
        name = request.form.get('name')
        date = request.form.get('date')

        # Store initial record in memory
        current_attendance = {"name": name, "date": date, "Attendance": "No"}
        attendance_marked = False

        # Write initial attendance record to CSV
        file_exists = os.path.isfile(CSV_FILE)
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Name", "Date", "Attendance"])
            writer.writerow([name, date, "No"])

        return redirect(url_for('attendance'))
    return render_template('register.html', today=today)


@app.route('/attendance')
def attendance():
    return render_template('attendance.html', attendance_marked=attendance_marked)

@app.route('/video_feed')
def video_feed():
    global camera_process, frame_queue

    if camera_process is None or not camera_process.is_alive():
        frame_queue = Queue()
        camera_process = Process(target=generate_frames, args=(frame_queue,))
        camera_process.start()
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop', methods=['POST'])
def stop_feed():
    global camera_process
    if camera_process is not None:
        camera_process.terminate()
        camera_active = False
        camera_process.join()
        camera_process = None
    return '', 204

@app.route('/restart_feed', methods=['POST'])
def restart_feed():
    global camera_process
    if camera_process is not None:
        camera_process.terminate()
        camera_process.join()
    camera_process = Process(target=generate_frames)
    camera_process.start()
    return '', 204

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_active
    camera_active = not camera_active
    return jsonify({'status': 'ok', 'active': camera_active})

@app.route('/reset_verification', methods=['POST'])
def reset_verification():
    global attendance_marked, attendance_marked_time, current_attendance, camera_active
    attendance_marked = False
    attendance_marked_time = None
    current_attendance = []
    camera_active = True
    return jsonify({'status': 'reset'})

@app.route('/check_attendance')
def check_attendance():
    name = request.args.get('name')
    date = request.args.get('date')

    if not name or not date:
        return jsonify({'error': 'Missing name or date parameter'}), 400

    if not os.path.isfile(CSV_FILE):
        return jsonify({'error': 'Attendance file not found'}), 404

    with open(CSV_FILE, 'r') as f:
        reader = list(csv.reader(f))
        header = reader[0] if reader else []
        rows = reader[1:]

    # Indices
    try:
        name_idx = header.index("Name")
        date_idx = header.index("Date")
        attendance_idx = header.index("Attendance")
    except ValueError:
        return jsonify({'error': 'CSV header missing required fields'}), 500

    # Search for the latest matching record
    matched = None
    for row in reversed(rows):
        if len(row) > max(name_idx, date_idx, attendance_idx):
            if row[name_idx] == name and row[date_idx] == date:
                matched = row
                break

    if matched:
        return jsonify({'marked': matched[attendance_idx] == "Yes"})
    else:
        return jsonify({'marked': False, 'message': 'No matching record found'})
    

@app.route('/get_logs')
def get_logs():
    return jsonify({'logs': attendance_logs})

@app.route('/logs')
def logs():
    records = []
    if os.path.isfile(CSV_FILE):
        with open(CSV_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    return render_template('logs.html', records=records)

@app.route('/reset')
def reset():
    # For demo/testing: reset the attendance marker and current record
    global attendance_marked, current_attendance
    attendance_marked = False
    current_attendance = {}
    return redirect(url_for('index'))

if _name_ == '_main_':
    # Run Flask in debug mode if desired.
    app.run(host='0.0.0.0', port=5000, debug=True)