import cv2
import numpy as np
import os
import json
import time
from datetime import datetime

# --- CONFIGURATION ---
CONFIG = {
    "DATA_DIR": "attendance_data",
    "FACES_DIR": "attendance_data/face_templates",
    "LOGS_FILE": "attendance_data/attendance_logs.json",
    "LOGS_DAILY_DIR": "attendance_data/daily_logs",
    "FACE_TEMPLATE_SIZE": (128, 128),
    "SIMILARITY_THRESHOLD": 0.90, # 90% template matching / MSE similarity
    "ANTI_SPOOFING_PASS_COUNT": 2, # Requires 2 out of 3 liveness checks
    "HAARCASCADE_PATH": cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
    "STATUS_COLORS": {
        "pass": (0, 200, 0),    # Green
        "fail": (0, 0, 200),    # Red
        "wait": (255, 165, 0),  # Orange
        "neutral": (100, 100, 100), # Gray
        "message": (255, 255, 255) # White
    },
    "UI": {
        "CARD_COLOR": (30, 30, 30),
        "TEXT_COLOR": (200, 200, 200),
        "FONT": cv2.FONT_HERSHEY_SIMPLEX,
        "SHADOW_OFFSET": 5,
    }
}

# --- HELPER FUNCTIONS ---

def create_initial_directories():
    """Initializes necessary data storage directories and files."""
    os.makedirs(CONFIG["FACES_DIR"], exist_ok=True)
    os.makedirs(CONFIG["LOGS_DAILY_DIR"], exist_ok=True)
    if not os.path.exists(CONFIG["LOGS_FILE"]):
        with open(CONFIG["LOGS_FILE"], 'w') as f:
            json.dump({}, f)

def load_attendance_logs():
    """Loads the main attendance log file."""
    try:
        with open(CONFIG["LOGS_FILE"], 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def save_attendance_logs(logs):
    """Saves the main attendance log file."""
    with open(CONFIG["LOGS_FILE"], 'w') as f:
        json.dump(logs, f, indent=4)

def save_daily_log(employee_id, name, log_entry):
    """Saves a daily, per-employee log file for detailed records."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(CONFIG["LOGS_DAILY_DIR"], f"{employee_id}_{date_str}.json")
    
    daily_data = {}
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                daily_data = json.load(f)
        except json.JSONDecodeError:
            pass # Start fresh if file is corrupted

    daily_data[datetime.now().strftime("%H:%M:%S")] = log_entry
    
    with open(filename, 'w') as f:
        json.dump(daily_data, f, indent=4)

def draw_modern_card(frame, x, y, w, h, header_text="", status_text="", status_color=CONFIG["STATUS_COLORS"]["neutral"]):
    """Renders a modern, rounded card-style UI overlay."""
    
    # Rounded Rectangle for main card
    radius = 20
    thickness = -1
    color = CONFIG["UI"]["CARD_COLOR"]
    cv2.rectangle(frame, (x + radius, y), (x + w - radius, y + h), color, thickness)
    cv2.rectangle(frame, (x, y + radius), (x + w, y + h - radius), color, thickness)
    cv2.circle(frame, (x + radius, y + radius), radius, color, thickness)
    cv2.circle(frame, (x + w - radius, y + radius), radius, color, thickness)
    cv2.circle(frame, (x + radius, y + h - radius), radius, color, thickness)
    cv2.circle(frame, (x + w - radius, y + h - radius), radius, color, thickness)

    # Shadow effect (simple offset)
    shadow_color = (0, 0, 0)
    shadow_offset = CONFIG["UI"]["SHADOW_OFFSET"]
    cv2.rectangle(frame, (x + shadow_offset + radius, y + shadow_offset), 
                  (x + w - radius + shadow_offset, y + h + shadow_offset), shadow_color, thickness)
    
    # Redraw main card over shadow
    cv2.rectangle(frame, (x + radius, y), (x + w - radius, y + h), color, thickness)
    cv2.rectangle(frame, (x, y + radius), (x + w, y + h - radius), color, thickness)
    cv2.circle(frame, (x + radius, y + radius), radius, color, thickness)
    cv2.circle(frame, (x + w - radius, y + radius), radius, color, thickness)
    cv2.circle(frame, (x + radius, y + h - radius), radius, color, thickness)
    cv2.circle(frame, (x + w - radius, y + h - radius), radius, color, thickness)

    # Header text
    cv2.putText(frame, header_text, (x + 20, y + 40), CONFIG["UI"]["FONT"], 0.8, CONFIG["UI"]["TEXT_COLOR"], 2)

    # Status text
    cv2.putText(frame, status_text, (x + 20, y + 80), CONFIG["UI"]["FONT"], 0.7, status_color, 2)
    
    # Draw a line separator
    cv2.line(frame, (x + 10, y + 55), (x + w - 10, y + 55), status_color, 2)


# --- CORE LOGIC CLASSES ---

class EmotionDetector:
    """
    Detects emotion using lightweight, brightness/variance-based heuristics.
    Heuristics:
    1. Happiness/Positive: Brighter lower face (cheeks/mouth) compared to upper (eyes/forehead).
    2. Stress/Concentration: High variance/texture in forehead/eyes, combined with darker lower face.
    3. Sad/Neutral: Low variance and relatively uniform brightness.
    """
    EMOTION_MAP = {
        "happy": {"msg": "Feeling good, have a productive day! 😊", "color": (50, 200, 50)},
        "stressed": {"msg": "Deep focus, take a short break if needed. 😬", "color": (50, 50, 200)},
        "sad": {"msg": "Hope your day gets better. We're here for you. 😟", "color": (200, 50, 50)},
        "neutral": {"msg": "Ready to work! Welcome. 👍", "color": (200, 200, 50)}
    }
    
    def detect(self, face_roi):
        """Analyzes face ROI for emotion."""
        if face_roi is None or face_roi.size == 0:
            return "neutral", self.EMOTION_MAP["neutral"]

        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        h, w = gray_face.shape

        # 1. Brightness Comparison (Upper vs Lower Face)
        upper_face = gray_face[0:h//2, :]
        lower_face = gray_face[h//2:h, :]
        
        # Calculate mean brightness
        mean_upper = np.mean(upper_face)
        mean_lower = np.mean(lower_face)
        brightness_ratio = mean_lower / (mean_upper + 1e-6) # Add epsilon to prevent division by zero

        # 2. Variance-based Stress/Texture (Higher variance can indicate strain or focus)
        # Use variance of the whole face for overall texture/movement micro-expressions
        face_variance = np.var(gray_face)

        # Heuristic Thresholds (Adjustable)
        HAPPY_RATIO_THRESH = 1.10  # Lower face 10% brighter than upper
        STRESS_VAR_THRESH = 1500   # High variance threshold for stress/focus
        
        emotion_key = "neutral"

        if brightness_ratio > HAPPY_RATIO_THRESH:
            emotion_key = "happy"
        elif face_variance > STRESS_VAR_THRESH:
            emotion_key = "stressed"
        # A low variance might indicate sadness/lack of expression, but we'll default 
        # to neutral unless a specific low-variance sad-pattern is found (omitting for lightweight)
        # elif face_variance < 500:
        #     emotion_key = "sad"
        else:
            emotion_key = "neutral"

        return emotion_key, self.EMOTION_MAP[emotion_key]

class AntiSpoofing:
    """
    Performs liveness detection using lightweight multi-factor heuristics.
    Factors: Motion, Blink Pattern, Texture (Laplacian variance).
    Requires 2-of-3 factors to pass.
    """
    def __init__(self):
        self.prev_centroid = None
        self.frame_history = []
        self.max_history = 5
        self.blink_frames = 0
        self.blink_count = 0
        self.blink_active_threshold = 2 # frames dark region persists
        self.blink_reset_threshold = 10 # frames to reset blink count after a detection
        self.last_blink_frame = 0

    def analyze(self, frame, face_rect):
        """Analyzes frame and face for liveness indicators."""
        x, y, w, h = face_rect
        face_roi = frame[y:y+h, x:x+w]
        
        # Convert to grayscale for consistent analysis
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # --- 1. Motion Analysis (Centroid Variance) ---
        motion_pass = False
        current_centroid = (x + w // 2, y + h // 2)
        
        if self.prev_centroid:
            # Calculate distance moved
            distance = np.sqrt((current_centroid[0] - self.prev_centroid[0])**2 + 
                               (current_centroid[1] - self.prev_centroid[1])**2)
            
            # Simple threshold for significant movement between frames (5 pixels)
            if distance > 5:
                motion_pass = True
        
        self.prev_centroid = current_centroid
        
        # --- 2. Blink Detection (Dark Region Analysis - Extremely lightweight) ---
        blink_pass = False
        # Simplistic heuristic: Look for temporary, significant darkening of the eye region
        # This requires more accurate eye region detection, but for a lightweight, single-file 
        # solution, we'll approximate: look for rapid, brief decrease in mean face brightness.
        self.frame_history.append(np.mean(gray))
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)

        if len(self.frame_history) == self.max_history:
            current_mean = self.frame_history[-1]
            # Compare current frame to the average of the previous history
            historical_mean = np.mean(self.frame_history[:-1])
            
            # If current frame is significantly darker (e.g., due to eyes closing)
            # Threshold: 10% drop in brightness
            if historical_mean - current_mean > historical_mean * 0.10: 
                self.blink_frames += 1
            else:
                if self.blink_frames >= 1: # A drop was detected
                    self.blink_count += 1
                    self.last_blink_frame = time.time()
                self.blink_frames = 0
        
        # Blink pass is met if at least one blink has been detected recently
        if self.blink_count > 0 and (time.time() - self.last_blink_frame < 5):
            blink_pass = True

        # --- 3. Texture Analysis (Laplacian Variance) ---
        texture_pass = False
        # Calculate Laplacian variance. High variance indicates texture/detail (real skin).
        # Low variance (blur, high smoothness) can indicate a printout or screen.
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Threshold: Prints/Screens usually have a low laplacian variance
        TEXTURE_VAR_THRESH = 100.0 # Varies heavily based on camera/lighting
        if laplacian_var > TEXTURE_VAR_THRESH:
            texture_pass = True

        # --- Multi-factor Verdict ---
        pass_count = sum([motion_pass, blink_pass, texture_pass])
        is_liveness_pass = pass_count >= CONFIG["ANTI_SPOOFING_PASS_COUNT"]
        
        status = {
            "motion": motion_pass,
            "blink": blink_pass,
            "texture": texture_pass,
            "pass_count": pass_count,
            "liveness_pass": is_liveness_pass
        }
        
        return status

class ModernAttendanceSystem:
    """
    Manages the overall system, including face templates, recognition, 
    attendance logic, and UI rendering.
    """
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(CONFIG["HAARCASCADE_PATH"])
        self.emotion_detector = EmotionDetector()
        self.anti_spoofing = AntiSpoofing()
        
        self.attendance_logs = load_attendance_logs()
        self.known_faces = self._load_known_faces()
        
        self.mode = "CHECK" # CHECK or ENROLL
        self.enroll_id = None
        self.enroll_name = None
        self.enroll_template = None
        
    def _load_known_faces(self):
        """Loads all stored face templates from the directory."""
        known_faces = {}
        for filename in os.listdir(CONFIG["FACES_DIR"]):
            if filename.endswith(".npy"):
                employee_id = filename.split('_')[0]
                name = filename.split('_')[1].split('.npy')[0]
                filepath = os.path.join(CONFIG["FACES_DIR"], filename)
                
                try:
                    template = np.load(filepath)
                    # Store as (template, name) tuple
                    known_faces[employee_id] = (template, name)
                except Exception as e:
                    print(f"Error loading template {filename}: {e}")
        return known_faces

    def start_enrollment(self, employee_id, name):
        """Initializes the system for employee enrollment."""
        # Sanitize inputs
        employee_id = str(employee_id).strip()
        name = str(name).strip()
        
        if not employee_id or not name:
            return "Enrollment Failed: ID and Name cannot be empty.", CONFIG["STATUS_COLORS"]["fail"]

        if employee_id in self.known_faces:
            return f"Enrollment Failed: Employee ID {employee_id} already exists.", CONFIG["STATUS_COLORS"]["fail"]
            
        self.mode = "ENROLL"
        self.enroll_id = employee_id
        self.enroll_name = name
        self.enroll_template = None
        
        return f"Ready to enroll ID: {employee_id}, Name: {name}. Look at the camera.", CONFIG["STATUS_COLORS"]["wait"]

    def process_frame(self, frame):
        """Main frame processing loop."""
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Face detection (Lightweight Haarcascade)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        # Default UI state
        system_status = f"System Mode: {self.mode}"
        status_color = CONFIG["STATUS_COLORS"]["neutral"]
        face_detected = False
        
        # Largest face processing (Focus on the primary face)
        if len(faces) > 0:
            face_detected = True
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3]) # Get largest face
            
            # Draw face bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_roi = frame[y:y+h, x:x+w]
            
            # --- Anti-Spoofing & Emotion (Requires a face) ---
            liveness_status = self.anti_spoofing.analyze(frame, (x, y, w, h))
            emotion_key, emotion_data = self.emotion_detector.detect(face_roi)

            # --- UI: Liveness Indicators ---
            y_offset = 200
            liveness_pass = liveness_status["liveness_pass"]
            progress_val = liveness_status["pass_count"] / CONFIG["ANTI_SPOOFING_PASS_COUNT"]
            
            for i, (key, is_pass) in enumerate(liveness_status.items()):
                if key in ["motion", "blink", "texture"]:
                    text = f"{key.capitalize()}: {'PASS' if is_pass else 'FAIL'}"
                    color = CONFIG["STATUS_COLORS"]["pass"] if is_pass else CONFIG["STATUS_COLORS"]["fail"]
                    cv2.putText(frame, text, (10, y_offset + i*30), CONFIG["UI"]["FONT"], 0.6, color, 1)
            
            # Liveness Progress Bar (simple rectangle)
            bar_w = 200
            bar_h = 10
            bar_x = 10
            bar_y = y_offset + 100
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1) # Background
            progress_bar_color = CONFIG["STATUS_COLORS"]["pass"] if liveness_pass else CONFIG["STATUS_COLORS"]["wait"]
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress_val), bar_y + bar_h), progress_bar_color, -1)
            cv2.putText(frame, "LIVENESS", (bar_x, bar_y - 5), CONFIG["UI"]["FONT"], 0.5, CONFIG["UI"]["TEXT_COLOR"], 1)

            # --- ENROLLMENT MODE LOGIC ---
            if self.mode == "ENROLL":
                system_status = f"ENROLLING: ID {self.enroll_id} | Name: {self.enroll_name}"
                status_color = CONFIG["STATUS_COLORS"]["wait"]
                
                # Check for liveness before capturing template
                if liveness_pass:
                    # Resize ROI to a standard template size (128x128)
                    self.enroll_template = cv2.resize(face_roi, CONFIG["FACE_TEMPLATE_SIZE"])
                    
                    # Display the captured template preview
                    preview_size = 150
                    frame[10:10+preview_size, frame.shape[1]-10-preview_size:frame.shape[1]-10] = self.enroll_template[:preview_size, :preview_size]
                    cv2.putText(frame, "Template Captured!", (frame.shape[1]-200, 20), CONFIG["UI"]["FONT"], 0.6, CONFIG["STATUS_COLORS"]["pass"], 2)
                    
                    # The user will press 'S' to save this template
                    system_status += " | Press 'S' to Save."

                else:
                    self.enroll_template = None
                    system_status += f" | Status: Liveness Failed ({liveness_status['pass_count']}/{CONFIG['ANTI_SPOOFING_PASS_COUNT']})"

            # --- CHECK-IN/CHECK-OUT MODE LOGIC ---
            elif self.mode == "CHECK":
                if liveness_pass:
                    # 1. Prepare candidate template
                    candidate_template = cv2.resize(face_roi, CONFIG["FACE_TEMPLATE_SIZE"])
                    
                    # 2. Recognition
                    recognized_id, recognized_name, similarity = self._recognize_face(candidate_template)
                    
                    if recognized_id:
                        status_color = CONFIG["STATUS_COLORS"]["pass"]
                        system_status = f"Recognized: {recognized_name} (ID: {recognized_id}) | Sim: {similarity:.2f}"
                        
                        # 3. Attendance Logic
                        log_message, log_color = self._mark_attendance(recognized_id, recognized_name, emotion_key, emotion_data)
                        
                        # Use the specific log message color for the main status display
                        status_color = log_color
                        # Prepend the emotion message to the log message
                        system_status = f"Emotion: {emotion_data['msg']} | " + log_message
                        
                    else:
                        status_color = CONFIG["STATUS_COLORS"]["fail"]
                        system_status = f"Recognition Failed: Not in database or Sim < {CONFIG['SIMILARITY_THRESHOLD']:.2f}"
                else:
                    status_color = CONFIG["STATUS_COLORS"]["wait"]
                    system_status = f"Awaiting Liveness Check ({liveness_status['pass_count']}/{CONFIG['ANTI_SPOOFING_PASS_COUNT']})"
        
        # If no face is detected
        if not face_detected:
            self.anti_spoofing.prev_centroid = None # Reset motion tracking
            if self.mode == "CHECK":
                system_status = "Awaiting Face Recognition..."
            elif self.mode == "ENROLL":
                system_status = "Place face in center to enroll."

        # --- Draw Main UI Card ---
        card_w, card_h = 500, 120
        card_x = frame.shape[1] // 2 - card_w // 2
        card_y = frame.shape[0] - card_h - 10
        draw_modern_card(frame, card_x, card_y, card_w, card_h, 
                         header_text="Intelligent Attendance System", 
                         status_text=system_status, 
                         status_color=status_color)
        
        return frame

    def _recognize_face(self, candidate_template):
        """Compares candidate template against all known faces."""
        candidate_gray = cv2.cvtColor(candidate_template, cv2.COLOR_BGR2GRAY).astype(np.float32)
        best_match = None
        max_similarity = 0
        
        for employee_id, (template, name) in self.known_faces.items():
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY).astype(np.float32)
            
            # --- Lightweight Template Comparison (MSE Similarity) ---
            # 1. Template Matching (Normalized Cross Correlation)
            res = cv2.matchTemplate(candidate_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            similarity = np.max(res)
            
            # 2. Add a fallback/blend with Inverse MSE for robust similarity
            # Calculate MSE
            mse = np.mean((candidate_gray - template_gray)**2)
            
            # Inverse MSE scaling (smaller MSE is better) - scale to [0, 1] range for simplicity
            # A perfect match (MSE=0) would be 1.0. A high MSE (e.g., 2000) would be near 0.
            # Using an exponential falloff: Sim_MSE = exp(-k * MSE)
            k = 0.0005 # A tunable constant
            similarity_mse = np.exp(-k * mse)

            # Blend the two scores (simple average)
            blended_similarity = (similarity + similarity_mse) / 2

            if blended_similarity > max_similarity:
                max_similarity = blended_similarity
                best_match = (employee_id, name)
                
        if max_similarity >= CONFIG["SIMILARITY_THRESHOLD"]:
            return best_match[0], best_match[1], max_similarity
        
        return None, None, max_similarity

    def _mark_attendance(self, employee_id, name, emotion_key, emotion_data):
        """Implements the check-in/check-out and logging logic."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Ensure employee entry exists for the day
        if employee_id not in self.attendance_logs:
            self.attendance_logs[employee_id] = {}
            
        if current_date not in self.attendance_logs[employee_id]:
            self.attendance_logs[employee_id][current_date] = {
                "name": name,
                "check_in": None,
                "check_out": None,
                "duration": "0h 0m",
                "emotion_in": None,
                "emotion_out": None
            }
        
        log = self.attendance_logs[employee_id][current_date]
        
        status_color = CONFIG["STATUS_COLORS"]["neutral"]
        
        # --- Check-in Logic ---
        if log["check_in"] is None:
            log["check_in"] = current_time
            log["emotion_in"] = emotion_key
            save_attendance_logs(self.attendance_logs)
            save_daily_log(employee_id, name, {"action": "check_in", "emotion": emotion_key, "message": emotion_data['msg']})
            status_color = CONFIG["STATUS_COLORS"]["pass"]
            return f"CHECK-IN SUCCESSFUL at {current_time}.", status_color
            
        # --- Check-out Logic ---
        elif log["check_in"] is not None and log["check_out"] is None:
            log["check_out"] = current_time
            log["emotion_out"] = emotion_key
            
            # Calculate duration
            in_dt = datetime.strptime(log["check_in"], "%H:%M:%S")
            out_dt = datetime.strptime(log["check_out"], "%H:%M:%S")
            duration = out_dt - in_dt
            
            total_seconds = int(duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            
            log["duration"] = f"{hours}h {minutes}m"
            
            save_attendance_logs(self.attendance_logs)
            save_daily_log(employee_id, name, {"action": "check_out", "emotion": emotion_key, "message": emotion_data['msg'], "duration": log["duration"]})
            status_color = CONFIG["STATUS_COLORS"]["pass"]
            return f"CHECK-OUT SUCCESSFUL at {current_time}. Total Time: {log['duration']}", status_color
            
        # --- Already Done ---
        elif log["check_in"] is not None and log["check_out"] is not None:
            status_color = CONFIG["STATUS_COLORS"]["wait"]
            return f"Attendance complete for today. Duration: {log['duration']}", status_color

        return "Attendance System Error.", CONFIG["STATUS_COLORS"]["fail"]


    def save_enrollment(self):
        """Saves the captured face template to the disk and updates the system."""
        if self.mode == "ENROLL" and self.enroll_template is not None:
            filename = f"{self.enroll_id}_{self.enroll_name}.npy"
            filepath = os.path.join(CONFIG["FACES_DIR"], filename)
            
            # Save the template (NumPy array)
            np.save(filepath, self.enroll_template)
            
            # Update in-memory cache
            self.known_faces[self.enroll_id] = (self.enroll_template, self.enroll_name)
            
            # Reset mode
            self.mode = "CHECK"
            self.enroll_id = None
            self.enroll_name = None
            self.enroll_template = None
            
            return "Enrollment Saved: System returned to CHECK mode.", CONFIG["STATUS_COLORS"]["pass"]
        else:
            return "Enrollment Failed: No template captured or not in ENROLL mode.", CONFIG["STATUS_COLORS"]["fail"]

def main():
    """Main execution function for the Attendance System application."""
    
    # 1. Setup
    create_initial_directories()
    system = ModernAttendanceSystem()

    # 2. Camera Setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cv2.namedWindow('Attendance System', cv2.WINDOW_AUTOSIZE)

    print("\n--- System Initialized ---")
    print(f"Known Employees: {len(system.known_faces)}")
    print("Press 'E' to start Enrollment.")
    print("Press 'C' to switch to Check-in/Check-out mode.")
    print("In Enrollment mode, press 'S' to save the template.")
    print("Press 'Q' to quit.")
    print("--------------------------\n")

    # 3. Main Loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Flip the frame for a mirror effect (more intuitive UI)
        frame = cv2.flip(frame, 1)

        # Process and draw on the frame
        processed_frame = system.process_frame(frame)

        # Display the frame
        cv2.imshow('Attendance System', processed_frame)

        # 4. Keyboard Input Handling
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # Enrollment mode switch
        elif key == ord('e'):
            # Prompt for ID and Name
            employee_id = input("Enter Employee ID (e.g., 1001): ").strip()
            name = input("Enter Employee Name: ").strip()
            
            message, color = system.start_enrollment(employee_id, name)
            print(f"MODE CHANGE: {message}")
            
        # Check mode switch
        elif key == ord('c'):
            system.mode = "CHECK"
            system.enroll_id = None
            system.enroll_name = None
            system.enroll_template = None
            print("MODE CHANGE: Switched to CHECK (Attendance) mode.")
            
        # Save Enrollment
        elif key == ord('s'):
            if system.mode == "ENROLL":
                message, color = system.save_enrollment()
                print(f"ENROLLMENT: {message}")
            else:
                print("ACTION FAILED: 'S' key is only active in ENROLL mode.")

    # 5. Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n--- System Shutdown ---")
    print(f"Final logs saved to {CONFIG['LOGS_FILE']}")

if __name__ == "__main__":
    main()