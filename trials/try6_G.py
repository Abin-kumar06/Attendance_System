import cv2
import numpy as np
import os
import datetime
import json

# EmotionDetector class: Handles emotion detection using lightweight heuristics
class EmotionDetector:
    def __init__(self):
        pass  # No initialization needed

    def detect(self, gray_face):
        """
        Detect emotion based on brightness comparison and variance.
        - Upper/lower face brightness for happy/sad/neutral.
        - High variance indicates stress.
        Returns emotion and a positive message with emoji.
        """
        h, w = gray_face.shape
        upper = gray_face[:h//2, :]
        lower = gray_face[h//2:, :]
        mean_upper = np.mean(upper)
        mean_lower = np.mean(lower)
        variance = cv2.Laplacian(gray_face, cv2.CV_64F).var()

        if variance > 1000:  # Arbitrary threshold for high variance indicating stress
            emotion = "stressed"
            message = "Take a deep breath! 😌"
        elif mean_lower > mean_upper + 10:  # Lower face brighter (smile)
            emotion = "happy"
            message = "Great to see you smiling! 😊"
        elif mean_upper > mean_lower + 10:  # Upper face brighter (frown)
            emotion = "sad"
            message = "Cheer up! 🌟"
        else:
            emotion = "neutral"
            message = "Looking good! 👍"
        
        return emotion, message

# AntiSpoofing class: Handles multi-factor anti-spoofing analysis
class AntiSpoofing:
    def __init__(self):
        self.centroids = []  # For motion tracking
        self.blink_count = 0  # Count of detected blinks
        self.prev_eyes_detected = False  # Previous frame's eye detection state
        self.frames = 0  # Frame counter
        self.texture_pass = False  # Texture check result

    def update(self, frame, face_rect, eyes_detected):
        """
        Update anti-spoofing factors over frames.
        - Motion: Add centroid if face detected.
        - Blink: Check change in eye detection.
        - Texture: Laplacian variance on face (once).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if face_rect is None:
            self.frames = 0  # Reset if no face to require consistent presence
            return
        
        # Motion: Track centroid
        x, y, w, h = face_rect
        cx = x + w / 2
        cy = y + h / 2
        self.centroids.append((cx, cy))
        
        # Blink: Detect closure if eyes were detected before but not now
        if self.prev_eyes_detected and not eyes_detected:
            self.blink_count += 1
        self.prev_eyes_detected = eyes_detected
        
        # Texture: Check once
        if not self.texture_pass:
            face_gray = gray[y:y+h, x:x+w]
            var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
            self.texture_pass = var > 50  # Arbitrary threshold for real texture vs. photo
        
        self.frames += 1

    def is_live(self):
        """
        Check if live based on accumulated data.
        Requires at least 10 frames and 2/3 factors passing.
        Returns live status and individual factor statuses.
        """
        if self.frames < 10:
            return False, False, False, False
        
        # Motion: Std dev of last 5 centroids
        motion_pass = False
        if len(self.centroids) >= 5:
            recent_centroids = np.array(self.centroids[-5:])
            std = np.std(recent_centroids, axis=0)
            motion_pass = np.mean(std) > 5  # Arbitrary pixel threshold for movement
        
        # Blink
        blink_pass = self.blink_count > 0
        
        # Texture
        texture_pass = self.texture_pass
        
        passes = sum([motion_pass, blink_pass, texture_pass])
        return passes >= 2, motion_pass, blink_pass, texture_pass

# ModernAttendanceSystem class: Main system handling enrollment, attendance, and UI
class ModernAttendanceSystem:
    def __init__(self):
        """
        Initialize cascades, directories, and load attendance data.
        """
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.template_dir = 'templates'
        self.attendance_file = 'attendance.json'
        os.makedirs(self.template_dir, exist_ok=True)
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w') as f:
                json.dump([], f)
        self.load_attendance()
        self.emotion_detector = EmotionDetector()

    def load_attendance(self):
        """Load attendance records from JSON."""
        with open(self.attendance_file, 'r') as f:
            self.attendance = json.load(f)

    def save_attendance(self):
        """Save attendance records to JSON."""
        with open(self.attendance_file, 'w') as f:
            json.dump(self.attendance, f, indent=4)

    def get_today(self):
        """Get today's date in ISO format."""
        return datetime.date.today().isoformat()

    def log_attendance(self, name, mode, emotion, message):
        """
        Log check-in or check-out with emotion and message.
        Handles rules: no duplicate check-in, check-out only after check-in.
        Calculates hours on check-out.
        Returns True if successful.
        """
        today = self.get_today()
        time_now = datetime.datetime.now().strftime('%H:%M:%S')
        
        for rec in self.attendance:
            if rec['date'] == today and rec['employee'] == name:
                if mode == 'in':
                    return False  # Already checked in
                elif mode == 'out':
                    rec['check_out'] = time_now
                    rec['emotion_out'] = emotion
                    rec['msg_out'] = message
                    in_time = datetime.datetime.strptime(rec['check_in'], '%H:%M:%S')
                    out_time = datetime.datetime.strptime(time_now, '%H:%M:%S')
                    hours = (out_time - in_time).total_seconds() / 3600
                    rec['hours'] = round(hours, 2)
                    return True
        
        if mode == 'out':
            return False  # No check-in today
        
        # New check-in
        new_rec = {
            'date': today,
            'employee': name,
            'check_in': time_now,
            'emotion_in': emotion,
            'msg_in': message
        }
        self.attendance.append(new_rec)
        return True

    def recognize_face(self, gray_face):
        """
        Recognize face using normalized correlation template matching.
        Returns name if similarity > 0.8 threshold, else None.
        """
        face_resized = cv2.resize(gray_face, (128, 128))
        best_name = None
        best_score = 0
        for filename in os.listdir(self.template_dir):
            if filename.endswith('.jpg'):
                template_path = os.path.join(self.template_dir, filename)
                template = cv2.imread(template_path, 0)
                res = cv2.matchTemplate(face_resized, template, cv2.TM_CCORR_NORMED)
                score = res[0][0]
                if score > best_score:
                    best_score = score
                    best_name = filename[:-4]
        if best_score > 0.8:
            return best_name
        return None

    def draw_rounded_rect(self, img, pt1, pt2, color, thickness=2, radius=10):
        """
        Draw a rounded rectangle for modern UI look.
        """
        x1, y1 = pt1
        x2, y2 = pt2
        # Top left
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        # Top right
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        # Bottom right
        cv2.line(img, (x2 - radius, y2), (x1 + radius, y2), color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
        # Bottom left
        cv2.line(img, (x1, y2 - radius), (x1, y1 + radius), color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)

    def run(self):
        """
        Main loop: Camera preview, UI rendering, modes handling.
        Keys: 'e' enroll, 'i' check-in, 'o' check-out, 'q' quit.
        """
        print("Commands: Press 'e' to enroll (enter name in console), 'i' for check-in, 'o' for check-out, 'q' to quit.")
        cap = cv2.VideoCapture(0)
        mode = None
        anti_spoof = None
        status_message = ""
        message_timer = 0  # To display status for a few frames

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            face_rect = None
            eyes_detected = False
            if len(faces) == 1:
                face_rect = faces[0]
                x, y, w, h = face_rect
                self.draw_rounded_rect(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 20)
                eye_roi_gray = gray[y:y + h, x:x + w]
                eyes = self.eye_cascade.detectMultiScale(eye_roi_gray, 1.3, 5)
                eyes_detected = len(eyes) >= 1
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 0, 255), 2)

            # Modern card-style overlay (bottom bar)
            card_y = frame.shape[0] - 100
            cv2.rectangle(frame, (10, card_y), (frame.shape[1] - 10, frame.shape[0] - 10), (50, 50, 50), -1)  # Dark gray card
            cv2.putText(frame, f"Mode: {mode.upper() if mode else 'IDLE'}", (20, card_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if mode in ['in', 'out']:
                if anti_spoof is None:
                    anti_spoof = AntiSpoofing()
                anti_spoof.update(frame, face_rect, eyes_detected)
                is_live, motion_pass, blink_pass, texture_pass = anti_spoof.is_live()

                # Status indicators
                cv2.putText(frame, "Liveness:", (20, card_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # Motion indicator
                col = (0, 255, 0) if motion_pass else (0, 0, 255)
                cv2.circle(frame, (150, card_y + 60), 10, col, -1)
                cv2.putText(frame, "Motion", (170, card_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # Blink indicator
                col = (0, 255, 0) if blink_pass else (0, 0, 255)
                cv2.circle(frame, (250, card_y + 60), 10, col, -1)
                cv2.putText(frame, "Blink", (270, card_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # Texture indicator
                col = (0, 255, 0) if texture_pass else (0, 0, 255)
                cv2.circle(frame, (350, card_y + 60), 10, col, -1)
                cv2.putText(frame, "Texture", (370, card_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Progress bar
                passes = sum([motion_pass, blink_pass, texture_pass])
                bar_width = int((passes / 3.0) * 200)
                cv2.rectangle(frame, (20, card_y + 80), (220, card_y + 95), (100, 100, 100), -1)  # Background
                cv2.rectangle(frame, (20, card_y + 80), (20 + bar_width, card_y + 95), (0, 255, 0), -1)  # Fill

                if is_live and face_rect:
                    x, y, w, h = face_rect
                    gray_face = gray[y:y + h, x:x + w]
                    name = self.recognize_face(gray_face)
                    if name:
                        emotion, msg = self.emotion_detector.detect(cv2.resize(gray_face, (128, 128)))
                        success = self.log_attendance(name, mode, emotion, msg)
                        if success:
                            status_message = f"{mode.upper()} successful for {name}. {msg}"
                            message_color = (0, 255, 0)
                        else:
                            status_message = "Action invalid (already done or no check-in)."
                            message_color = (0, 0, 255)
                        self.save_attendance()
                        mode = None
                        anti_spoof = None
                        message_timer = 30  # Display for 30 frames
                    else:
                        status_message = "Unknown face."
                        message_color = (0, 0, 255)
                        message_timer = 10

            # Display status message if active
            if message_timer > 0:
                cv2.putText(frame, status_message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, message_color, 2)
                message_timer -= 1

            cv2.imshow('Modern Attendance System', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('i'):
                mode = 'in'
                anti_spoof = None
            elif key == ord('o'):
                mode = 'out'
                anti_spoof = None
            elif key == ord('e'):
                if face_rect:
                    name = input("Enter employee name: ").strip()
                    if name:
                        x, y, w, h = face_rect
                        gray_face = gray[y:y + h, x:x + w]
                        face_resized = cv2.resize(gray_face, (128, 128))
                        path = os.path.join(self.template_dir, f"{name}.jpg")
                        cv2.imwrite(path, face_resized)
                        status_message = f"Enrolled {name} successfully."
                        message_color = (0, 255, 0)
                        message_timer = 30
                else:
                    status_message = "No face detected for enrollment."
                    message_color = (0, 0, 255)
                    message_timer = 30

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = ModernAttendanceSystem()
    system.run()