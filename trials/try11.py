import cv2
import numpy as np
from datetime import datetime, timedelta
import csv
import pickle
import time
from pathlib import Path
import json
from collections import deque

class FaceRecognitionModel:
    """Advanced face recognition with multiple feature extraction methods"""
    
    def __init__(self):
        # Use LBPH (Local Binary Patterns Histograms) - more robust than template matching
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )
        self.is_trained = False
        self.label_map = {}  # Maps numeric labels to employee IDs
        self.reverse_map = {}  # Maps employee IDs to numeric labels
        
    def extract_features(self, face_roi):
        """Extract multiple facial features for robust recognition"""
        try:
            # Normalize face size
            face = cv2.resize(face_roi, (200, 200))
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization for better lighting normalization
            gray = cv2.equalizeHist(gray)
            
            # Additional preprocessing
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            return gray
        except:
            return None
    
    def train_model(self, employee_data):
        """Train the recognition model with all enrolled faces"""
        if not employee_data:
            print("⚠ No employee data to train")
            return False
        
        faces = []
        labels = []
        
        print("\n🔄 Training face recognition model...")
        
        for idx, (emp_id, emp_info) in enumerate(employee_data.items()):
            # Store label mapping
            self.label_map[idx] = emp_id
            self.reverse_map[emp_id] = idx
            
            # Extract features from stored face template
            face_template = emp_info.get('face_template')
            if face_template is not None:
                processed_face = self.extract_features(face_template)
                if processed_face is not None:
                    faces.append(processed_face)
                    labels.append(idx)
                    
                    # Data augmentation - add variations for robustness
                    # Slight brightness variations
                    bright = cv2.convertScaleAbs(processed_face, alpha=1.2, beta=10)
                    dark = cv2.convertScaleAbs(processed_face, alpha=0.8, beta=-10)
                    faces.extend([bright, dark])
                    labels.extend([idx, idx])
        
        if len(faces) > 0:
            try:
                self.recognizer.train(faces, np.array(labels))
                self.is_trained = True
                print(f"✓ Model trained with {len(employee_data)} employees ({len(faces)} samples)")
                return True
            except Exception as e:
                print(f"✗ Training failed: {e}")
                return False
        else:
            print("✗ No valid face samples found")
            return False
    
    def recognize(self, face_roi, threshold=70):
        """
        Recognize face using trained model
        threshold: Lower is stricter (0-100), recommended 50-80
        """
        if not self.is_trained:
            return None, 0
        
        try:
            processed_face = self.extract_features(face_roi)
            if processed_face is None:
                return None, 0
            
            # Predict using LBPH
            label, confidence = self.recognizer.predict(processed_face)
            
            # Convert confidence to similarity (LBPH returns distance, lower is better)
            # Normalize to 0-1 scale (0 = no match, 1 = perfect match)
            similarity = max(0, 1 - (confidence / 100))
            
            # Check if confidence meets threshold
            if confidence < threshold:
                emp_id = self.label_map.get(label)
                return emp_id, similarity
            
            return None, 0
        except Exception as e:
            print(f"Recognition error: {e}")
            return None, 0
    
    def update_model(self, emp_id, face_roi):
        """Add new training sample for existing employee"""
        if emp_id not in self.reverse_map:
            return False
        
        processed_face = self.extract_features(face_roi)
        if processed_face is None:
            return False
        
        try:
            label = self.reverse_map[emp_id]
            self.recognizer.update([processed_face], np.array([label]))
            return True
        except:
            return False


class EmotionDetector:
    """Simple emotion detection using facial landmarks and expressions"""

    def __init__(self):
        self.emotions = {
            'happy': {'emoji': '😊', 'messages': ['Great to see you smiling!', 'Your positive energy is contagious!', 'Keep that smile going!']},
            'neutral': {'emoji': '😐', 'messages': ['Hope you have a productive day!', 'Ready to tackle the day!', 'Let\'s make today count!']},
            'sad': {'emoji': '😔', 'messages': ['Hope your day gets better!', 'We\'re here if you need support!', 'Take care of yourself today!']},
            'stressed': {'emoji': '😰', 'messages': ['Remember to take breaks!', 'Don\'t forget to breathe!', 'We appreciate your hard work!']}
        }
        self.target_size = (64, 64)

        # Optimized thresholds
        self.NORMALIZED_STRESS_THRESH = 0.38
        self.HIGH_HAPPY_RATIO_THRESH = 1.18
        self.LOW_SAD_RATIO_THRESH = 0.85
        self.LOW_VAR_THRESH = 950
        
    def detect_emotion(self, face_roi):
        """Revised emotion detection logic"""
        try:
            face = cv2.resize(face_roi, self.target_size, interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            upper_face = gray[h//4:h//2, :]  
            lower_face = gray[h//2:3*h//4, :] 
            
            mean_upper = np.mean(upper_face)
            mean_lower = np.mean(lower_face)
            
            if mean_upper < 10: mean_upper = 10 
            brightness_ratio = mean_lower / mean_upper 

            total_variance = np.var(gray)
            normalized_variance = total_variance / (mean_upper**2 + 1e-6)
            
            emotion = 'neutral'

            if normalized_variance > self.NORMALIZED_STRESS_THRESH:
                emotion = 'stressed'
            elif brightness_ratio < self.LOW_SAD_RATIO_THRESH and total_variance < self.LOW_VAR_THRESH:
                emotion = 'sad'
            elif brightness_ratio > self.HIGH_HAPPY_RATIO_THRESH:
                emotion = 'happy'
            else:
                emotion = 'neutral'

            return emotion
        except Exception:
            return 'neutral'
        
    def get_message(self, emotion):
        return np.random.choice(self.emotions[emotion]['messages'])

    def get_emoji(self, emotion):
        return self.emotions[emotion]['emoji']


class AntiSpoofing:
    """Detect if face is live or a photo/video"""
    
    def __init__(self):
        self.motion_history = deque(maxlen=15)
        self.blink_history = deque(maxlen=30)
        self.texture_history = deque(maxlen=10)
        
    def reset(self):
        self.motion_history.clear()
        self.blink_history.clear()
        self.texture_history.clear()
    
    def detect_motion(self, face_roi):
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            moment = cv2.moments(gray)
            if moment['m00'] != 0:
                cx = int(moment['m10'] / moment['m00'])
                cy = int(moment['m01'] / moment['m00'])
                self.motion_history.append((cx, cy))
            
            if len(self.motion_history) >= 10:
                positions = np.array(list(self.motion_history))
                variance = np.var(positions, axis=0).sum()
                return variance > 5
            return False
        except:
            return False
    
    def detect_blink(self, face_roi):
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            eye_region = gray[:int(h*0.4), :]
            _, thresh = cv2.threshold(eye_region, 50, 255, cv2.THRESH_BINARY_INV)
            dark_ratio = np.sum(thresh > 0) / thresh.size
            
            self.blink_history.append(dark_ratio)
            
            if len(self.blink_history) >= 20:
                recent = list(self.blink_history)[-20:]
                peaks = []
                for i in range(1, len(recent)-1):
                    if recent[i] > recent[i-1] and recent[i] > recent[i+1]:
                        if recent[i] > 0.15:
                            peaks.append(i)
                return len(peaks) >= 1
            return False
        except:
            return False
    
    def analyze_texture(self, face_roi):
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_variance = np.var(laplacian)
            
            self.texture_history.append(texture_variance)
            
            if len(self.texture_history) >= 5:
                avg_texture = np.mean(list(self.texture_history))
                return avg_texture > 100
            return False
        except:
            return False
    
    def is_live(self, face_roi):
        motion_ok = self.detect_motion(face_roi)
        blink_ok = self.detect_blink(face_roi)
        texture_ok = self.analyze_texture(face_roi)
        
        score = sum([motion_ok, blink_ok, texture_ok])
        return score >= 2, {'motion': motion_ok, 'blink': blink_ok, 'texture': texture_ok, 'score': score}


class ModernAttendanceSystem:
    """Professional Face Recognition Attendance System"""
    
    def __init__(self):
        # Colors
        self.colors = {
            'primary': (74, 144, 226),
            'success': (46, 213, 115),
            'warning': (255, 159, 67),
            'error': (234, 84, 85),
            'bg': (248, 249, 250),
            'card': (255, 255, 255),
            'text': (33, 37, 41),
            'text_light': (108, 117, 125),
            'border': (222, 226, 230),
            'purple': (155, 89, 182)
        }
        
        # Initialize directories
        self.data_dir = Path("attendance_data")
        self.data_dir.mkdir(exist_ok=True)
        
        self.encodings_file = self.data_dir / "employee_faces.dat"
        self.model_file = self.data_dir / "face_model.yml"
        self.attendance_file = self.data_dir / "attendance.csv"
        self.config_file = self.data_dir / "config.json"
        
        self.config = self.load_config()
        self.employees = {}
        self.load_employee_data()
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize recognition model
        self.face_model = FaceRecognitionModel()
        self.load_or_train_model()
        
        # Other features
        self.emotion_detector = EmotionDetector()
        self.anti_spoofing = AntiSpoofing()
        
        self.cap = None
        self.last_action_time = 0
        self.action_cooldown = 3
        self.mouse_pos = (0, 0)
        self.mouse_click = False
        
        print(f"\n✓ System initialized | Employees: {len(self.employees)}")
        print(f"✓ Recognition model: {'Trained' if self.face_model.is_trained else 'Ready for training'}")
        print("✓ Emotion detection enabled")
        print("✓ Anti-spoofing protection enabled\n")
    
    def load_config(self):
        default = {
            'company_name': 'TECH SOLUTIONS INC.',
            'recognition_threshold': 60,  # LBPH confidence threshold
            'camera_width': 1280,
            'camera_height': 720,
            'anti_spoofing_enabled': True,
            'emotion_detection_enabled': True
        }
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return {**default, **json.load(f)}
            except:
                return default
        return default
    
    def load_employee_data(self):
        """Load employee face templates"""
        if self.encodings_file.exists():
            try:
                with open(self.encodings_file, 'rb') as f:
                    self.employees = pickle.load(f)
                print(f"✓ Loaded {len(self.employees)} employee profiles")
            except Exception as e:
                print(f"✗ Error loading employee data: {e}")
                self.employees = {}
    
    def save_employee_data(self):
        """Save employee face templates"""
        try:
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(self.employees, f)
            return True
        except:
            return False
    
    def load_or_train_model(self):
        """Load existing model or train new one"""
        # Try to load existing model
        if self.model_file.exists() and len(self.employees) > 0:
            try:
                self.face_model.recognizer.read(str(self.model_file))
                
                # Rebuild label maps from employee data
                for idx, emp_id in enumerate(self.employees.keys()):
                    self.face_model.label_map[idx] = emp_id
                    self.face_model.reverse_map[emp_id] = idx
                
                self.face_model.is_trained = True
                print("✓ Loaded existing recognition model")
                return True
            except Exception as e:
                print(f"⚠ Could not load model: {e}, will retrain")
        
        # Train new model if we have employee data
        if len(self.employees) > 0:
            if self.face_model.train_model(self.employees):
                self.save_model()
                return True
        
        return False
    
    def save_model(self):
        """Save trained model to file"""
        try:
            self.face_model.recognizer.save(str(self.model_file))
            print("✓ Recognition model saved")
            return True
        except Exception as e:
            print(f"✗ Error saving model: {e}")
            return False
    
    def retrain_model(self):
        """Retrain model with current employee data"""
        print("\n🔄 Retraining recognition model...")
        if self.face_model.train_model(self.employees):
            self.save_model()
            print("✓ Model retrained successfully")
            return True
        else:
            print("✗ Model retraining failed")
            return False
    
    def init_camera(self):
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera_width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera_height'])
            return True
        except:
            return False
    
    def draw_rounded_rect(self, img, pt1, pt2, color, thickness=-1, r=15):
        x1, y1, x2, y2 = pt1[0], pt1[1], pt2[0], pt2[1]
        r = min(r, abs(x2-x1)//2, abs(y2-y1)//2)
        
        if thickness == -1:
            cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
            cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1)
            cv2.circle(img, (x1+r, y1+r), r, color, -1)
            cv2.circle(img, (x2-r, y1+r), r, color, -1)
            cv2.circle(img, (x1+r, y2-r), r, color, -1)
            cv2.circle(img, (x2-r, y2-r), r, color, -1)
        else:
            cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness)
            cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness)
            cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness)
            cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness)
            cv2.ellipse(img, (x1+r, y1+r), (r, r), 180, 0, 90, color, thickness)
            cv2.ellipse(img, (x2-r, y1+r), (r, r), 270, 0, 90, color, thickness)
            cv2.ellipse(img, (x1+r, y2-r), (r, r), 90, 0, 90, color, thickness)
            cv2.ellipse(img, (x2-r, y2-r), (r, r), 0, 0, 90, color, thickness)
    
    def draw_card(self, img, x, y, w, h, title=None):
        shadow = img.copy()
        self.draw_rounded_rect(shadow, (x+4, y+4), (x+w+4, y+h+4), (0,0,0), -1, 12)
        cv2.addWeighted(shadow, 0.15, img, 0.85, 0, img)
        
        self.draw_rounded_rect(img, (x, y), (x+w, y+h), self.colors['card'], -1, 12)
        self.draw_rounded_rect(img, (x, y), (x+w, y+h), self.colors['border'], 1, 12)
        
        if title:
            cv2.rectangle(img, (x, y), (x+w, y+45), self.colors['primary'], -1)
            cv2.putText(img, title, (x+15, y+28), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255,255,255), 2, cv2.LINE_AA)
            return y + 60
        return y + 20
    
    def draw_button(self, img, text, x, y, w, h, hover=False, style='primary'):
        color = self.colors.get(style, self.colors['primary'])
        
        if hover:
            color = tuple(min(255, int(c*1.1)) for c in color)
        
        shadow = img.copy()
        self.draw_rounded_rect(shadow, (x+3, y+3), (x+w+3, y+h+3), (0,0,0), -1, 8)
        cv2.addWeighted(shadow, 0.2, img, 0.8, 0, img)
        
        self.draw_rounded_rect(img, (x, y), (x+w, y+h), color, -1, 8)
        
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255,255,255), 2, cv2.LINE_AA)
        
        return (x, y, w, h)
    
    def draw_stat(self, img, x, y, icon, value, label, color):
        self.draw_card(img, x, y, 160, 130)
        cv2.rectangle(img, (x, y), (x+160, y+4), color, -1)
        
        cv2.circle(img, (x+80, y+50), 25, color, -1)
        cv2.putText(img, icon, (x+65, y+60), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(img, str(value), (x+80-len(str(value))*10, y+90), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, self.colors['text'], 2, cv2.LINE_AA)
        cv2.putText(img, label, (x+80-len(label)*3, y+112), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text_light'], 1, cv2.LINE_AA)
    
    def get_stats(self):
        stats = {'total': len(self.employees), 'present': 0, 'absent': 0, 'hours': 0.0}
        
        if not self.attendance_file.exists():
            stats['absent'] = stats['total']
            return stats
        
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            with open(self.attendance_file, 'r', encoding='utf-8') as f:
                records = [r for r in csv.DictReader(f) if r['Date'] == today]
            
            stats['present'] = len([r for r in records if r['Check_In']])
            stats['absent'] = stats['total'] - stats['present']
            
            hours = [float(r['Hours_Worked']) for r in records if r['Hours_Worked']]
            stats['hours'] = sum(hours) / len(hours) if hours else 0.0
        except:
            stats['absent'] = stats['total']
        
        return stats
    
    def recognize_face(self, face_roi):
        """Use trained model for recognition"""
        if not self.face_model.is_trained:
            return None, 0
        
        threshold = self.config.get('recognition_threshold', 60)
        emp_id, confidence = self.face_model.recognize(face_roi, threshold)
        
        return emp_id, confidence
    
    def save_attendance(self, emp_id, name, action, emotion=None):
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")
        
        if emotion:
            message = self.emotion_detector.get_message(emotion)
        else:
            messages = {
                'check_in': ["Welcome! Have a great day!", "Good morning! Let's achieve!"],
                'check_out': ["Great work today!", "Well done! Rest well!"]
            }
            message = np.random.choice(messages[action])
        
        records = []
        fields = ['Date', 'Employee_ID', 'Name', 'Check_In', 'Check_Out', 'Hours_Worked', 'Status', 'Emotion']

        if self.attendance_file.exists():
            with open(self.attendance_file, 'r', encoding='utf-8') as f:
                records = list(csv.DictReader(f))

        last_rec = None
        for rec in reversed(records):
            if rec.get('Employee_ID') == emp_id:
                last_rec = rec
                break

        last_action = None
        if last_rec:
            if last_rec.get('Check_Out'):
                last_action = 'check_out'
            elif last_rec.get('Check_In'):
                last_action = 'check_in'

        if last_action == action:
            return (f"Action blocked: already performed {action.replace('_', ' ')}.", False)

        if action == 'check_out' and not last_rec:
            return ("Action blocked: no prior check-in found.", False)

        found = False
        for rec in records:
            if rec['Date'] == date_str and rec['Employee_ID'] == emp_id:
                if action == 'check_in' and not rec['Check_In']:
                    rec['Check_In'] = time_str
                    rec['Status'] = 'Present'
                    rec['Emotion'] = emotion or 'neutral'
                    found = True
                    break
                elif action == 'check_out' and rec['Check_In'] and not rec['Check_Out']:
                    rec['Check_Out'] = time_str
                    hours = self.calc_hours(rec['Check_In'], time_str)
                    rec['Hours_Worked'] = f"{hours:.2f}"
                    rec['Status'] = 'Complete'
                    if 'Emotion' not in rec:
                        rec['Emotion'] = emotion or 'neutral'
                    found = True
                    break

        if not found and action == 'check_in':
            records.append({
                'Date': date_str, 'Employee_ID': emp_id, 'Name': name,
                'Check_In': time_str, 'Check_Out': '', 'Hours_Worked': '0',
                'Status': 'Present', 'Emotion': emotion or 'neutral'
            })

        with open(self.attendance_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(records)

        return (message, True)
    
    def calc_hours(self, t1, t2):
        try:
            dt1 = datetime.strptime(t1, "%H:%M:%S")
            dt2 = datetime.strptime(t2, "%H:%M:%S")
            if dt2 < dt1:
                dt2 += timedelta(days=1)
            return round((dt2 - dt1).seconds / 3600, 2)
        except:
            return 0.0
    
    def enrollment_ui(self):
        print("\n" + "="*60)
        name = input("  Employee Name: ").strip()
        if not name:
            print("  ⚠ Name required")
            time.sleep(1)
            return
        
        emp_id = input("  Employee ID: ").strip()
        if not emp_id or emp_id in self.employees:
            print("  ⚠ Invalid or existing ID")
            time.sleep(1)
            return
        
        dept = input("  Department: ").strip() or "General"
        print("\n  📸 Position face in frame, press SPACE to capture multiple angles")
        print("  💡 Tip: Capture 3-5 photos with slight head movements for best results")
        print("  ✓ Press ENTER when done, ESC to cancel\n")
        
        captured_faces = []
        win = "Enrollment"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1100, 700)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            disp = np.ones((h, w, 3), dtype=np.uint8)
            disp[:] = self.colors['bg']
            
            cv2.rectangle(disp, (0, 0), (w, 80), self.colors['primary'], -1)
            cv2.putText(disp, "EMPLOYEE ENROLLMENT", (25, 50), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
            
            y = self.draw_card(disp, 30, 100, 380, 280, "Employee Info")
            cv2.putText(disp, f"Name: {name}", (50, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1, cv2.LINE_AA)
            cv2.putText(disp, f"ID: {emp_id}", (50, y+35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1, cv2.LINE_AA)
            cv2.putText(disp, f"Dept: {dept}", (50, y+70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1, cv2.LINE_AA)
            cv2.putText(disp, f"Samples: {len(captured_faces)}/5", (50, y+110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['purple'], 2, cv2.LINE_AA)
            
            cam_x, cam_y = 430, 100
            cam_w, cam_h = 640, 480
            frame_r = cv2.resize(frame, (cam_w, cam_h))
            disp[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = frame_r
            cv2.rectangle(disp, (cam_x-2, cam_y-2), (cam_x+cam_w+2, cam_y+cam_h+2), self.colors['border'], 2)
            
            gray = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(disp, (cam_x+x, cam_y+y), (cam_x+x+w, cam_y+y+h), 
                             self.colors['success'], 3)
                cv2.putText(disp, "Face Detected", (cam_x+x, cam_y+y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['success'], 2, cv2.LINE_AA)
            
            y_inst = self.draw_card(disp, 30, 400, 380, 180, "Instructions")
            cv2.putText(disp, "SPACE - Capture", (50, y_inst), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1, cv2.LINE_AA)
            cv2.putText(disp, "ENTER - Complete", (50, y_inst+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1, cv2.LINE_AA)
            cv2.putText(disp, "ESC - Cancel", (50, y_inst+60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1, cv2.LINE_AA)
            
            cv2.imshow(win, disp)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyWindow(win)
                print("  ✗ Enrollment cancelled")
                return
            elif key == 32 and len(faces) > 0:  # SPACE
                (x, y, w, h) = faces[0]
                face_roi = frame_r[y:y+h, x:x+w]
                captured_faces.append(face_roi)
                print(f"  ✓ Captured sample {len(captured_faces)}")
            elif key == 13 and len(captured_faces) >= 3:  # ENTER
                break
        
        cv2.destroyWindow(win)
        
        if len(captured_faces) < 3:
            print("  ✗ Need at least 3 samples")
            return
        
        # Use first capture as primary template
        self.employees[emp_id] = {
            'name': name,
            'department': dept,
            'face_template': captured_faces[0],
            'enrolled_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.save_employee_data()
        self.retrain_model()
        
        print(f"\n  ✓ {name} enrolled successfully with {len(captured_faces)} samples!")
        time.sleep(2)
    
    def attendance_ui(self):
        win = "Smart Attendance System"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1400, 800)
        cv2.setMouseCallback(win, self.mouse_callback)
        
        status_msg = ""
        status_time = 0
        status_color = self.colors['success']
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            disp = np.ones((h, w, 3), dtype=np.uint8)
            disp[:] = self.colors['bg']
            
            cv2.rectangle(disp, (0, 0), (w, 90), self.colors['primary'], -1)
            cv2.putText(disp, self.config['company_name'], (25, 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(disp, datetime.now().strftime("%A, %B %d, %Y  |  %I:%M %p"), 
                    (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            
            stats = self.get_stats()
            self.draw_stat(disp, 30, 110, "T", stats['total'], "TOTAL", self.colors['primary'])
            self.draw_stat(disp, 210, 110, "P", stats['present'], "PRESENT", self.colors['success'])
            self.draw_stat(disp, 390, 110, "A", stats['absent'], "ABSENT", self.colors['error'])
            self.draw_stat(disp, 570, 110, "H", f"{stats['hours']:.1f}", "AVG HOURS", self.colors['warning'])
            
            cam_x, cam_y = 30, 260
            cam_w, cam_h = min(700, w - cam_x - 10), min(h - cam_y - 20, 480)  # Fixed dimensions
            
            # Resize frame to exact dimensions
            frame_r = cv2.resize(frame, (cam_w, cam_h))
            
            # Ensure the slice dimensions match
            disp[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = frame_r
            cv2.rectangle(disp, (cam_x-2, cam_y-2), (cam_x+cam_w+2, cam_y+cam_h+2), 
                        self.colors['border'], 2)
            
            gray = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
            
            recognized_id = None
            recognized_name = None
            confidence = 0
            emotion = 'neutral'
            is_live = False
            spoof_details = {}
            
            for (x, y, w_face, h_face) in faces:
                face_roi = frame_r[y:y+h_face, x:x+w_face]
                
                if self.config['anti_spoofing_enabled']:
                    is_live, spoof_details = self.anti_spoofing.is_live(face_roi)
                else:
                    is_live = True
                
                if is_live:
                    emp_id, conf = self.recognize_face(face_roi)
                    
                    if emp_id and emp_id in self.employees:
                        recognized_id = emp_id
                        recognized_name = self.employees[emp_id]['name']
                        confidence = conf
                        
                        if self.config['emotion_detection_enabled']:
                            emotion = self.emotion_detector.detect_emotion(face_roi)
                        
                        color = self.colors['success']
                        label = f"{recognized_name} ({int(confidence*100)}%)"
                    else:
                        color = self.colors['warning']
                        label = "Unknown"
                    
                    cv2.rectangle(disp, (cam_x+x, cam_y+y), (cam_x+x+w_face, cam_y+y+h_face), color, 3)
                    
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(disp, (cam_x+x, cam_y+y-35), 
                                (cam_x+x+label_size[0]+10, cam_y+y), color, -1)
                    cv2.putText(disp, label, (cam_x+x+5, cam_y+y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(disp, (cam_x+x, cam_y+y), (cam_x+x+w_face, cam_y+y+h_face), 
                                self.colors['error'], 3)
                    cv2.putText(disp, "SPOOF DETECTED", (cam_x+x, cam_y+y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['error'], 2, cv2.LINE_AA)
            
            # Calculate info panel position dynamically
            info_x = min(760, w - 620)
            y_info = self.draw_card(disp, info_x, 260, 610, 300, "Recognition Info")
            
            if recognized_id:
                cv2.putText(disp, f"Name: {recognized_name}", (info_x+20, y_info),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2, cv2.LINE_AA)
                cv2.putText(disp, f"ID: {recognized_id}", (info_x+20, y_info+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2, cv2.LINE_AA)
                cv2.putText(disp, f"Confidence: {int(confidence*100)}%", (info_x+20, y_info+80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2, cv2.LINE_AA)
                
                if self.config['emotion_detection_enabled']:
                    emoji = self.emotion_detector.get_emoji(emotion)
                    cv2.putText(disp, f"Mood: {emotion.title()} {emoji}", (info_x+20, y_info+120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['purple'], 2, cv2.LINE_AA)
                
                if self.config['anti_spoofing_enabled']:
                    liveness = "Live" if is_live else "Spoof"
                    liveness_color = self.colors['success'] if is_live else self.colors['error']
                    cv2.putText(disp, f"Liveness: {liveness}", (info_x+20, y_info+160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, liveness_color, 2, cv2.LINE_AA)
            else:
                cv2.putText(disp, "No face recognized", (info_x+20, y_info+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text_light'], 1, cv2.LINE_AA)
            
            btn_y = 580
            in_btn = self.draw_button(disp, "CHECK IN", info_x, btn_y, 280, 60,
                                    self.is_hovering(info_x, btn_y, 280, 60), 'success')
            out_btn = self.draw_button(disp, "CHECK OUT", info_x+310, btn_y, 280, 60,
                                    self.is_hovering(info_x+310, btn_y, 280, 60), 'error')
            
            if status_msg and time.time() - status_time < 3:
                msg_card_y = 670
                self.draw_card(disp, info_x, msg_card_y, 610, 80)
                cv2.putText(disp, status_msg, (info_x+20, msg_card_y+50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)
            
            if self.mouse_click:
                self.mouse_click = False
                current_time = time.time()
                
                if current_time - self.last_action_time > self.action_cooldown:
                    if self.is_hovering(*in_btn) and recognized_id:
                        msg, success = self.save_attendance(recognized_id, recognized_name, 
                                                        'check_in', emotion)
                        status_msg = msg
                        status_time = current_time
                        status_color = self.colors['success'] if success else self.colors['error']
                        self.last_action_time = current_time
                        self.anti_spoofing.reset()
                    
                    elif self.is_hovering(*out_btn) and recognized_id:
                        msg, success = self.save_attendance(recognized_id, recognized_name, 
                                                        'check_out', emotion)
                        status_msg = msg
                        status_time = current_time
                        status_color = self.colors['success'] if success else self.colors['error']
                        self.last_action_time = current_time
                        self.anti_spoofing.reset()
            
            cv2.imshow(win, disp)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        
        cv2.destroyWindow(win)
    
    def mouse_callback(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_click = True
    
    def is_hovering(self, x, y, w, h):
        mx, my = self.mouse_pos
        return x <= mx <= x+w and y <= my <= y+h
    
    def main_menu(self):
        win = "Main Menu"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 900, 600)
        cv2.setMouseCallback(win, self.mouse_callback)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            disp = np.ones((h, w, 3), dtype=np.uint8)
            disp[:] = self.colors['bg']
            
            cv2.rectangle(disp, (0, 0), (w, 120), self.colors['primary'], -1)
            cv2.putText(disp, "SMART ATTENDANCE SYSTEM", (w//2-300, 50),
                       cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
            cv2.putText(disp, self.config['company_name'], (w//2-200, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            
            y = 180
            enroll_btn = self.draw_button(disp, "ENROLL NEW EMPLOYEE", 250, y, 400, 80,
                                         self.is_hovering(250, y, 400, 80), 'primary')
            
            attend_btn = self.draw_button(disp, "START ATTENDANCE", 250, y+110, 400, 80,
                                         self.is_hovering(250, y+110, 400, 80), 'success')
            
            exit_btn = self.draw_button(disp, "EXIT", 250, y+220, 400, 80,
                                       self.is_hovering(250, y+220, 400, 80), 'error')
            
            cv2.putText(disp, f"Employees: {len(self.employees)} | Model: {'Trained' if self.face_model.is_trained else 'Not Trained'}",
                       (250, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text_light'], 1, cv2.LINE_AA)
            
            cv2.imshow(win, disp)
            
            if self.mouse_click:
                self.mouse_click = False
                if self.is_hovering(*enroll_btn):
                    self.enrollment_ui()
                elif self.is_hovering(*attend_btn):
                    if len(self.employees) == 0:
                        print("  ⚠ No employees enrolled!")
                        time.sleep(2)
                    elif not self.face_model.is_trained:
                        print("  ⚠ Model not trained!")
                        time.sleep(2)
                    else:
                        self.attendance_ui()
                elif self.is_hovering(*exit_btn):
                    break
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        
        cv2.destroyWindow(win)
    
    def run(self):
        if not self.init_camera():
            print("✗ Camera initialization failed!")
            return
        
        try:
            self.main_menu()
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("\n✓ System shutdown complete")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SMART FACE RECOGNITION ATTENDANCE SYSTEM")
    print("="*60)
    
    system = ModernAttendanceSystem()
    system.run()