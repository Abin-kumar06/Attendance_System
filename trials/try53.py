import cv2
import numpy as np
from datetime import datetime, timedelta
import csv
import pickle
import time
from pathlib import Path
import json
import tensorflow as tf
from tensorflow import keras

class ModernAttendanceSystem:
    """Professional Face Recognition with Mood Detection & ML Anti-Spoofing"""
    
    def __init__(self):
        # Colors - Modern flat design palette
        self.colors = {
            'primary': (74, 144, 226),      # Blue
            'success': (46, 213, 115),      # Green
            'warning': (255, 159, 67),      # Orange
            'error': (234, 84, 85),         # Red
            'bg': (248, 249, 250),          # Light gray
            'card': (255, 255, 255),        # White
            'text': (33, 37, 41),           # Dark gray
            'text_light': (108, 117, 125),  # Medium gray
            'border': (222, 226, 230)       # Light border
        }
        
        # Mood colors
        self.mood_colors = {
            'Happy': (46, 213, 115),
            'Sad': (108, 117, 125),
            'Angry': (234, 84, 85),
            'Neutral': (74, 144, 226),
            'Surprised': (255, 159, 67)
        }
        
        # Initialize
        self.data_dir = Path("attendance_data")
        self.data_dir.mkdir(exist_ok=True)
        
        self.encodings_file = self.data_dir / "employee_faces.dat"
        self.attendance_file = self.data_dir / "attendance.csv"
        self.config_file = self.data_dir / "config.json"
        
        self.config = self.load_config()
        self.employees = {}
        self.load_employee_data()
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Load anti-spoofing model
        self.antispoofing_model = None
        self.load_antispoofing_model()
        
        self.cap = None
        self.last_action_time = 0
        self.action_cooldown = 3
        self.mouse_pos = (0, 0)
        self.mouse_click = False
        
        # Anti-spoofing buffer
        self.frame_buffer = []
        self.buffer_size = 5
        
        print(f"\n✓ System initialized | Employees: {len(self.employees)}")
        print(f"✓ ML Anti-Spoofing: {'Enabled' if self.antispoofing_model else 'Using Fallback'}")
        print("✓ Mood Detection: Enabled\n")
    
    def load_antispoofing_model(self):
        """Load trained anti-spoofing model"""
        model_path = Path("models/antispoofing_model.h5")
        if model_path.exists():
            try:
                self.antispoofing_model = keras.models.load_model(model_path)
                self.img_size = 64
                print("✓ Anti-spoofing model loaded")
            except Exception as e:
                print(f"⚠ Could not load model: {e}")
                self.antispoofing_model = None
        else:
            print("⚠ Anti-spoofing model not found (using fallback)")
    
    def load_config(self):
        default = {
            'company_name': 'Dsignz Media',
            'threshold': 0.7,
            'camera_width': 1280,
            'camera_height': 720
        }
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return {**default, **json.load(f)}
            except:
                return default
        return default
    
    def load_employee_data(self):
        if self.encodings_file.exists():
            try:
                with open(self.encodings_file, 'rb') as f:
                    self.employees = pickle.load(f)
            except:
                self.employees = {}
    
    def save_employee_data(self):
        try:
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(self.employees, f)
            return True
        except:
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
    
    def detect_mood(self, face_roi):
        """Detect mood/emotion from facial features"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
            
            # Analyze facial regions
            h, w = gray.shape
            upper_face = gray[0:int(h*0.5), :]
            lower_face = gray[int(h*0.5):, :]
            
            # Calculate intensity metrics
            upper_intensity = np.mean(upper_face)
            lower_intensity = np.mean(lower_face)
            intensity_diff = abs(upper_intensity - lower_intensity)
            
            # Edge detection for expression
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Variance (expression movement)
            variance = np.var(gray)
            
            # Mood classification
            if len(eyes) >= 2 and edge_density > 0.15:
                if intensity_diff < 15 and variance > 500:
                    mood = 'Happy'
                    confidence = 0.75
                elif intensity_diff > 25:
                    mood = 'Surprised'
                    confidence = 0.70
                else:
                    mood = 'Neutral'
                    confidence = 0.65
            elif len(eyes) < 2 or edge_density < 0.10:
                mood = 'Sad'
                confidence = 0.60
            elif edge_density > 0.20 and variance > 600:
                mood = 'Angry'
                confidence = 0.65
            else:
                mood = 'Neutral'
                confidence = 0.70
            
            return mood, confidence
        except:
            return 'Neutral', 0.5
    
    def check_liveness_ml(self, face_img):
        """ML-based anti-spoofing detection"""
        if self.antispoofing_model is None:
            return self.check_liveness_fallback(face_img)
        
        try:
            # Preprocess
            img = cv2.resize(face_img, (self.img_size, self.img_size))
            img = img.astype('float32') / 255.0
            img_batch = np.expand_dims(img, axis=0)
            
            # Predict
            pred = self.antispoofing_model.predict(img_batch, verbose=0)[0][0]
            
            is_real = pred > 0.5
            confidence = pred if is_real else (1 - pred)
            
            return is_real, confidence
        except:
            return self.check_liveness_fallback(face_img)
    
    def check_liveness_fallback(self, face_img):
        """Fallback anti-spoofing using traditional CV methods"""
        try:
            # Resize to fixed size for buffer consistency
            face_resized = cv2.resize(face_img, (64, 64))
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Add to buffer
            self.frame_buffer.append(gray.copy())
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)
            
            # Need enough frames for motion detection
            if len(self.frame_buffer) < self.buffer_size:
                return False, 0.0
            
            # 1. Motion detection
            motion_scores = []
            for i in range(len(self.frame_buffer) - 1):
                diff = cv2.absdiff(self.frame_buffer[i], self.frame_buffer[i+1])
                motion = np.mean(diff)
                motion_scores.append(motion)
            avg_motion = np.mean(motion_scores)
            motion_pass = avg_motion > 2.0
            
            # 2. Texture analysis
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_var = laplacian.var()
            texture_pass = texture_var > 100
            
            # 3. Color check
            hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)
            color_std = np.std(hsv[:,:,1])
            color_pass = color_std > 15
            
            # 4. Eye detection
            eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(10, 10))
            eye_pass = len(eyes) >= 1
            
            # Combine scores
            score = (motion_pass * 0.35 + texture_pass * 0.25 + 
                    color_pass * 0.20 + eye_pass * 0.20)
            is_live = score >= 0.5
            
            return is_live, score
        except:
            return True, 0.5
    
    def get_mood_message(self, mood, action):
        """Get personalized message based on mood"""
        messages = {
            'check_in': {
                'Happy': "Your smile brightens our day! ☀️",
                'Sad': "Hope your day gets better! We're here for you 💙",
                'Angry': "Take a deep breath, fresh start today! 🌸",
                'Neutral': "Ready for a productive day! 💼",
                'Surprised': "Exciting day ahead! Let's go! 🚀"
            },
            'check_out': {
                'Happy': "Great work today! Keep smiling! 😊",
                'Sad': "Tomorrow is a new day! Rest up 💤",
                'Angry': "Decompress and relax tonight! 🛀",
                'Neutral': "Another day done! Well deserved rest 👏",
                'Surprised': "What a day! Rest up! 🌟"
            }
        }
        return messages.get(action, {}).get(mood, "Have a great day!")
    
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
        if not self.employees:
            return None, 0
        
        best_match, best_conf = None, 0
        
        try:
            face = cv2.resize(face_roi, (150, 150))
            gray = cv2.equalizeHist(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))
            
            for emp_id, data in self.employees.items():
                stored = cv2.resize(data['face_template'], (150, 150))
                stored_gray = cv2.equalizeHist(cv2.cvtColor(stored, cv2.COLOR_BGR2GRAY))
                
                result = cv2.matchTemplate(gray, stored_gray, cv2.TM_CCOEFF_NORMED)
                conf = cv2.minMaxLoc(result)[1]
                
                mse = np.mean((gray.astype(float) - stored_gray.astype(float)) ** 2)
                sim = max(0.0, 1 - (mse / 20000))
                combined = conf * 0.8 + sim * 0.2
                if combined > best_conf and combined >= self.config.get('threshold', 0.7):
                    best_conf, best_match = combined, emp_id
        except:
            pass
        
        return best_match, best_conf
    
    def save_attendance(self, emp_id, name, action, mood=None):
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")
        
        message = self.get_mood_message(mood, action) if mood else "Success!"
        
        records = []
        fields = ['Date', 'Employee_ID', 'Name', 'Check_In', 'Check_Out', 'Hours_Worked', 'Status', 'Mood']

        if self.attendance_file.exists():
            with open(self.attendance_file, 'r', encoding='utf-8') as f:
                records = list(csv.DictReader(f))
                for rec in records:
                    if 'Mood' not in rec:
                        rec['Mood'] = ''

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
            return (f"Already {action.replace('_', ' ')}ed!", False)

        if action == 'check_out' and not last_rec:
            return ("No prior check-in found!", False)

        found = False
        for rec in records:
            if rec['Date'] == date_str and rec['Employee_ID'] == emp_id:
                if action == 'check_in' and not rec['Check_In']:
                    rec['Check_In'] = time_str
                    rec['Status'] = 'Present'
                    rec['Mood'] = mood or ''
                    found = True
                    break
                elif action == 'check_out' and rec['Check_In'] and not rec['Check_Out']:
                    rec['Check_Out'] = time_str
                    hours = self.calc_hours(rec['Check_In'], time_str)
                    rec['Hours_Worked'] = f"{hours:.2f}"
                    rec['Status'] = 'Complete'
                    if not rec.get('Mood'):
                        rec['Mood'] = mood or ''
                    found = True
                    break

        if not found and action == 'check_in':
            records.append({
                'Date': date_str, 'Employee_ID': emp_id, 'Name': name,
                'Check_In': time_str, 'Check_Out': '', 'Hours_Worked': '0',
                'Status': 'Present', 'Mood': mood or ''
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
        print("\n  Position face in frame, SPACE to capture, ENTER to save, ESC to cancel\n")
        
        captured = None
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
            
            y = self.draw_card(disp, 30, 100, 380, 230, "Employee Info")
            cv2.putText(disp, f"Name: {name}", (50, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1, cv2.LINE_AA)
            cv2.putText(disp, f"ID: {emp_id}", (50, y+35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1, cv2.LINE_AA)
            cv2.putText(disp, f"Dept: {dept}", (50, y+70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1, cv2.LINE_AA)
            
            cam_x, cam_y = 430, 100
            cam_w, cam_h = 640, 480
            frame_r = cv2.resize(frame, (cam_w, cam_h))
            disp[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = frame_r
            cv2.rectangle(disp, (cam_x-2, cam_y-2), (cam_x+cam_w+2, cam_y+cam_h+2), 
                         self.colors['primary'], 3)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100,100))
            
            for (x, y, fw, fh) in faces:
                scale_x, scale_y = cam_w/frame.shape[1], cam_h/frame.shape[0]
                nx, ny = int(x*scale_x)+cam_x, int(y*scale_y)+cam_y
                nw, nh = int(fw*scale_x), int(fh*scale_y)
                
                color = self.colors['success'] if len(faces)==1 else self.colors['warning']
                cv2.rectangle(disp, (nx, ny), (nx+nw, ny+nh), color, 3)
            
            if captured is not None:
                prev = cv2.resize(captured, (150, 150))
                disp[450:600, 120:270] = prev
                cv2.rectangle(disp, (117, 447), (273, 603), self.colors['success'], 3)
                cv2.putText(disp, "Captured!", (125, 620), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['success'], 2, cv2.LINE_AA)
            
            cv2.imshow(win, disp)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32 and len(faces) == 1:
                x, y, fw, fh = faces[0]
                captured = frame[y:y+fh, x:x+fw].copy()
            elif key == 13 and captured is not None:
                self.employees[emp_id] = {
                    'name': name, 'department': dept,
                    'face_template': captured,
                    'enrolled': datetime.now().strftime("%Y-%m-%d")
                }
                if self.save_employee_data():
                    print(f"\n  ✓ {name} enrolled successfully!\n")
                cv2.destroyWindow(win)
                return
            elif key == 27:
                cv2.destroyWindow(win)
                return
        
        cv2.destroyWindow(win)
    
    def check_ui(self, action):
        win = action.replace('_', ' ').title()
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1000, 700)
        
        start = time.time()
        timeout = 30
        self.frame_buffer = []  # Reset buffer
        
        while time.time() - start < timeout:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            disp = np.ones((h, w, 3), dtype=np.uint8)
            disp[:] = self.colors['bg']
            
            color = self.colors['primary'] if action == 'check_in' else self.colors['success']
            cv2.rectangle(disp, (0, 0), (w, 80), color, -1)
            title = "CHECK IN" if action == 'check_in' else "CHECK OUT"
            cv2.putText(disp, title, (30, 50), cv2.FONT_HERSHEY_DUPLEX, 
                       1.5, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.putText(disp, datetime.now().strftime("%H:%M:%S"), (w-200, 50), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
            
            cam_x, cam_y = 50, 100
            cam_w, cam_h = w-100, h-150
            frame_r = cv2.resize(frame, (cam_w, cam_h))
            disp[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = frame_r
            cv2.rectangle(disp, (cam_x-3, cam_y-3), (cam_x+cam_w+3, cam_y+cam_h+3), 
                         color, 4)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100,100))
            
            if len(faces) == 1:
                x, y, fw, fh = faces[0]
                face_roi = frame[y:y+fh, x:x+fw]
                
                # Anti-spoofing check
                is_live, liveness = self.check_liveness_ml(face_roi)
                
                if not is_live:
                    scale_x, scale_y = cam_w/frame.shape[1], cam_h/frame.shape[0]
                    nx, ny = int(x*scale_x)+cam_x, int(y*scale_y)+cam_y
                    nw, nh = int(fw*scale_x), int(fh*scale_y)
                    
                    cv2.rectangle(disp, (nx, ny), (nx+nw, ny+nh), self.colors['error'], 4)
                    
                    info_y = ny + nh + 20
                    if info_y + 100 > h:
                        info_y = ny - 120
                    
                    self.draw_rounded_rect(disp, (nx, info_y), (nx+nw, info_y+100), 
                                          self.colors['error'], -1, 10)
                    
                    cv2.putText(disp, "SPOOFING DETECTED!", (nx+15, info_y+35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                    cv2.putText(disp, "Please use real face", (nx+15, info_y+65), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                else:
                    # Recognize face
                    emp_id, conf = self.recognize_face(face_roi)
                    
                    if emp_id:
                        emp = self.employees[emp_id]
                        
                        # Detect mood
                        mood, mood_conf = self.detect_mood(face_roi)
                        
                        scale_x, scale_y = cam_w/frame.shape[1], cam_h/frame.shape[0]
                        nx, ny = int(x*scale_x)+cam_x, int(y*scale_y)+cam_y
                        nw, nh = int(fw*scale_x), int(fh*scale_y)
                        
                        cv2.rectangle(disp, (nx, ny), (nx+nw, ny+nh), self.colors['success'], 4)
                        
                        info_y = ny + nh + 20
                        if info_y + 200 > h:
                            info_y = ny - 220
                        
                        self.draw_rounded_rect(disp, (nx, info_y), (nx+nw, info_y+200), 
                                              self.colors['card'], -1, 10)
                        self.draw_rounded_rect(disp, (nx, info_y), (nx+nw, info_y+200), 
                                              self.colors['success'], 2, 10)
                        
                        cv2.putText(disp, emp['name'], (nx+15, info_y+30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2, cv2.LINE_AA)
                        cv2.putText(disp, f"ID: {emp_id}", (nx+15, info_y+60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1, cv2.LINE_AA)
                        cv2.putText(disp, f"Match: {conf:.0%}", (nx+15, info_y+85), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1, cv2.LINE_AA)
                        cv2.putText(disp, f"Liveness: {liveness:.0%}", (nx+15, info_y+110), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['success'], 1, cv2.LINE_AA)
                        
                        # Mood indicator
                        mood_color = self.mood_colors.get(mood, self.colors['primary'])
                        cv2.rectangle(disp, (nx+10, info_y+125), (nx+nw-10, info_y+130), mood_color, -1)
                        cv2.putText(disp, f"Mood: {mood}", (nx+15, info_y+155), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mood_color, 2, cv2.LINE_AA)
                        
                        if time.time() - self.last_action_time > self.action_cooldown:
                            msg, success = self.save_attendance(emp_id, emp['name'], action, mood)
                            self.last_action_time = time.time()

                            if success:
                                cv2.putText(disp, "SUCCESS!", (nx+15, info_y+180), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['success'], 2, cv2.LINE_AA)
                                print(f"\n  ✓ {emp['name']} {action.replace('_', ' ')} successful!")
                                print(f"  Mood: {mood} | {msg}")
                                cv2.imshow(win, disp)
                                cv2.waitKey(2500)
                                cv2.destroyWindow(win)
                                return
                            else:
                                cv2.putText(disp, str(msg)[:30], (nx+15, info_y+180),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['error'], 2, cv2.LINE_AA)
                                cv2.imshow(win, disp)
                                cv2.waitKey(2000)
            
            cv2.imshow(win, disp)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow(win)
                return
        
        cv2.destroyWindow(win)
    
    def dashboard(self):
        if not self.init_camera():
            print("Camera error")
            return
        
        win = "Attendance Dashboard"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 800)
        
        def mouse(event, x, y, flags, param):
            self.mouse_pos = (x, y)
            if event == cv2.EVENT_LBUTTONDOWN:
                self.mouse_click = True
        
        cv2.setMouseCallback(win, mouse)
        
        buttons = {
            'enroll': {'text': 'Enroll', 'pos': (50, 280), 'size': (280, 65), 'style': 'primary'},
            'in': {'text': 'Check In', 'pos': (50, 365), 'size': (280, 65), 'style': 'success'},
            'out': {'text': 'Check Out', 'pos': (50, 450), 'size': (280, 65), 'style': 'warning'},
            'exit': {'text': 'Exit', 'pos': (50, 535), 'size': (280, 65), 'style': 'error'}
        }
        
        areas = {}
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            stats = self.get_stats()
            
            disp = np.ones((h, w, 3), dtype=np.uint8)
            disp[:] = self.colors['bg']
            
            cv2.rectangle(disp, (0, 0), (w, 100), self.colors['primary'], -1)
            cv2.putText(disp, self.config['company_name'], (30, 45), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(disp, "Smart Attendance System", (30, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            
            cv2.putText(disp, datetime.now().strftime("%B %d, %Y"), (w-280, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(disp, datetime.now().strftime("%H:%M:%S"), (w-280, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            
            cv2.rectangle(disp, (0, 100), (380, h), (245, 245, 245), -1)
            
            self.draw_stat(disp, 50, 130, "👥", stats['total'], "Total", self.colors['primary'])
            self.draw_stat(disp, 220, 130, "✓", stats['present'], "Present", self.colors['success'])
            
            for btn_id, btn in buttons.items():
                hover = (btn['pos'][0] <= self.mouse_pos[0] <= btn['pos'][0]+btn['size'][0] and
                        btn['pos'][1] <= self.mouse_pos[1] <= btn['pos'][1]+btn['size'][1])
                areas[btn_id] = self.draw_button(disp, btn['text'], btn['pos'][0], btn['pos'][1],
                                                 btn['size'][0], btn['size'][1], hover, btn['style'])
            
            cam_x, cam_y = 400, 120
            cam_w, cam_h = w-420, h-140
            frame_r = cv2.resize(frame, (cam_w, cam_h))
            disp[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = frame_r
            cv2.rectangle(disp, (cam_x-2, cam_y-2), (cam_x+cam_w+2, cam_y+cam_h+2), 
                         self.colors['primary'], 3)
            
            if self.mouse_click:
                for btn_id, area in areas.items():
                    x, y, w, h = area
                    if x <= self.mouse_pos[0] <= x+w and y <= self.mouse_pos[1] <= y+h:
                        if btn_id == 'enroll':
                            cv2.destroyWindow(win)
                            self.enrollment_ui()
                            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(win, 1280, 800)
                            cv2.setMouseCallback(win, mouse)
                        elif btn_id == 'in':
                            cv2.destroyWindow(win)
                            self.check_ui('check_in')
                            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(win, 1280, 800)
                            cv2.setMouseCallback(win, mouse)
                        elif btn_id == 'out':
                            cv2.destroyWindow(win)
                            self.check_ui('check_out')
                            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(win, 1280, 800)
                            cv2.setMouseCallback(win, mouse)
                        elif btn_id == 'exit':
                            self.cleanup()
                            return
                
                self.mouse_click = False
            
            cv2.imshow(win, disp)
            
            if cv2.waitKey(1) & 0xFF == 27:
                self.cleanup()
                return
    
    def cleanup(self):
        print("\n✓ System shutdown complete\n")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def run(self):
        try:
            self.dashboard()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            self.cleanup()
        except Exception as e:
            print(f"\nError: {e}")
            self.cleanup()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SMART ATTENDANCE SYSTEM")
    print("  ✓ Mood Detection Enabled")
    print("  ✓ ML Anti-Spoofing Protection")
    print("="*60 + "\n")
    
    system = ModernAttendanceSystem()
    system.run()