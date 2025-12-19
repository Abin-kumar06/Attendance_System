import cv2
import numpy as np
from datetime import datetime, timedelta
import csv
import pickle
from pathlib import Path
import json
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow import keras

class SmartAttendanceSystem:
    """Optimized Attendance with MTCNN, ML Anti-Spoofing & Mood Detection"""
    
    def __init__(self):
        # Colors
        self.colors = {
            'primary': (74, 144, 226),
            'success': (46, 213, 115),
            'warning': (255, 159, 67),
            'error': (234, 84, 85),
            'bg': (248, 249, 250),
            'card': (255, 255, 255),
            'text': (33, 37, 41)
        }
        
        self.mood_colors = {
            'Happy': (46, 213, 115), 'Sad': (108, 117, 125),
            'Angry': (234, 84, 85), 'Neutral': (74, 144, 226)
        }
        
        # Initialize MTCNN
        self.detector = MTCNN()
        
        # Load anti-spoofing model
        self.antispoofing_model = None
        self.load_antispoofing_model()
        
        # Data setup
        self.data_dir = Path("attendance_data")
        self.data_dir.mkdir(exist_ok=True)
        self.encodings_file = self.data_dir / "faces.dat"
        self.attendance_file = self.data_dir / "attendance.csv"
        
        self.employees = {}
        self.load_data()
        
        self.cap = None
        self.last_action = 0
        self.cooldown = 3
        
        print(f"\n✓ MTCNN initialized | Employees: {len(self.employees)}")
        print(f"✓ ML Anti-Spoofing: {'Enabled' if self.antispoofing_model else 'Disabled (using fallback)'}")
        print("✓ Features: Mood Detection + Anti-Spoofing\n")
    
    def load_antispoofing_model(self):
        """Load trained anti-spoofing model"""
        model_path = Path("models/antispoofing_model.h5")
        if model_path.exists():
            try:
                self.antispoofing_model = keras.models.load_model(model_path)
                self.img_size = 64  # Model input size
                print("✓ Anti-spoofing model loaded")
            except Exception as e:
                print(f"⚠ Could not load anti-spoofing model: {e}")
                self.antispoofing_model = None
        else:
            print("⚠ Anti-spoofing model not found. Using fallback method.")
            print("  Run antispoofing trainer to create the model.")
    
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
        """Fallback anti-spoofing using traditional methods"""
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Texture analysis
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_var = laplacian.var()
            
            # Color distribution
            hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
            color_std = np.std(hsv[:,:,1])
            
            # Simple scoring
            is_real = texture_var > 100 and color_std > 15
            score = min(1.0, (texture_var / 200 + color_std / 30) / 2)
            
            return is_real, score
        except:
            return True, 0.5
    
    def load_data(self):
        """Load employee data"""
        if self.encodings_file.exists():
            try:
                with open(self.encodings_file, 'rb') as f:
                    self.employees = pickle.load(f)
            except:
                pass
    
    def save_data(self):
        """Save employee data"""
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(self.employees, f)
    
    def init_camera(self):
        """Initialize webcam"""
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return True
        return False
    
    def detect_faces(self, frame):
        """Detect faces using MTCNN"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(rgb)
        return results
    
    def get_face_encoding(self, face_img):
        """Simple face encoding"""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))
        gray = cv2.equalizeHist(gray)
        return gray.flatten()
    
    def compare_faces(self, encoding1, encoding2):
        """Compare face encodings"""
        correlation = np.corrcoef(encoding1, encoding2)[0, 1]
        return max(0, correlation)
    
    def recognize_face(self, face_img):
        """Recognize face"""
        if not self.employees:
            return None, 0
        
        encoding = self.get_face_encoding(face_img)
        best_match, best_score = None, 0
        
        for emp_id, data in self.employees.items():
            score = self.compare_faces(encoding, data['encoding'])
            if score > best_score and score > 0.7:
                best_score = score
                best_match = emp_id
        
        return best_match, best_score
    
    def detect_mood(self, face_img, landmarks):
        """Detect mood from landmarks"""
        try:
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            mouth_left = landmarks['mouth_left']
            mouth_right = landmarks['mouth_right']
            
            eye_dist = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
            mouth_width = np.linalg.norm(np.array(mouth_left) - np.array(mouth_right))
            smile_ratio = mouth_width / eye_dist if eye_dist > 0 else 0
            
            if smile_ratio > 0.65:
                return 'Happy', 0.8
            elif smile_ratio < 0.45:
                return 'Sad', 0.7
            else:
                return 'Neutral', 0.75
        except:
            return 'Neutral', 0.5
    
    def get_mood_message(self, mood, action):
        """Get mood-based message"""
        messages = {
            'check_in': {
                'Happy': "Your smile brightens our day! ☀️",
                'Sad': "Hope your day gets better! 💙",
                'Angry': "Take a deep breath, fresh start! 🌸",
                'Neutral': "Ready for a productive day! 💼"
            },
            'check_out': {
                'Happy': "Great work today! Keep smiling! 😊",
                'Sad': "Tomorrow is a new day! Rest up 💤",
                'Angry': "Relax and decompress tonight! 🛀",
                'Neutral': "Good job today! See you tomorrow! 👋"
            }
        }
        return messages.get(action, {}).get(mood, "Have a great day!")
    
    def save_attendance(self, emp_id, name, action, mood):
        """Save attendance record"""
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        fields = ['Date', 'Employee_ID', 'Name', 'Check_In', 'Check_Out', 'Hours', 'Mood']
        records = []
        
        if self.attendance_file.exists():
            with open(self.attendance_file, 'r') as f:
                records = list(csv.DictReader(f))
        
        today_record = None
        for rec in records:
            if rec['Date'] == date and rec['Employee_ID'] == emp_id:
                today_record = rec
                break
        
        message = self.get_mood_message(mood, action)
        
        if action == 'check_in':
            if today_record and today_record.get('Check_In'):
                return "Already checked in!", False
            
            if today_record:
                today_record['Check_In'] = time_str
                today_record['Mood'] = mood
            else:
                records.append({
                    'Date': date, 'Employee_ID': emp_id, 'Name': name,
                    'Check_In': time_str, 'Check_Out': '', 'Hours': '0', 'Mood': mood
                })
        
        elif action == 'check_out':
            if not today_record or not today_record.get('Check_In'):
                return "No check-in found!", False
            if today_record.get('Check_Out'):
                return "Already checked out!", False
            
            today_record['Check_Out'] = time_str
            in_time = datetime.strptime(today_record['Check_In'], "%H:%M:%S")
            out_time = datetime.strptime(time_str, "%H:%M:%S")
            hours = (out_time - in_time).seconds / 3600
            today_record['Hours'] = f"{hours:.2f}"
        
        with open(self.attendance_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(records)
        
        return message, True
    
    def draw_ui(self, frame, faces, mode='dashboard'):
        """Draw UI"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        color = self.colors['primary'] if mode == 'dashboard' else self.colors['success']
        cv2.rectangle(overlay, (0, 0), (w, 70), color, -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        title = "SMART ATTENDANCE" if mode == 'dashboard' else mode.upper()
        cv2.putText(frame, title, (20, 45), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), 2)
        
        time_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, time_str, (w-200, 45), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 2)
        
        return frame
    
    def enrollment(self):
        """Enroll employee"""
        print("\n" + "="*50)
        name = input("Employee Name: ").strip()
        if not name:
            print("Name required!")
            return
        
        emp_id = input("Employee ID: ").strip()
        if not emp_id or emp_id in self.employees:
            print("Invalid or existing ID!")
            return
        
        print("\nPosition face, press SPACE to capture, ESC to cancel")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            faces = self.detect_faces(frame)
            
            for face in faces:
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)
                color = self.colors['success'] if len(faces) == 1 else self.colors['warning']
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                if len(faces) == 1:
                    cv2.putText(frame, "Press SPACE", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            frame = self.draw_ui(frame, faces, 'ENROLLMENT')
            cv2.imshow("Enrollment", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32 and len(faces) == 1:
                face = faces[0]
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)
                captured = frame[y:y+h, x:x+w]
                
                encoding = self.get_face_encoding(captured)
                self.employees[emp_id] = {
                    'name': name,
                    'encoding': encoding,
                    'enrolled': datetime.now().strftime("%Y-%m-%d")
                }
                self.save_data()
                print(f"\n✓ {name} enrolled!")
                break
            elif key == 27:
                break
        
        cv2.destroyAllWindows()
    
    def check_attendance(self, action):
        """Check in/out"""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < 30:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            faces = self.detect_faces(frame)
            
            if len(faces) == 1:
                face = faces[0]
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)
                
                face_img = frame[y:y+h, x:x+w]
                
                # ML Anti-spoofing
                is_live, liveness = self.check_liveness_ml(face_img)
                
                if not is_live:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), self.colors['error'], 3)
                    cv2.putText(frame, "SPOOFING DETECTED!", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['error'], 2)
                else:
                    emp_id, conf = self.recognize_face(face_img)
                    
                    if emp_id and conf > 0.7:
                        emp = self.employees[emp_id]
                        mood, mood_conf = self.detect_mood(face_img, face['keypoints'])
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), self.colors['success'], 3)
                        
                        info_y = y + h + 10
                        cv2.rectangle(frame, (x, info_y), (x+w, info_y+140), (255,255,255), -1)
                        cv2.rectangle(frame, (x, info_y), (x+w, info_y+140), self.colors['success'], 2)
                        
                        cv2.putText(frame, emp['name'], (x+10, info_y+25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
                        cv2.putText(frame, f"ID: {emp_id}", (x+10, info_y+50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
                        cv2.putText(frame, f"Match: {conf:.0%}", (x+10, info_y+75), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
                        cv2.putText(frame, f"Liveness: {liveness:.0%}", (x+10, info_y+100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['success'], 1)
                        
                        mood_color = self.mood_colors.get(mood, self.colors['primary'])
                        cv2.putText(frame, f"Mood: {mood}", (x+10, info_y+125), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, mood_color, 2)
                        
                        if (datetime.now().timestamp() - self.last_action) > self.cooldown:
                            msg, success = self.save_attendance(emp_id, emp['name'], action, mood)
                            self.last_action = datetime.now().timestamp()
                            
                            if success:
                                print(f"\n✓ {emp['name']} {action.replace('_', ' ')} - {mood}")
                                print(f"  {msg}\n")
                                cv2.imshow(action.replace('_', ' ').title(), frame)
                                cv2.waitKey(2000)
                                cv2.destroyAllWindows()
                                return
            
            frame = self.draw_ui(frame, faces, action.replace('_', ' '))
            cv2.imshow(action.replace('_', ' ').title(), frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cv2.destroyAllWindows()
    
    def run(self):
        """Main menu"""
        if not self.init_camera():
            print("❌ Camera failed!")
            return
        
        while True:
            print("\n" + "="*50)
            print("SMART ATTENDANCE SYSTEM")
            print("="*50)
            print("1. Enroll Employee")
            print("2. Check In")
            print("3. Check Out")
            print("4. Exit")
            print("="*50)
            
            choice = input("\nSelect: ").strip()
            
            if choice == '1':
                self.enrollment()
            elif choice == '2':
                self.check_attendance('check_in')
            elif choice == '3':
                self.check_attendance('check_out')
            elif choice == '4':
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✓ System closed\n")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("SMART ATTENDANCE SYSTEM")
    print("✓ MTCNN Face Detection")
    print("✓ ML Anti-Spoofing")
    print("✓ Mood Detection")
    print("="*50)
    
    system = SmartAttendanceSystem()
    system.run()