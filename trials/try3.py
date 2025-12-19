import cv2
import numpy as np
from datetime import datetime, timedelta
import csv
import pickle
from pathlib import Path
import json
from mtcnn import MTCNN

class SmartAttendanceSystem:
    """Optimized Face Recognition Attendance with MTCNN, Mood Detection & Anti-Spoofing"""
    
    def __init__(self):
        # Simple color palette
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
        
        # Initialize MTCNN detector
        self.detector = MTCNN()
        
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
        
        # Anti-spoofing buffer
        self.frame_buffer = []
        self.buffer_size = 5
        
        print(f"\n✓ MTCNN initialized | Employees: {len(self.employees)}")
        print("✓ Features: Mood Detection + Anti-Spoofing\n")
    
    def load_data(self):
        """Load employee face data"""
        if self.encodings_file.exists():
            try:
                with open(self.encodings_file, 'rb') as f:
                    self.employees = pickle.load(f)
            except:
                pass
    
    def save_data(self):
        """Save employee face data"""
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
        """Simple face encoding using histogram"""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))
        gray = cv2.equalizeHist(gray)
        return gray.flatten()
    
    def compare_faces(self, encoding1, encoding2):
        """Compare two face encodings"""
        correlation = np.corrcoef(encoding1, encoding2)[0, 1]
        return max(0, correlation)
    
    def recognize_face(self, face_img):
        """Recognize face against database"""
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
        """Simplified mood detection using facial landmarks"""
        try:
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            mouth_left = landmarks['mouth_left']
            mouth_right = landmarks['mouth_right']
            
            # Calculate mouth width vs eye distance
            eye_dist = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
            mouth_width = np.linalg.norm(np.array(mouth_left) - np.array(mouth_right))
            smile_ratio = mouth_width / eye_dist if eye_dist > 0 else 0
            
            # Simple mood classification
            if smile_ratio > 0.65:
                return 'Happy', 0.8
            elif smile_ratio < 0.45:
                return 'Sad', 0.7
            else:
                return 'Neutral', 0.75
        except:
            return 'Neutral', 0.5
    
    def check_liveness(self, frame, bbox):
        """Anti-spoofing check using motion and texture"""
        x, y, w, h = bbox
        face = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Add to buffer
        self.frame_buffer.append(gray)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        
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
        
        # 2. Texture analysis (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_var = laplacian.var()
        texture_pass = texture_var > 100
        
        # 3. Color check
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        color_std = np.std(hsv[:,:,1])
        color_pass = color_std > 15
        
        # Combine scores
        score = (motion_pass * 0.5 + texture_pass * 0.3 + color_pass * 0.2)
        is_live = score >= 0.5
        
        return is_live, score
    
    def get_mood_message(self, mood, action):
        """Get personalized message based on mood"""
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
        
        # Load existing records
        if self.attendance_file.exists():
            with open(self.attendance_file, 'r') as f:
                records = list(csv.DictReader(f))
        
        # Find today's record for this employee
        today_record = None
        for rec in records:
            if rec['Date'] == date and rec['Employee_ID'] == emp_id:
                today_record = rec
                break
        
        message = self.get_mood_message(mood, action)
        
        # Check-in logic
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
        
        # Check-out logic
        elif action == 'check_out':
            if not today_record or not today_record.get('Check_In'):
                return "No check-in found!", False
            if today_record.get('Check_Out'):
                return "Already checked out!", False
            
            today_record['Check_Out'] = time_str
            # Calculate hours
            in_time = datetime.strptime(today_record['Check_In'], "%H:%M:%S")
            out_time = datetime.strptime(time_str, "%H:%M:%S")
            hours = (out_time - in_time).seconds / 3600
            today_record['Hours'] = f"{hours:.2f}"
        
        # Save records
        with open(self.attendance_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(records)
        
        return message, True
    
    def draw_ui(self, frame, faces, mode='dashboard'):
        """Draw UI elements on frame"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Header
        color = self.colors['primary'] if mode == 'dashboard' else self.colors['success']
        cv2.rectangle(overlay, (0, 0), (w, 70), color, -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        title = "SMART ATTENDANCE" if mode == 'dashboard' else mode.upper()
        cv2.putText(frame, title, (20, 45), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), 2)
        
        # Time
        time_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, time_str, (w-200, 45), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 2)
        
        return frame
    
    def enrollment(self):
        """Enroll new employee"""
        print("\n" + "="*50)
        name = input("Employee Name: ").strip()
        if not name:
            print("Name required!")
            return
        
        emp_id = input("Employee ID: ").strip()
        if not emp_id or emp_id in self.employees:
            print("Invalid or existing ID!")
            return
        
        print("\nPosition face in frame, press SPACE to capture, ESC to cancel")
        
        captured = None
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            faces = self.detect_faces(frame)
            
            # Draw faces
            for face in faces:
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)
                color = self.colors['success'] if len(faces) == 1 else self.colors['warning']
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                if len(faces) == 1:
                    cv2.putText(frame, "Press SPACE to capture", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            frame = self.draw_ui(frame, faces, 'ENROLLMENT')
            cv2.imshow("Enrollment", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32 and len(faces) == 1:  # SPACE
                face = faces[0]
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)
                captured = frame[y:y+h, x:x+w]
                
                # Save employee
                encoding = self.get_face_encoding(captured)
                self.employees[emp_id] = {
                    'name': name,
                    'encoding': encoding,
                    'enrolled': datetime.now().strftime("%Y-%m-%d")
                }
                self.save_data()
                print(f"\n✓ {name} enrolled successfully!")
                break
            elif key == 27:  # ESC
                break
        
        cv2.destroyAllWindows()
    
    def check_attendance(self, action):
        """Check in/out with face recognition"""
        self.frame_buffer = []  # Reset buffer
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
                
                # Anti-spoofing check
                is_live, liveness = self.check_liveness(frame, (x, y, w, h))
                
                if not is_live:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), self.colors['error'], 3)
                    cv2.putText(frame, "SPOOFING DETECTED!", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['error'], 2)
                else:
                    # Recognize face
                    emp_id, conf = self.recognize_face(face_img)
                    
                    if emp_id and conf > 0.7:
                        emp = self.employees[emp_id]
                        mood, mood_conf = self.detect_mood(face_img, face['keypoints'])
                        
                        # Draw info
                        cv2.rectangle(frame, (x, y), (x+w, y+h), self.colors['success'], 3)
                        
                        # Info box
                        info_y = y + h + 10
                        cv2.rectangle(frame, (x, info_y), (x+w, info_y+120), (255,255,255), -1)
                        cv2.rectangle(frame, (x, info_y), (x+w, info_y+120), self.colors['success'], 2)
                        
                        cv2.putText(frame, emp['name'], (x+10, info_y+25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
                        cv2.putText(frame, f"ID: {emp_id}", (x+10, info_y+50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
                        cv2.putText(frame, f"Match: {conf:.0%}", (x+10, info_y+75), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
                        
                        mood_color = self.mood_colors.get(mood, self.colors['primary'])
                        cv2.putText(frame, f"Mood: {mood}", (x+10, info_y+100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, mood_color, 2)
                        
                        # Save attendance
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
            print("❌ Camera initialization failed!")
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
            
            choice = input("\nSelect option: ").strip()
            
            if choice == '1':
                self.enrollment()
            elif choice == '2':
                self.check_attendance('check_in')
            elif choice == '3':
                self.check_attendance('check_out')
            elif choice == '4':
                break
            else:
                print("Invalid option!")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✓ System closed\n")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("SMART ATTENDANCE SYSTEM")
    print("✓ MTCNN Face Detection")
    print("✓ Mood Detection")
    print("✓ Anti-Spoofing Protection")
    print("="*50)
    
    system = SmartAttendanceSystem()
    system.run()