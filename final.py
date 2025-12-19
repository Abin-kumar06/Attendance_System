import cv2
import numpy as np
from datetime import datetime, timedelta
import csv
import pickle
import time
from pathlib import Path
import json
from collections import deque

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
        self.NORMALIZED_STRESS_THRESH = 0.38
        self.HIGH_HAPPY_RATIO_THRESH = 1.18
        self.LOW_SAD_RATIO_THRESH = 0.85
        self.LOW_VAR_THRESH = 950
        
    def detect_emotion(self, face_roi):
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
    """Professional Face Recognition Attendance System with Emotion Detection & Anti-Spoofing"""
    
    def __init__(self):
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
        
        self.data_dir = Path("attendance_data")
        self.data_dir.mkdir(exist_ok=True)
        
        self.encodings_file = self.data_dir / "employee_faces.dat"
        self.attendance_file = self.data_dir / "attendance.csv"
        self.config_file = self.data_dir / "config.json"
        
        self.config = self.load_config()
        self.employees = {}
        self.load_employee_data()
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.emotion_detector = EmotionDetector()
        self.anti_spoofing = AntiSpoofing()
        
        self.cap = None
        self.last_action_time = 0
        self.action_cooldown = 3
        self.mouse_pos = (0, 0)
        self.mouse_click = False
        
        print(f"\n✓ System initialized | Employees: {len(self.employees)}")
        print("✓ Emotion detection enabled")
        print("✓ Anti-spoofing protection enabled\n")
    
    def load_config(self):
        default = {
            'company_name': 'TECH SOLUTIONS INC.',
            'threshold': 0.7,
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
    
    def save_attendance(self, emp_id, name, action, emotion=None):
        """
        Save attendance with emotion detection for both check-in and check-out
        Emotion format in CSV: "check_in_emotion -> check_out_emotion"
        """
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")
        
        # Get emotion-based message
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
                    rec['Emotion'] = emotion or 'neutral'  # Store check-in emotion
                    found = True
                    break
                elif action == 'check_out' and rec['Check_In'] and not rec['Check_Out']:
                    rec['Check_Out'] = time_str
                    hours = self.calc_hours(rec['Check_In'], time_str)
                    rec['Hours_Worked'] = f"{hours:.2f}"
                    rec['Status'] = 'Complete'
                    # Store both check-in and check-out emotions
                    checkin_emotion = rec.get('Emotion', 'neutral')
                    checkout_emotion = emotion or 'neutral'
                    rec['Emotion'] = f"{checkin_emotion} -> {checkout_emotion}"
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
    
    def remove_employee_ui(self):
        """UI for removing an employee registration"""
        print("\n" + "="*60)
        print("  REMOVE EMPLOYEE REGISTRATION")
        print("="*60)
        
        if not self.employees:
            print("  ⚠ No employees registered in the system")
            time.sleep(2)
            return
        
        print("\n  Registered Employees:")
        print("  " + "-"*56)
        for emp_id, data in self.employees.items():
            print(f"  ID: {emp_id:10} | Name: {data['name']:20} | Dept: {data.get('department', 'N/A')}")
        print("  " + "-"*56)
        
        emp_id = input("\n  Enter Employee ID to remove (or press Enter to cancel): ").strip()
        
        if not emp_id:
            print("  ℹ Removal cancelled")
            time.sleep(1)
            return
        
        if emp_id not in self.employees:
            print(f"  ⚠ Employee ID '{emp_id}' not found")
            time.sleep(2)
            return
        
        emp_name = self.employees[emp_id]['name']
        
        print(f"\n  ⚠ WARNING: This will permanently delete registration for:")
        print(f"     Name: {emp_name}")
        print(f"     ID: {emp_id}")
        confirm = input("\n  Type 'DELETE' to confirm removal: ").strip()
        
        if confirm != 'DELETE':
            print("  ℹ Removal cancelled")
            time.sleep(1)
            return
        
        try:
            del self.employees[emp_id]
            if self.save_employee_data():
                print(f"\n  ✓ Employee '{emp_name}' (ID: {emp_id}) removed successfully!")
                print("  ℹ Note: Past attendance records are preserved")
            else:
                print(f"\n  ✗ Failed to save changes")
        except Exception as e:
            print(f"\n  ✗ Error removing employee: {e}")
        
        time.sleep(2)
    
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
        
        self.anti_spoofing.reset()
        start = time.time()
        timeout = 30
        liveness_frames = 0
        required_frames = 15
        
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
                
                is_live, spoof_details = self.anti_spoofing.is_live(face_roi)
                
                if is_live:
                    liveness_frames += 1
                else:
                    liveness_frames = max(0, liveness_frames - 1)
                
                indicator_x, indicator_y = cam_x + 20, cam_y + 20
                self.draw_rounded_rect(disp, (indicator_x, indicator_y), 
                                      (indicator_x + 250, indicator_y + 120), 
                                      self.colors['card'], -1, 10)
                
                cv2.putText(disp, "Liveness Check", (indicator_x + 15, indicator_y + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2, cv2.LINE_AA)
                
                progress = min(100, int((liveness_frames / required_frames) * 100))
                bar_width = int((220 * progress) / 100)
                
                cv2.rectangle(disp, (indicator_x + 15, indicator_y + 45), 
                             (indicator_x + 235, indicator_y + 65), self.colors['border'], -1)
                
                bar_color = self.colors['success'] if progress == 100 else self.colors['warning']
                if progress < 50:
                    bar_color = self.colors['error']
                
                cv2.rectangle(disp, (indicator_x + 15, indicator_y + 45), 
                             (indicator_x + 15 + bar_width, indicator_y + 65), bar_color, -1)
                
                cv2.putText(disp, f"{progress}%", (indicator_x + 100, indicator_y + 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                
                status_y = indicator_y + 80
                check_mark = "✓" if spoof_details['motion'] else "✗"
                check_color = self.colors['success'] if spoof_details['motion'] else self.colors['error']
                cv2.putText(disp, f"{check_mark} Motion", (indicator_x + 15, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, check_color, 1, cv2.LINE_AA)
                
                check_mark = "✓" if spoof_details['blink'] else "✗"
                check_color = self.colors['success'] if spoof_details['blink'] else self.colors['error']
                cv2.putText(disp, f"{check_mark} Blink", (indicator_x + 95, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, check_color, 1, cv2.LINE_AA)
                
                check_mark = "✓" if spoof_details['texture'] else "✗"
                check_color = self.colors['success'] if spoof_details['texture'] else self.colors['error']
                cv2.putText(disp, f"{check_mark} Texture", (indicator_x + 165, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, check_color, 1, cv2.LINE_AA)
                
                if liveness_frames >= required_frames:
                    emp_id, conf = self.recognize_face(face_roi)
                    
                    if emp_id:
                        emp = self.employees[emp_id]
                        
                        # IMPORTANT: Detect emotion fresh for BOTH check-in AND check-out
                        emotion = 'neutral'
                        if self.config.get('emotion_detection_enabled', True):
                            emotion = self.emotion_detector.detect_emotion(face_roi)
                        
                        scale_x, scale_y = cam_w/frame.shape[1], cam_h/frame.shape[0]
                        nx, ny = int(x*scale_x)+cam_x, int(y*scale_y)+cam_y
                        nw, nh = int(fw*scale_x), int(fh*scale_y)
                        
                        cv2.rectangle(disp, (nx, ny), (nx+nw, ny+nh), self.colors['success'], 4)
                        
                        info_y = ny + nh + 20
                        if info_y + 220 > h:
                            info_y = ny - 240
                        
                        self.draw_rounded_rect(disp, (nx, info_y), (nx+nw, info_y+220), 
                                              self.colors['card'], -1, 10)
                        self.draw_rounded_rect(disp, (nx, info_y), (nx+nw, info_y+220), 
                                              self.colors['success'], 2, 10)
                        
                        cv2.putText(disp, emp['name'], (nx+15, info_y+35), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2, cv2.LINE_AA)
                        cv2.putText(disp, f"ID: {emp_id}", (nx+15, info_y+65), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1, cv2.LINE_AA)
                        cv2.putText(disp, f"Match: {conf:.0%}", (nx+15, info_y+90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1, cv2.LINE_AA)
                        
                        emoji = self.emotion_detector.get_emoji(emotion)
                        cv2.putText(disp, f"Mood: {emotion.title()}", (nx+15, info_y+120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['purple'], 1, cv2.LINE_AA)
                        cv2.putText(disp, emoji, (nx+15, info_y+150), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.colors['text'], 2, cv2.LINE_AA)
                        
                        if time.time() - self.last_action_time > self.action_cooldown:
                            msg, success = self.save_attendance(emp_id, emp['name'], action, emotion)
                            self.last_action_time = time.time()

                            if success:
                                cv2.putText(disp, "SUCCESS!", (nx+15, info_y+190), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['success'], 2, cv2.LINE_AA)
                                print(f"\n  ✓ {emp['name']} {action.replace('_', ' ')} successful!")
                                print(f"  Mood: {emotion} {emoji}")
                                print(f"  Message: {msg}")
                                cv2.imshow(win, disp)
                                cv2.waitKey(3000)
                                cv2.destroyWindow(win)
                                return
                            else:
                                cv2.putText(disp, str(msg)[:30], (nx+15, info_y+190),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['error'], 1, cv2.LINE_AA)
                                cv2.imshow(win, disp)
                                cv2.waitKey(2000)
                else:
                    if liveness_frames < 5:
                        warn_x = cam_x + cam_w//2 - 200
                        warn_y = cam_y + cam_h - 100
                        self.draw_rounded_rect(disp, (warn_x, warn_y), 
                                              (warn_x + 400, warn_y + 70), 
                                              self.colors['error'], -1, 10)
                        cv2.putText(disp, "⚠ SPOOFING DETECTED", (warn_x + 50, warn_y + 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
                        cv2.putText(disp, "Please use live face, not photo/video", (warn_x + 30, warn_y + 55), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
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
            'remove': {'text': 'Remove', 'pos': (50, 365), 'size': (280, 65), 'style': 'error'},
            'in': {'text': 'Check In', 'pos': (50, 450), 'size': (280, 65), 'style': 'success'},
            'out': {'text': 'Check Out', 'pos': (50, 535), 'size': (280, 65), 'style': 'warning'},
            'exit': {'text': 'Exit', 'pos': (50, 620), 'size': (280, 65), 'style': 'error'}
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
                        elif btn_id == 'remove':
                            cv2.destroyWindow(win)
                            self.remove_employee_ui()
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
    print("  with Emotion Detection & Anti-Spoofing")
    print("="*60 + "\n")
    
    system = ModernAttendanceSystem()
    system.run()