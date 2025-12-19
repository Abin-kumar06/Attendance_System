import cv2
import numpy as np
from datetime import datetime, timedelta
import csv
import pickle
import time
from pathlib import Path
import json

class ModernAttendanceSystem:
    """Professional Face Recognition Attendance System with Modern UI"""
    
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
        
        # Initialize
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
        
        self.cap = None
        self.last_action_time = 0
        self.action_cooldown = 3
        self.mouse_pos = (0, 0)
        self.mouse_click = False
        
        print(f"\n✓ System initialized | Employees: {len(self.employees)}\n")
    
    def load_config(self):
        default = {
            'company_name': 'TECH SOLUTIONS INC.',
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
        # Shadow
        shadow = img.copy()
        self.draw_rounded_rect(shadow, (x+4, y+4), (x+w+4, y+h+4), (0,0,0), -1, 12)
        cv2.addWeighted(shadow, 0.15, img, 0.85, 0, img)
        
        # Card
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
        
        # Shadow
        shadow = img.copy()
        self.draw_rounded_rect(shadow, (x+3, y+3), (x+w+3, y+h+3), (0,0,0), -1, 8)
        cv2.addWeighted(shadow, 0.2, img, 0.8, 0, img)
        
        # Button
        self.draw_rounded_rect(img, (x, y), (x+w, y+h), color, -1, 8)
        
        # Text
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
    
    def save_attendance(self, emp_id, name, action):
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")
        
        messages = {
            'check_in': ["Welcome! Have a great day!", "Good morning! Let's achieve!"],
            'check_out': ["Great work today!", "Well done! Rest well!"]
        }
        message = np.random.choice(messages[action])
        
        records = []
        fields = ['Date', 'Employee_ID', 'Name', 'Check_In', 'Check_Out', 'Hours_Worked', 'Status']

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
                    found = True
                    break
                elif action == 'check_out' and rec['Check_In'] and not rec['Check_Out']:
                    rec['Check_Out'] = time_str
                    hours = self.calc_hours(rec['Check_In'], time_str)
                    rec['Hours_Worked'] = f"{hours:.2f}"
                    rec['Status'] = 'Complete'
                    found = True
                    break

        if not found and action == 'check_in':
            records.append({
                'Date': date_str, 'Employee_ID': emp_id, 'Name': name,
                'Check_In': time_str, 'Check_Out': '', 'Hours_Worked': '0',
                'Status': 'Present'
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
            
            # Header
            cv2.rectangle(disp, (0, 0), (w, 80), self.colors['primary'], -1)
            cv2.putText(disp, "EMPLOYEE ENROLLMENT", (25, 50), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
            
            # Info card
            y = self.draw_card(disp, 30, 100, 380, 230, "Employee Info")
            cv2.putText(disp, f"Name: {name}", (50, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1, cv2.LINE_AA)
            cv2.putText(disp, f"ID: {emp_id}", (50, y+35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1, cv2.LINE_AA)
            cv2.putText(disp, f"Dept: {dept}", (50, y+70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1, cv2.LINE_AA)
            
            # Camera
            cam_x, cam_y = 430, 100
            cam_w, cam_h = 640, 480
            frame_r = cv2.resize(frame, (cam_w, cam_h))
            disp[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = frame_r
            cv2.rectangle(disp, (cam_x-2, cam_y-2), (cam_x+cam_w+2, cam_y+cam_h+2), 
                         self.colors['primary'], 3)
            
            # Face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100,100))
            
            for (x, y, fw, fh) in faces:
                scale_x, scale_y = cam_w/frame.shape[1], cam_h/frame.shape[0]
                nx, ny = int(x*scale_x)+cam_x, int(y*scale_y)+cam_y
                nw, nh = int(fw*scale_x), int(fh*scale_y)
                
                color = self.colors['success'] if len(faces)==1 else self.colors['warning']
                cv2.rectangle(disp, (nx, ny), (nx+nw, ny+nh), color, 3)
            
            # Preview
            if captured is not None:
                prev = cv2.resize(captured, (150, 150))
                disp[450:600, 120:270] = prev
                cv2.rectangle(disp, (117, 447), (273, 603), self.colors['success'], 3)
                cv2.putText(disp, "Captured!", (125, 620), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['success'], 2, cv2.LINE_AA)
            
            cv2.imshow(win, disp)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32 and len(faces) == 1:  # SPACE
                x, y, fw, fh = faces[0]
                captured = frame[y:y+fh, x:x+fw].copy()
            elif key == 13 and captured is not None:  # ENTER
                self.employees[emp_id] = {
                    'name': name, 'department': dept,
                    'face_template': captured,
                    'enrolled': datetime.now().strftime("%Y-%m-%d")
                }
                if self.save_employee_data():
                    print(f"\n  ✓ {name} enrolled successfully!\n")
                cv2.destroyWindow(win)
                return
            elif key == 27:  # ESC
                cv2.destroyWindow(win)
                return
        
        cv2.destroyWindow(win)
    
    def check_ui(self, action):
        win = action.replace('_', ' ').title()
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1000, 700)
        
        start = time.time()
        timeout = 30
        
        while time.time() - start < timeout:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            disp = np.ones((h, w, 3), dtype=np.uint8)
            disp[:] = self.colors['bg']
            
            # Header
            color = self.colors['primary'] if action == 'check_in' else self.colors['success']
            cv2.rectangle(disp, (0, 0), (w, 80), color, -1)
            title = "CHECK IN" if action == 'check_in' else "CHECK OUT"
            cv2.putText(disp, title, (30, 50), cv2.FONT_HERSHEY_DUPLEX, 
                       1.5, (255,255,255), 2, cv2.LINE_AA)
            
            # Time
            cv2.putText(disp, datetime.now().strftime("%H:%M:%S"), (w-200, 50), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
            
            # Camera
            cam_x, cam_y = 50, 100
            cam_w, cam_h = w-100, h-150
            frame_r = cv2.resize(frame, (cam_w, cam_h))
            disp[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = frame_r
            cv2.rectangle(disp, (cam_x-3, cam_y-3), (cam_x+cam_w+3, cam_y+cam_h+3), 
                         color, 4)
            
            # Detect & recognize
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100,100))
            
            if len(faces) == 1:
                x, y, fw, fh = faces[0]
                face_roi = frame[y:y+fh, x:x+fw]
                emp_id, conf = self.recognize_face(face_roi)
                
                if emp_id:
                    emp = self.employees[emp_id]
                    scale_x, scale_y = cam_w/frame.shape[1], cam_h/frame.shape[0]
                    nx, ny = int(x*scale_x)+cam_x, int(y*scale_y)+cam_y
                    nw, nh = int(fw*scale_x), int(fh*scale_y)
                    
                    cv2.rectangle(disp, (nx, ny), (nx+nw, ny+nh), self.colors['success'], 4)
                    
                    # Info box
                    info_y = ny + nh + 20
                    if info_y + 150 > h:
                        info_y = ny - 170
                    
                    self.draw_rounded_rect(disp, (nx, info_y), (nx+nw, info_y+150), 
                                          self.colors['card'], -1, 10)
                    self.draw_rounded_rect(disp, (nx, info_y), (nx+nw, info_y+150), 
                                          self.colors['success'], 2, 10)
                    
                    cv2.putText(disp, emp['name'], (nx+15, info_y+35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2, cv2.LINE_AA)
                    cv2.putText(disp, f"ID: {emp_id}", (nx+15, info_y+65), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1, cv2.LINE_AA)
                    cv2.putText(disp, f"Match: {conf:.0%}", (nx+15, info_y+90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1, cv2.LINE_AA)
                    
                    if time.time() - self.last_action_time > self.action_cooldown:
                        msg, success = self.save_attendance(emp_id, emp['name'], action)
                        self.last_action_time = time.time()

                        if success:
                            cv2.putText(disp, "SUCCESS!", (nx+15, info_y+120), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['success'], 2, cv2.LINE_AA)
                            print(f"\n  ✓ {emp['name']} {action.replace('_', ' ')} successful!")
                            cv2.imshow(win, disp)
                            cv2.waitKey(2000)
                            cv2.destroyWindow(win)
                            return
                        else:
                            cv2.putText(disp, str(msg), (nx+15, info_y+120),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['error'], 2, cv2.LINE_AA)
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
            
            # Header
            cv2.rectangle(disp, (0, 0), (w, 100), self.colors['primary'], -1)
            cv2.putText(disp, self.config['company_name'], (30, 45), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(disp, "Smart Attendance System", (30, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            
            # Date/Time
            cv2.putText(disp, datetime.now().strftime("%B %d, %Y"), (w-280, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(disp, datetime.now().strftime("%H:%M:%S"), (w-280, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            
            # Sidebar
            cv2.rectangle(disp, (0, 100), (380, h), (245, 245, 245), -1)
            
            # Stats
            self.draw_stat(disp, 50, 130, "👥", stats['total'], "Total", self.colors['primary'])
            self.draw_stat(disp, 220, 130, "✓", stats['present'], "Present", self.colors['success'])
            
            # Buttons
            for btn_id, btn in buttons.items():
                hover = (btn['pos'][0] <= self.mouse_pos[0] <= btn['pos'][0]+btn['size'][0] and
                        btn['pos'][1] <= self.mouse_pos[1] <= btn['pos'][1]+btn['size'][1])
                areas[btn_id] = self.draw_button(disp, btn['text'], btn['pos'][0], btn['pos'][1],
                                                 btn['size'][0], btn['size'][1], hover, btn['style'])
            
            # Camera
            cam_x, cam_y = 400, 120
            cam_w, cam_h = w-420, h-140
            frame_r = cv2.resize(frame, (cam_w, cam_h))
            disp[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = frame_r
            cv2.rectangle(disp, (cam_x-2, cam_y-2), (cam_x+cam_w+2, cam_y+cam_h+2), 
                         self.colors['primary'], 3)
            
            # Handle clicks
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
    print("="*60 + "\n")
    
    system = ModernAttendanceSystem()
    system.run()