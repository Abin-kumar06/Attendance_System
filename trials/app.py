from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from deepface import DeepFace
import cv2
import numpy as np
import base64
from datetime import datetime
import csv
import json
from pathlib import Path
import os

app = Flask(__name__)
CORS(app)

# Configuration
DATA_DIR = Path("attendance_data")
DATA_DIR.mkdir(exist_ok=True)
FACES_DIR = DATA_DIR / "faces"
FACES_DIR.mkdir(exist_ok=True)
ATTENDANCE_FILE = DATA_DIR / "attendance.csv"
EMPLOYEES_FILE = DATA_DIR / "employees.json"

# Emotion to mood mapping
EMOTION_MOOD_MAP = {
    'happy': 'Happy',
    'sad': 'Sad',
    'angry': 'Angry',
    'fear': 'Sad',
    'surprise': 'Surprised',
    'disgust': 'Angry',
    'neutral': 'Neutral'
}

# Mood messages
MOOD_MESSAGES = {
    'check_in': {
        'Happy': "Your smile brightens our day! ☀️",
        'Sad': "Hope your day gets better! 💙",
        'Angry': "Take a deep breath, fresh start! 🌸",
        'Neutral': "Ready for a productive day! 💼",
        'Surprised': "Exciting day ahead! 🚀"
    },
    'check_out': {
        'Happy': "Great work today! Keep smiling! 😊",
        'Sad': "Tomorrow is a new day! Rest up 💤",
        'Angry': "Relax and decompress tonight! 🛀",
        'Neutral': "Good job today! See you tomorrow! 👋",
        'Surprised': "What a day! Rest up! 🌟"
    }
}

def load_employees():
    """Load employee database"""
    if EMPLOYEES_FILE.exists():
        with open(EMPLOYEES_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_employees(employees):
    """Save employee database"""
    with open(EMPLOYEES_FILE, 'w') as f:
        json.dump(employees, f, indent=2)

def decode_image(image_data):
    """Decode base64 image"""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def get_mood_message(mood, action):
    """Get personalized message"""
    return MOOD_MESSAGES.get(action, {}).get(mood, "Have a great day!")

def check_liveness(img):
    """Simple anti-spoofing check"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Texture analysis
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_var = laplacian.var()
        
        # Color distribution
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color_std = np.std(hsv[:,:,1])
        
        # Simple scoring
        is_live = texture_var > 100 and color_std > 15
        score = min(1.0, (texture_var / 200 + color_std / 30) / 2)
        
        return is_live, score
    except:
        return True, 0.5  # Default to allowing if check fails

def save_attendance(emp_id, name, action, mood):
    """Save attendance record"""
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    fields = ['Date', 'Employee_ID', 'Name', 'Check_In', 'Check_Out', 'Hours', 'Mood']
    records = []
    
    # Load existing
    if ATTENDANCE_FILE.exists():
        with open(ATTENDANCE_FILE, 'r') as f:
            records = list(csv.DictReader(f))
    
    # Find today's record
    today_record = None
    for rec in records:
        if rec['Date'] == date and rec['Employee_ID'] == emp_id:
            today_record = rec
            break
    
    message = get_mood_message(mood, action)
    
    if action == 'check_in':
        if today_record and today_record.get('Check_In'):
            return False, "Already checked in today!"
        
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
            return False, "No check-in found!"
        if today_record.get('Check_Out'):
            return False, "Already checked out!"
        
        today_record['Check_Out'] = time_str
        in_time = datetime.strptime(today_record['Check_In'], "%H:%M:%S")
        out_time = datetime.strptime(time_str, "%H:%M:%S")
        hours = (out_time - in_time).seconds / 3600
        today_record['Hours'] = f"{hours:.2f}"
    
    # Save
    with open(ATTENDANCE_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(records)
    
    return True, message

@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')

@app.route('/api/employees', methods=['GET'])
def get_employees():
    """Get all employees"""
    employees = load_employees()
    return jsonify({
        'success': True,
        'employees': employees,
        'count': len(employees)
    })

@app.route('/api/enroll', methods=['POST'])
def enroll():
    """Enroll new employee"""
    try:
        data = request.json
        emp_id = data.get('employee_id')
        name = data.get('name')
        department = data.get('department', 'General')
        image_data = data.get('image')
        
        if not emp_id or not name or not image_data:
            return jsonify({'success': False, 'message': 'Missing required fields'})
        
        # Check if employee exists
        employees = load_employees()
        if emp_id in employees:
            return jsonify({'success': False, 'message': 'Employee ID already exists'})
        
        # Decode and save image
        img = decode_image(image_data)
        if img is None:
            return jsonify({'success': False, 'message': 'Invalid image data'})
        
        # Save face image
        face_path = FACES_DIR / f"{emp_id}.jpg"
        cv2.imwrite(str(face_path), img)
        
        # Add to database
        employees[emp_id] = {
            'name': name,
            'department': department,
            'face_path': str(face_path),
            'enrolled': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_employees(employees)
        
        return jsonify({
            'success': True,
            'message': f'{name} enrolled successfully!',
            'employee_id': emp_id
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/api/recognize', methods=['POST'])
def recognize():
    """Recognize face and detect emotion"""
    try:
        data = request.json
        image_data = data.get('image')
        action = data.get('action', 'check_in')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image provided'})
        
        # Decode image
        img = decode_image(image_data)
        if img is None:
            return jsonify({'success': False, 'message': 'Invalid image'})
        
        # Save temp image
        temp_path = DATA_DIR / "temp_face.jpg"
        cv2.imwrite(str(temp_path), img)
        
        # Anti-spoofing check
        is_live, liveness_score = check_liveness(img)
        if not is_live:
            return jsonify({
                'success': False,
                'message': 'Spoofing detected! Please use a real face.',
                'liveness_score': float(liveness_score)
            })
        
        # Load employees
        employees = load_employees()
        if not employees:
            return jsonify({'success': False, 'message': 'No employees enrolled'})
        
        # Try to recognize face
        best_match = None
        best_distance = float('inf')
        
        for emp_id, emp_data in employees.items():
            try:
                result = DeepFace.verify(
                    img1_path=str(temp_path),
                    img2_path=emp_data['face_path'],
                    model_name='ArcFace',
                    detector_backend='retinaface',
                    enforce_detection=False
                )
                
                distance = result['distance']
                if distance < best_distance and result['verified']:
                    best_distance = distance
                    best_match = emp_id
            except:
                continue
        
        if not best_match:
            return jsonify({
                'success': False,
                'message': 'Face not recognized. Please enroll first.'
            })
        
        # Detect emotion
        try:
            emotion_result = DeepFace.analyze(
                img_path=str(temp_path),
                actions=['emotion'],
                detector_backend='retinaface',
                enforce_detection=False
            )
            
            if isinstance(emotion_result, list):
                emotion_result = emotion_result[0]
            
            dominant_emotion = emotion_result['dominant_emotion']
            mood = EMOTION_MOOD_MAP.get(dominant_emotion, 'Neutral')
        except:
            mood = 'Neutral'
        
        # Save attendance
        emp = employees[best_match]
        success, message = save_attendance(best_match, emp['name'], action, mood)
        
        if not success:
            return jsonify({'success': False, 'message': message})
        
        return jsonify({
            'success': True,
            'employee_id': best_match,
            'name': emp['name'],
            'department': emp['department'],
            'mood': mood,
            'message': message,
            'confidence': float(1 - best_distance),
            'liveness_score': float(liveness_score),
            'action': action
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get today's attendance statistics"""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        employees = load_employees()
        total = len(employees)
        present = 0
        
        if ATTENDANCE_FILE.exists():
            with open(ATTENDANCE_FILE, 'r') as f:
                records = [r for r in csv.DictReader(f) if r['Date'] == today]
                present = len([r for r in records if r.get('Check_In')])
        
        return jsonify({
            'success': True,
            'total': total,
            'present': present,
            'absent': total - present
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 SMART ATTENDANCE SYSTEM - FLASK BACKEND")
    print("="*60)
    print("✓ DeepFace with ArcFace recognition")
    print("✓ RetinaFace detector")
    print("✓ Emotion detection")
    print("✓ Anti-spoofing protection")
    print("="*60)
    print(f"\n📁 Data directory: {DATA_DIR.absolute()}")
    print(f"👥 Employees enrolled: {len(load_employees())}")
    print(f"\n🌐 Server starting at http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)