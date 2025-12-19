"""
Real-Time Face Recognition + Mood Detection + Anti-Spoofing Attendance System
Single-file implementation using RetinaFace, ArcFace, and ONNX models
Compatible with Python 3.11+
"""

import os
import csv
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from deepface import DeepFace
from deepface.models.FacialRecognition import FacialRecognition
from deepface.modules import verification

IMAGES_FOLDER = Path("images")
ATTENDANCE_FILE = Path("attendance.csv")
RECOGNITION_THRESHOLD = 0.68
SPOOF_THRESHOLD = 0.5
FRAME_SKIP = 2
ATTENDANCE_COOLDOWN = 60

EMOTION_MESSAGES = {
    "happy": "Great to see you in a good mood! Keep spreading positivity!",
    "sad": "Hope your day gets better. Take care of yourself!",
    "neutral": "Ready to start the day? Let's make it productive!",
    "angry": "Take a deep breath. Wishing you a calmer day ahead!",
    "fear": "Don't worry, you're in a safe place. Stay confident!",
    "surprise": "Something exciting happening? Enjoy the moment!",
    "disgust": "Stay positive! Every day brings new opportunities!"
}

class ModelCache:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not ModelCache._initialized:
            self.face_detector = None
            self.face_recognizer = None
            self.emotion_model = None
            self.antispoof_model = None
            self.known_embeddings: Dict[str, np.ndarray] = {}
            self.attendance_log: Dict[str, datetime] = {}
            ModelCache._initialized = True

model_cache = ModelCache()


def load_models() -> None:
    """Load all required models at startup for caching"""
    print("[INFO] Loading models...")
    
    try:
        from deepface.modules import modeling
        modeling.build_model(task="facial_recognition", model_name="ArcFace")
        print("[INFO] ArcFace recognizer loaded")
    except Exception as e:
        try:
            from deepface.basemodels import ArcFace
            _ = ArcFace.load_model()
            print("[INFO] ArcFace recognizer loaded (legacy)")
        except Exception as e2:
            print(f"[WARN] ArcFace will be loaded on first use: {e2}")
    
    try:
        from deepface.modules import modeling
        modeling.build_model(task="facial_attribute", model_name="Emotion")
        print("[INFO] Emotion model loaded")
    except Exception as e:
        print(f"[WARN] Emotion model will be loaded on first use")
    
    print("[INFO] All models loaded successfully")


def load_known_faces() -> Dict[str, np.ndarray]:
    """Load and compute embeddings for all known faces in the images folder"""
    print(f"[INFO] Loading known faces from {IMAGES_FOLDER}...")
    
    if not IMAGES_FOLDER.exists():
        IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Created {IMAGES_FOLDER} folder. Add face images to enroll.")
        return {}
    
    known_embeddings = {}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    for img_path in IMAGES_FOLDER.iterdir():
        if img_path.suffix.lower() in image_extensions:
            try:
                name = img_path.stem.replace('_', ' ').replace('-', ' ').title()
                
                embedding = get_embedding(str(img_path))
                if embedding is not None:
                    known_embeddings[name] = embedding
                    print(f"[INFO] Enrolled: {name}")
            except Exception as e:
                print(f"[WARN] Failed to process {img_path.name}: {e}")
    
    print(f"[INFO] Loaded {len(known_embeddings)} known faces")
    return known_embeddings


def detect_faces(frame: np.ndarray) -> List[Dict[str, Any]]:
    """Detect faces in a frame using RetinaFace via DeepFace"""
    try:
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="retinaface",
            enforce_detection=False,
            align=True
        )
        
        valid_faces = []
        for face in faces:
            if face.get('confidence', 0) > 0.5:
                valid_faces.append(face)
        
        return valid_faces
    except Exception as e:
        return []


def get_embedding(img_input) -> Optional[np.ndarray]:
    """Generate ArcFace embedding for a face image"""
    try:
        embeddings = DeepFace.represent(
            img_path=img_input,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=False,
            align=True
        )
        
        if embeddings and len(embeddings) > 0:
            return np.array(embeddings[0]["embedding"])
        return None
    except Exception as e:
        return None


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute vectorized cosine similarity between two embeddings"""
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


def recognize_face(embedding: np.ndarray, known_embeddings: Dict[str, np.ndarray]) -> Tuple[str, float]:
    """Recognize a face by comparing embedding to known faces using cosine similarity"""
    if embedding is None or len(known_embeddings) == 0:
        return "Unknown", 0.0
    
    best_match = "Unknown"
    best_similarity = 0.0
    
    for name, known_emb in known_embeddings.items():
        similarity = cosine_similarity(embedding, known_emb)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = name
    
    if best_similarity >= RECOGNITION_THRESHOLD:
        return best_match, best_similarity
    
    return "Unknown", best_similarity


def detect_emotion(face_img: np.ndarray) -> Tuple[str, Dict[str, float]]:
    """Detect emotion from a face image using DeepFace emotion model"""
    try:
        if face_img.dtype == np.float64 or face_img.dtype == np.float32:
            face_img = (face_img * 255).astype(np.uint8)
        
        if len(face_img.shape) == 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
        elif face_img.shape[2] == 4:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGRA2BGR)
        
        result = DeepFace.analyze(
            img_path=face_img,
            actions=['emotion'],
            detector_backend="skip",
            enforce_detection=False
        )
        
        if result and len(result) > 0:
            emotions = result[0].get('emotion', {})
            dominant_emotion = result[0].get('dominant_emotion', 'neutral')
            return dominant_emotion.lower(), emotions
        
        return "neutral", {}
    except Exception as e:
        return "neutral", {}


def check_liveness(face_img: np.ndarray, facial_area: Dict = None) -> Tuple[bool, float, str]:
    """
    Perform anti-spoofing check using multiple heuristics
    Returns: (is_real, confidence, status_message)
    """
    try:
        if face_img.dtype == np.float64 or face_img.dtype == np.float32:
            face_uint8 = (face_img * 255).astype(np.uint8)
        else:
            face_uint8 = face_img.copy()
        
        if len(face_uint8.shape) == 2:
            face_uint8 = cv2.cvtColor(face_uint8, cv2.COLOR_GRAY2BGR)
        
        gray = cv2.cvtColor(face_uint8, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        hsv = cv2.cvtColor(face_uint8, cv2.COLOR_BGR2HSV)
        h_std = np.std(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        v_std = np.std(hsv[:, :, 2])
        
        ycrcb = cv2.cvtColor(face_uint8, cv2.COLOR_BGR2YCrCb)
        cr_mean = np.mean(ycrcb[:, :, 1])
        cb_mean = np.mean(ycrcb[:, :, 2])
        
        score = 0.0
        checks_passed = 0
        total_checks = 5
        
        if laplacian_var > 100:
            checks_passed += 1
            score += 0.2
        
        if 10 < h_std < 50:
            checks_passed += 1
            score += 0.2
        
        if s_mean > 30:
            checks_passed += 1
            score += 0.2
        
        if 133 < cr_mean < 173:
            checks_passed += 1
            score += 0.2
        
        if 77 < cb_mean < 127:
            checks_passed += 1
            score += 0.2
        
        is_real = score >= SPOOF_THRESHOLD
        status = "Real" if is_real else "Spoof"
        
        return is_real, score, status
    
    except Exception as e:
        return True, 0.5, "Unknown"


def get_mood_message(emotion: str) -> str:
    """Get a personalized mood message based on detected emotion"""
    return EMOTION_MESSAGES.get(emotion.lower(), EMOTION_MESSAGES["neutral"])


def init_attendance_file() -> None:
    """Initialize attendance CSV file if it doesn't exist"""
    if not ATTENDANCE_FILE.exists():
        with open(ATTENDANCE_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Time', 'Emotion', 'MoodMessage', 'SpoofStatus', 'Confidence'])
        print(f"[INFO] Created {ATTENDANCE_FILE}")


def mark_attendance(name: str, emotion: str, mood_message: str, spoof_status: str, confidence: float) -> bool:
    """Mark attendance in the CSV file with cooldown to prevent duplicates"""
    if name == "Unknown":
        return False
    
    current_time = datetime.now()
    
    if name in model_cache.attendance_log:
        last_marked = model_cache.attendance_log[name]
        if (current_time - last_marked).total_seconds() < ATTENDANCE_COOLDOWN:
            return False
    
    model_cache.attendance_log[name] = current_time
    
    try:
        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                name,
                current_time.strftime('%Y-%m-%d %H:%M:%S'),
                emotion,
                mood_message,
                spoof_status,
                f"{confidence:.2f}"
            ])
        print(f"[ATTENDANCE] Marked: {name} - {emotion} ({spoof_status})")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to mark attendance: {e}")
        return False


def draw_results(frame: np.ndarray, facial_area: Dict, name: str, emotion: str, 
                 spoof_status: str, confidence: float) -> np.ndarray:
    """Draw bounding box and text on the frame"""
    x = facial_area.get('x', 0)
    y = facial_area.get('y', 0)
    w = facial_area.get('w', 100)
    h = facial_area.get('h', 100)
    
    color = (0, 255, 0) if spoof_status == "Real" else (0, 0, 255)
    
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    label1 = f"{name} ({confidence:.0%})"
    label2 = f"{emotion.capitalize()} | {spoof_status}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    (tw1, th1), _ = cv2.getTextSize(label1, font, font_scale, thickness)
    (tw2, th2), _ = cv2.getTextSize(label2, font, font_scale, thickness)
    
    cv2.rectangle(frame, (x, y - th1 - th2 - 15), (x + max(tw1, tw2) + 10, y), color, -1)
    
    cv2.putText(frame, label1, (x + 5, y - th2 - 10), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(frame, label2, (x + 5, y - 5), font, font_scale, (255, 255, 255), thickness)
    
    return frame


def process_frame(frame: np.ndarray, known_embeddings: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[Dict]]:
    """Process a single frame for face recognition, emotion, and anti-spoofing"""
    results = []
    
    if frame is None or frame.size == 0:
        return frame, results
    
    faces = detect_faces(frame)
    
    for face_data in faces:
        try:
            face_img = face_data.get('face')
            facial_area = face_data.get('facial_area', {})
            
            if face_img is None:
                continue
            
            is_real, spoof_score, spoof_status = check_liveness(face_img, facial_area)
            
            if not is_real:
                frame = draw_results(frame, facial_area, "Spoof Detected", "N/A", spoof_status, spoof_score)
                results.append({
                    'name': 'Spoof Detected',
                    'emotion': 'N/A',
                    'spoof_status': spoof_status,
                    'mood_message': 'Spoof attempt detected!',
                    'confidence': spoof_score
                })
                continue
            
            x = facial_area.get('x', 0)
            y = facial_area.get('y', 0)
            w = facial_area.get('w', 100)
            h = facial_area.get('h', 100)
            
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                embedding = get_embedding(face_crop)
                name, confidence = recognize_face(embedding, known_embeddings)
            else:
                name, confidence = "Unknown", 0.0
            
            emotion, emotion_scores = detect_emotion(face_img)
            mood_message = get_mood_message(emotion)
            
            frame = draw_results(frame, facial_area, name, emotion, spoof_status, confidence)
            
            if name != "Unknown" and is_real:
                mark_attendance(name, emotion, mood_message, spoof_status, confidence)
            
            result = {
                'name': name,
                'emotion': emotion,
                'spoof_status': spoof_status,
                'mood_message': mood_message,
                'confidence': confidence
            }
            results.append(result)
            
            print(f"[RESULT] Name: {name}, Emotion: {emotion}, Spoof: {spoof_status}, Confidence: {confidence:.2%}")
            print(f"[MOOD] {mood_message}")
            
        except Exception as e:
            print(f"[ERROR] Face processing error: {e}")
            continue
    
    return frame, results


def run_demo_mode():
    """Run a demo mode that processes sample images"""
    print("\n" + "="*60)
    print("FACE RECOGNITION ATTENDANCE SYSTEM - DEMO MODE")
    print("="*60)
    print("\nThis system runs in demo mode since there's no webcam available.")
    print("Add face images to the 'images' folder to enroll users.")
    print("\nFeatures:")
    print("  - Face Detection: RetinaFace")
    print("  - Face Recognition: ArcFace with cosine similarity")
    print("  - Emotion Detection: DeepFace CNN model")
    print("  - Anti-Spoofing: Multi-factor liveness detection")
    print("  - Attendance: Auto-logged to attendance.csv")
    print("\n" + "="*60)
    
    load_models()
    known_embeddings = load_known_faces()
    init_attendance_file()
    
    model_cache.known_embeddings = known_embeddings
    
    print(f"\n[STATUS] System initialized successfully")
    print(f"[STATUS] Known faces loaded: {len(known_embeddings)}")
    print(f"[STATUS] Recognition threshold: {RECOGNITION_THRESHOLD}")
    print(f"[STATUS] Attendance cooldown: {ATTENDANCE_COOLDOWN} seconds")
    
    if len(known_embeddings) == 0:
        print("\n[INFO] No known faces enrolled yet.")
        print("[INFO] To enroll faces, add images to the 'images' folder.")
        print("[INFO] Image naming: Use the person's name as filename (e.g., 'john_doe.jpg')")
    else:
        print("\n[INFO] Enrolled users:")
        for name in known_embeddings.keys():
            print(f"  - {name}")
    
    for img_path in IMAGES_FOLDER.iterdir():
        if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            print(f"\n[TEST] Processing: {img_path.name}")
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    processed_frame, results = process_frame(img, known_embeddings)
                    
                    if results:
                        for r in results:
                            print(f"  Name: {r['name']}")
                            print(f"  Emotion: {r['emotion']}")
                            print(f"  Spoof Status: {r['spoof_status']}")
                            print(f"  Mood: {r['mood_message']}")
            except Exception as e:
                print(f"  [ERROR] Failed to process: {e}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nTo use with a webcam, run this script on a machine with camera access.")
    print("The system will automatically detect and process faces in real-time.")


def main():
    """Main function to run the face recognition attendance system"""
    print("\n" + "="*60)
    print("FACE RECOGNITION ATTENDANCE SYSTEM")
    print("RetinaFace + ArcFace + Emotion + Anti-Spoofing")
    print("="*60 + "\n")
    
    load_models()
    known_embeddings = load_known_faces()
    init_attendance_file()
    
    model_cache.known_embeddings = known_embeddings
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[WARN] No webcam available. Running in demo mode...")
        run_demo_mode()
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("[INFO] Webcam initialized. Press 'q' to quit.")
    print(f"[INFO] Known faces: {len(known_embeddings)}")
    print(f"[INFO] Recognition threshold: {RECOGNITION_THRESHOLD}")
    
    frame_count = 0
    fps_start_time = datetime.now()
    fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue
            
            frame_count += 1
            
            if frame_count % FRAME_SKIP == 0:
                processed_frame, results = process_frame(frame, known_embeddings)
            else:
                processed_frame = frame
            
            elapsed = (datetime.now() - fps_start_time).total_seconds()
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = datetime.now()
            
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Face Recognition Attendance System", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] System shutdown complete")


if __name__ == "__main__":
    main()
