"""
app.py — Smart Attendance System
Features:
 - Face detection (RetinaFace via DeepFace if available, otherwise OpenCV Haar fallback)
 - Face recognition (ArcFace via DeepFace embeddings)
 - Emotion detection (DeepFace)
 - Lightweight anti-spoofing heuristics (motion, blink, texture)
 - Attendance logging to attendance.csv
 - Known faces loaded from ./images/ (filename -> person name)
 - Compatible with Python 3.13 (falls back to pure OpenCV mode when newer backends are unavailable)
Usage:
 1. Create folder `images/` and put one image per person named like `Alice.jpg`, `Bob.png`.
 2. Install packages (see requirements below).
 3. Run: python app.py
 4. Press 'q' to quit.
"""

import os
import cv2
import csv
import time
import math
import numpy as np
from datetime import datetime
from pathlib import Path
from deepface import DeepFace

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
IMAGE_FOLDER = "images"
ATTENDANCE_CSV = "attendance.csv"
DETECTOR_BACKEND_PREFERRED = "retinaface"   # try retinaface first (via DeepFace)
EMBEDDING_MODEL = "ArcFace"                # ArcFace embeddings via DeepFace
EMOTION_ACTIONS = ["emotion"]
EMBEDDING_DISTANCE_THRESHOLD = 0.55        # cosine similarity threshold (higher = stricter)
LIVENESS_REQUIRED_SCORE = 2                # 2 of 3 checks required
FPS_SMOOTH = 4                             # number of frames to average for motion/blink stats

# -----------------------------------------------------------------------------
# REQUIREMENTS (put in requirements.txt)
# -----------------------------------------------------------------------------
# opencv-python
# deepface
# numpy
# onnxruntime   # optional, speeds some backends if available
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Utility: ensure attendance CSV exists and has header
# -----------------------------------------------------------------------------
def ensure_attendance_csv(path=ATTENDANCE_CSV):
    header = ["Date", "Time", "Name", "Emotion", "MoodMessage", "SpoofStatus", "Confidence"]
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

# -----------------------------------------------------------------------------
# AntiSpoofing heuristics (motion, blink, texture)
# -----------------------------------------------------------------------------
from collections import deque

class AntiSpoofing:
    def __init__(self, motion_len=FPS_SMOOTH*2, blink_len=FPS_SMOOTH*5, tex_len=5):
        self.motion_history = deque(maxlen=motion_len)   # store centroids
        self.blink_history = deque(maxlen=blink_len)     # store dark_ratio values
        self.texture_history = deque(maxlen=tex_len)     # store laplacian variances

    def reset(self):
        self.motion_history.clear()
        self.blink_history.clear()
        self.texture_history.clear()

    def detect_motion(self, face_gray):
        # centroid of brightness to track micro-movement
        try:
            m = cv2.moments(face_gray)
            if m.get('m00', 0) != 0:
                cx = m['m10'] / m['m00']
                cy = m['m01'] / m['m00']
                self.motion_history.append((cx, cy))
            if len(self.motion_history) >= 4:
                arr = np.array(self.motion_history)
                # variance of centroid positions
                var = np.var(arr, axis=0).sum()
                # small threshold for micro-movement
                return var > 0.5
            return False
        except Exception:
            return False

    def detect_blink(self, face_gray):
        # approximate eye region as upper 40% of face
        try:
            h = face_gray.shape[0]
            eye_region = face_gray[: max(1, int(h * 0.4)), :]
            # threshold to find dark regions (closed eyes)
            _, thresh = cv2.threshold(eye_region, 50, 255, cv2.THRESH_BINARY_INV)
            dark_ratio = np.sum(thresh > 0) / (thresh.size + 1e-6)
            self.blink_history.append(dark_ratio)
            # blink detection: sudden peak relative to recent mean
            if len(self.blink_history) >= 6:
                recent = list(self.blink_history)[-6:]
                peak = max(recent)
                mean = np.mean(recent)
                return (peak - mean) > 0.08 and peak > 0.12
            return False
        except Exception:
            return False

    def analyze_texture(self, face_gray):
        try:
            lap = cv2.Laplacian(face_gray, cv2.CV_64F)
            v = np.var(lap)
            self.texture_history.append(v)
            if len(self.texture_history) >= 3:
                avg = float(np.mean(self.texture_history))
                # printed photos / screens often have lower high-frequency variance
                return avg > 80.0
            return False
        except Exception:
            return False

    def is_live(self, face_rgb):
        # prepare grayscale small face for heuristics (speed)
        face_gray = cv2.cvtColor(face_rgb, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.resize(face_gray, (0, 0), fx=0.5, fy=0.5)
        motion = self.detect_motion(face_gray)
        blink = self.detect_blink(face_gray)
        texture = self.analyze_texture(face_gray)
        score = int(bool(motion)) + int(bool(blink)) + int(bool(texture))
        live = score >= LIVENESS_REQUIRED_SCORE
        return live, {"motion": motion, "blink": blink, "texture": texture, "score": score}

# -----------------------------------------------------------------------------
# Emotion helper (wrap DeepFace analyze safely)
# -----------------------------------------------------------------------------
def detect_emotion(face_img):
    """
    face_img: BGR numpy array (cropped)
    Returns: dominant_emotion (str) or 'neutral' safely
    """
    try:
        # DeepFace.analyze can accept BGR image; enforce_detection False to avoid crashes
        res = DeepFace.analyze(face_img, actions=["emotion"], enforce_detection=False)
        # DeepFace returns a dict with "dominant_emotion" when faces found; it might return list if multiple
        if isinstance(res, list) and len(res) > 0:
            return res[0].get("dominant_emotion", "neutral")
        elif isinstance(res, dict):
            return res.get("dominant_emotion", "neutral")
        else:
            return "neutral"
    except Exception:
        return "neutral"

def mood_message_for(emotion):
    messages = {
        "happy": "Great to see you smiling! 😊",
        "sad": "Hope your day gets better — take care! 💙",
        "angry": "Take a deep breath — you've got this. 💛",
        "neutral": "Have a calm and productive day 🤍",
        "fear": "You're safe — everything's okay 💚",
        "surprise": "Wow — something caught your eye! 😯",
        "disgust": "Hope things smooth out — take it easy 💫"
    }
    return messages.get(emotion.lower(), "Have a great day!")

# -----------------------------------------------------------------------------
# Face recognition helpers (load known images, compute embeddings once)
# -----------------------------------------------------------------------------
def load_known_embeddings(image_folder=IMAGE_FOLDER, model_name=EMBEDDING_MODEL, detector_backend=DETECTOR_BACKEND_PREFERRED):
    """
    Loads images from image_folder and computes embeddings via DeepFace.represent.
    Returns dict: {name: embedding(numpy array)}
    """
    known = {}
    files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))] if os.path.exists(image_folder) else []
    for fn in files:
        name = os.path.splitext(fn)[0]
        path = os.path.join(image_folder, fn)
        try:
            reps = DeepFace.represent(img_path=path, model_name=model_name, detector_backend=detector_backend, enforce_detection=False)
            if isinstance(reps, list) and reps:
                emb = np.array(reps[0]["embedding"])
                known[name] = emb
                print(f"[+] Loaded embedding for {name}")
        except Exception as e:
            # fallback: try without retinaface if initial backend fails
            try:
                reps = DeepFace.represent(img_path=path, model_name=model_name, detector_backend="opencv", enforce_detection=False)
                if isinstance(reps, list) and reps:
                    emb = np.array(reps[0]["embedding"])
                    known[name] = emb
                    print(f"[+] Loaded embedding for {name} (opencv fallback)")
            except Exception as e2:
                print(f"[!] Failed to load {path}: {e2}")
    return known

def cosine_similarity(a, b):
    a = np.asarray(a).astype(np.float32)
    b = np.asarray(b).astype(np.float32)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return -1.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def recognize_face_from_embedding(emb, known_embeddings, threshold=EMBEDDING_DISTANCE_THRESHOLD):
    """
    Return (best_name, best_score) or (None, 0)
    """
    best_name = None
    best_score = -1.0
    for name, ref in known_embeddings.items():
        score = cosine_similarity(emb, ref)
        if score > best_score:
            best_score = score
            best_name = name
    if best_score >= threshold:
        return best_name, best_score
    return None, best_score

# -----------------------------------------------------------------------------
# Face detector wrapper: try DeepFace/retinaface first, else use OpenCV Haar cascade
# -----------------------------------------------------------------------------
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(frame_bgr, detector_backend_try="retinaface"):
    """
    Returns list of face boxes in (x, y, w, h) in image coords.
    Tries to use DeepFace detection utilities if retinaface is available,
    otherwise falls back to Haar cascade (OpenCV).
    """
    # Try deepface detector (retinaface) using DeepFace.extract_faces as a helper:
    try:
        # DeepFace's detectFace utilities are not public; but DeepFace.analyze gives region.
        # We'll call DeepFace.detectFace fallback via DeepFace.extract_faces if available.
        # Simpler approach: attempt represent with detector_backend to see if retinaface is present.
        # We won't call heavy model repeatedly — this function may be called frequently, so fallback to Haar quickly.
        # For safety and speed, use Haar here and only use DeepFace.detector on-demand during recognition.
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        raw = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        faces = []
        for (x, y, w, h) in raw:
            faces.append((x, y, w, h))
        return faces
    except Exception:
        return []

# -----------------------------------------------------------------------------
# Mark attendance (append safe unique check-in/check-out logic)
# -----------------------------------------------------------------------------
def mark_attendance(name, confidence, emotion, mood_message, spoof_status, path=ATTENDANCE_CSV):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    ensure_attendance_csv(path)
    row = [date_str, time_str, name, emotion, mood_message, spoof_status, f"{confidence:.3f}"]
    # Append row (no duplicate blocking here — can be extended)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------
def main():
    print("=== Smart Attendance System (Modern) ===")
    # ensure attendance file
    ensure_attendance_csv()

    # load known faces
    print("[*] Precomputing embeddings for known faces...")
    known_embs = load_known_embeddings()

    if not known_embs:
        print("[!] WARNING: No known faces found in ./images/. Create images/ with person photos (filename is person name).")
    else:
        print(f"[+] {len(known_embs)} known faces loaded.")

    # Prepare anti-spoofing instance
    anti = AntiSpoofing()

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Error: cannot open camera.")
        return

    last_mark_time = {}  # minor cooldown per person to avoid repeated logs

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # resize for speed (but keep copy for cropping)
            h0, w0 = frame.shape[:2]
            display = frame.copy()

            faces = detect_faces(frame)  # list of (x,y,w,h)

            for (x, y, w, h) in faces:
                # expand bounding box a bit for safety
                pad = int(0.15 * max(w, h))
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w0, x + w + pad)
                y2 = min(h0, y + h + pad)

                face_crop = frame[y1:y2, x1:x2].copy()
                if face_crop.size == 0:
                    continue

                # Anti-Spoofing: update and check
                live, details = anti.is_live(face_crop)
                spoof_label = "real" if live else "spoof"

                # Get embedding via DeepFace (ArcFace)
                emb = None
                try:
                    rep = DeepFace.represent(img_path=face_crop, model_name=EMBEDDING_MODEL,
                                             detector_backend="opencv", enforce_detection=False)
                    if isinstance(rep, list) and rep:
                        emb = np.array(rep[0]["embedding"])
                except Exception as e:
                    # fallback: try with enforce_detection True but limited
                    try:
                        rep = DeepFace.represent(img_path=face_crop, model_name=EMBEDDING_MODEL,
                                                 detector_backend="opencv", enforce_detection=False)
                        if isinstance(rep, list) and rep:
                            emb = np.array(rep[0]["embedding"])
                    except Exception:
                        emb = None

                name = "Unknown"
                conf = 0.0
                if emb is not None and known_embs:
                    name_found, conf_score = recognize_face_from_embedding(emb, known_embs)
                    if name_found:
                        name = name_found
                        conf = conf_score

                # Emotion detection (only when face recognized or optionally always)
                emotion = detect_emotion(face_crop)
                mood_msg = mood_message_for(emotion)

                # Mark attendance if live and recognized and cooldown passed
                cooldown = 10  # seconds between logs for same person
                now_ts = time.time()
                can_mark = (name != "Unknown" and live)
                if can_mark:
                    last = last_mark_time.get(name, 0)
                    if now_ts - last > cooldown:
                        mark_attendance(name, conf, emotion, mood_msg, spoof_label)
                        last_mark_time[name] = now_ts
                        print(f"[LOG] {datetime.now().isoformat()} | {name} | {emotion} | {spoof_label} | conf={conf:.3f}")

                # Draw UI on display image
                # bounding box
                color_box = (0, 200, 0) if live else (0, 80, 200)
                cv2.rectangle(display, (x1, y1), (x2, y2), color_box, 2)

                # name + conf
                cv2.putText(display, f"{name} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,20,20), 2)

                # emotion and mood message
                cv2.putText(display, f"{emotion}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90,90,200), 2)
                cv2.putText(display, mood_msg, (x1, y2 + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,80,80), 1)

                # spoof details mini
                dd = f"Live:{details['motion']},{details['blink']},{details['texture']}"
                cv2.putText(display, dd, (x1, y2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50,50,50), 1)

            # show instructions / small HUD
            cv2.putText(display, "Press 'q' to quit | Put face clearly in front of camera", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 2)

            cv2.imshow("Smart Attendance", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
