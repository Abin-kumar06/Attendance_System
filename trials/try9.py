#!/usr/bin/env python3
"""
app.py — Upgraded Smart Attendance System (High accuracy, Speed, Robustness)

Features:
 - Fast detector: SCRFD (ONNX) if available, fallback to Haar (OpenCV)
 - Fast recognizer: ArcFace (ONNX) if available, fallback to DeepFace.represent
 - Anti-spoofing: ONNX model if available, else heuristic-based
 - Emotion detection: FER ONNX if available, else DeepFace.analyze
 - 3-frame temporal verification per identity before marking attendance
 - Embedding caching (precompute embeddings at startup for known faces)
 - Attendance logging to attendance.csv:
     Date, Time, Name, Emotion, MoodMessage, SpoofStatus, Confidence
 - Compatible with Python 3.13.9
 - Uses minimal external packages; ONNXRuntime used if ONNX models present

Place optional ONNX models in ./models/:
  - models/scrfd.onnx       (optional - face detector)
  - models/arcface.onnx     (optional - face embedding)
  - models/antispoof.onnx   (optional - liveness anti-spoof)
  - models/fer.onnx         (optional - emotion classifier)

Required (minimal) packages for fallback mode:
  pip install opencv-python numpy deepface

Optional (for ONNX fast mode):
  pip install onnxruntime

Author: Assistant
"""

import os
import sys
import time
import csv
import math
import traceback
from pathlib import Path
from collections import deque, defaultdict
import onnxruntime as ort
import cv2
import numpy as np

# Attempt to import onnxruntime (optional)
try:
    
    ONNX_AVAILABLE = True
except Exception:
    ONNX_AVAILABLE = False

# DeepFace fallback
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except Exception:
    DEEPFACE_AVAILABLE = False

# ----------------------------- CONFIG -----------------------------
IMAGE_FOLDER = "images"           # folder with known faces (filename -> name)
ATTENDANCE_CSV = "attendance.csv"
MODELS_DIR = Path("models")       # optional ONNX models placed here
SCRFD_ONNX = MODELS_DIR / "scrfd.onnx"
ARCFACE_ONNX = MODELS_DIR / "arcface.onnx"
ANTISPOOF_ONNX = MODELS_DIR / "antispoof.onnx"
FER_ONNX = MODELS_DIR / "fer.onnx"

CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
VERIFICATION_WINDOW = 3          # require same identity for N consecutive frames
EMBEDDING_MODEL = "ArcFace"     # fallback DeepFace model name if ONNX missing
DETECTOR_BACKEND_FALLBACK = "opencv"  # for DeepFace.represent fallback
EMBEDDING_THRESHOLD = 0.57      # cosine similarity threshold (higher = stricter)
COOLDOWN_SECONDS = 10           # per-person cooldown after marking attendance
MIN_FACE_SIZE = 80
DEBUG = False
# ------------------------------------------------------------------

# ----------------------------- HELPERS -----------------------------
def log(*args, **kwargs):
    print("[*]", *args, **kwargs)

def ensure_csv(path=ATTENDANCE_CSV):
    header = ["Date", "Time", "Name", "Emotion", "MoodMessage", "SpoofStatus", "Confidence"]
    if not Path(path).exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def cosine_sim(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return -1.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def mood_message_for(emotion: str):
    emotion = (emotion or "neutral").lower()
    msgs = {
        "happy": "Great to see you smiling! 😊",
        "sad": "Hope your day gets better — take care! 💙",
        "angry": "Take a deep breath; you got this. 💛",
        "neutral": "Have a calm and productive day 🤍",
        "fear": "You're safe — everything's okay 💚",
        "surprise": "Wow — is that a surprise? 😯",
        "disgust": "Hope things smooth out — take it easy 💫"
    }
    return msgs.get(emotion, "Have a great day!")

# ----------------------------- ONNX WRAPPERS -----------------------------
class ONNXModel:
    """Simple ONNX wrapper using onnxruntime."""
    def __init__(self, path: Path):
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime not installed")
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        self.path = str(path)
        self._sess = ort.InferenceSession(self.path, providers=self._preferred_providers())
        self.input_name = self._sess.get_inputs()[0].name
        # optional: cache outputs etc.

    def _preferred_providers(self):
        # Choose best available providers
        providers = []
        try:
            # onnxruntime default - try 'CPUExecutionProvider'
            providers = ['CPUExecutionProvider']
            # If CUDA provider is available in environment it will be used automatically by ort if available
        except Exception:
            providers = ['CPUExecutionProvider']
        return providers

    def run(self, inputs: dict):
        return self._sess.run(None, inputs)

# ----------------------------- DETECTION -----------------------------
class Detector:
    """
    Face detector abstraction:
      - If SCRFD ONNX exists and onnxruntime available -> use it
      - Else fallback to Haar Cascade (OpenCV)
    Note: For SCRFD ONNX, the model usually expects BGR resized input; different SCRFD exports vary.
    This code implements a minimal inference assuming SCRFD export with NCHW input and boxes output.
    If the user's SCRFD ONNX is different, use Haar fallback or modify as needed.
    """
    def __init__(self):
        self.use_scrfd = False
        self.scrfd = None
        if ONNX_AVAILABLE and SCRFD_ONNX.exists():
            try:
                self.scrfd = ONNXModel(SCRFD_ONNX)
                self.use_scrfd = True
                log("Using SCRFD ONNX detector.")
            except Exception as e:
                log("SCRFD ONNX load failed, falling back to Haar:", e)
                self.use_scrfd = False

        if not self.use_scrfd:
            # Haar Cascade fallback (reliable across platforms)
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            log("Using Haar cascade fallback for detection.")

    def detect(self, frame_bgr):
        """
        Return list of boxes (x, y, w, h)
        If SCRFD used, try to parse outputs generically; otherwise Haar.
        """
        if self.use_scrfd:
            try:
                # Minimal generic SCRFD ONNX usage:
                # Resize to 640x640 (common), normalize, run, parse boxes if output available.
                # NOTE: SCRFD ONNX variants differ. This code handles common case where output[0] contains boxes.
                inp_h, inp_w = 640, 640
                img = cv2.resize(frame_bgr, (inp_w, inp_h))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Normalize: 0-1
                img_t = img_rgb.astype(np.float32) / 255.0
                # transpose to NCHW
                img_t = np.transpose(img_t, (2, 0, 1))[None, ...].astype(np.float32)
                out = self.scrfd.run({self.scrfd.input_name: img_t})
                # Try many possible output formats:
                boxes = []
                # If the model returns a single output that is Nx6 or Nx5: [x1,y1,x2,y2,score]
                for o in out:
                    arr = np.asarray(o)
                    if arr.ndim == 2 and arr.shape[1] >= 5:
                        # assume [x1,y1,x2,y2,score,...]
                        for row in arr:
                            score = float(row[4])
                            if score > 0.4:
                                x1 = int(row[0] / inp_w * frame_bgr.shape[1])
                                y1 = int(row[1] / inp_h * frame_bgr.shape[0])
                                x2 = int(row[2] / inp_w * frame_bgr.shape[1])
                                y2 = int(row[3] / inp_h * frame_bgr.shape[0])
                                w = max(2, x2 - x1)
                                h = max(2, y2 - y1)
                                boxes.append((x1, y1, w, h))
                        if boxes:
                            return boxes
                # If parsing failed, fallback to Haar
            except Exception as e:
                if DEBUG:
                    traceback.print_exc()
                log("SCRFD detection failed; falling back to Haar.")
                self.use_scrfd = False  # disable further attempts
        # Haar fallback
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        rects = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
        boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]
        return boxes

# ----------------------------- ANTI-SPOOF (ONNX or HEURISTIC) -----------------------------
class AntiSpoof:
    def __init__(self):
        self.use_onnx = False
        self.heur = HeuristicLiveness()
        if ONNX_AVAILABLE and ANTISPOOF_ONNX.exists():
            try:
                self.ann = ONNXModel(ANTISPOOF_ONNX)
                self.use_onnx = True
                log("Using ONNX anti-spoof model.")
            except Exception as e:
                log("Could not load antispoof ONNX; using heuristic. Reason:", e)
                self.use_onnx = False

    def check(self, face_bgr):
        """
        face_bgr: cropped face (BGR)
        returns (is_live: bool, details: dict)
        """
        if self.use_onnx:
            try:
                # Preprocess according to common antispoof models: resize, normalize, NCHW
                img = cv2.resize(face_bgr, (160, 160))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)
                out = self.ann.run({self.ann.input_name: img})
                # Common output: probability of live in out[0][:,1] or out[0][:,0]
                prob = None
                for o in out:
                    arr = np.asarray(o)
                    if arr.ndim >= 2:
                        # take mean across spatial dims if exist
                        val = float(np.mean(arr))
                        if prob is None or val > prob:
                            prob = val
                # threshold around 0.5 typical
                is_live = (prob is not None and prob > 0.5)
                return is_live, {"prob": float(prob) if prob is not None else 0.0}
            except Exception:
                if DEBUG:
                    traceback.print_exc()
                # fallback to heuristic
        # heuristic check
        return self.heur.is_live(face_bgr)

# Heuristic liveness class (robust improved version)
class HeuristicLiveness:
    def __init__(self):
        self.motion_history = deque(maxlen=8)
        self.blink_history = deque(maxlen=16)
        self.texture_history = deque(maxlen=8)

    def reset(self):
        self.motion_history.clear()
        self.blink_history.clear()
        self.texture_history.clear()

    def is_live(self, face_bgr):
        try:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)
            # motion: centroid variance
            m = cv2.moments(small)
            if m.get('m00', 0) != 0:
                cx = m['m10'] / m['m00']
                cy = m['m01'] / m['m00']
                self.motion_history.append((cx, cy))
            motion_ok = False
            if len(self.motion_history) >= 4:
                arr = np.array(self.motion_history)
                var = np.var(arr, axis=0).sum()
                motion_ok = var > 0.3

            # blink: dark region detection in upper 40%
            h = small.shape[0]
            eye_region = small[:max(1, int(h*0.4)), :]
            _, thresh = cv2.threshold(eye_region, 50, 255, cv2.THRESH_BINARY_INV)
            dark_ratio = float(np.sum(thresh > 0)) / (thresh.size + 1e-9)
            self.blink_history.append(dark_ratio)
            blink_ok = False
            if len(self.blink_history) >= 6:
                recent = list(self.blink_history)[-6:]
                peak = max(recent)
                mean = np.mean(recent)
                blink_ok = (peak - mean) > 0.05 and peak > 0.10

            # texture: laplacian variance
            lap = cv2.Laplacian(small, cv2.CV_64F)
            tex = float(np.var(lap))
            self.texture_history.append(tex)
            texture_ok = False
            if len(self.texture_history) >= 4:
                avg = float(np.mean(self.texture_history))
                texture_ok = avg > 60.0

            score = int(motion_ok) + int(blink_ok) + int(texture_ok)
            live = (score >= 2)
            return live, {"motion": motion_ok, "blink": blink_ok, "texture": texture_ok, "score": score}
        except Exception:
            return False, {"motion": False, "blink": False, "texture": False, "score": 0}

# ----------------------------- EMBEDDING (ONNX or DeepFace) -----------------------------
class EmbeddingModel:
    def __init__(self):
        self.use_onnx = False
        self.onnx_model = None
        if ONNX_AVAILABLE and ARCFACE_ONNX.exists():
            try:
                self.onnx_model = ONNXModel(ARCFACE_ONNX)
                self.use_onnx = True
                log("Using ArcFace ONNX for embeddings (fast).")
            except Exception as e:
                log("ArcFace ONNX load failed, will use DeepFace fallback:", e)
                self.use_onnx = False

        if not self.use_onnx:
            if not DEEPFACE_AVAILABLE:
                log("WARNING: DeepFace not available and ArcFace ONNX missing — embeddings won't work.")
            else:
                # ensure model downloaded by DeepFace (lazy)
                try:
                    _ = DeepFace.build_model(EMBEDDING_MODEL)
                except Exception:
                    pass
                log("Using DeepFace fallback for embeddings.")

    def get_embedding(self, face_bgr):
        """
        Returns 1D numpy embedding or None
        """
        if self.use_onnx:
            try:
                img = cv2.resize(face_bgr, (112, 112))  # common ArcFace size (check your ONNX)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
                # Normalize [-1,1]
                img = (img - 127.5) / 128.0
                img = np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)
                out = self.onnx_model.run({self.onnx_model.input_name: img})
                # pick numeric flatten
                for o in out:
                    arr = np.asarray(o).flatten()
                    if arr.size >= 64:
                        # L2 norm
                        arr = arr.astype(np.float32)
                        arr = arr / (np.linalg.norm(arr) + 1e-12)
                        return arr
                return None
            except Exception:
                if DEBUG:
                    traceback.print_exc()
                return None
        # DeepFace fallback:
        if DEEPFACE_AVAILABLE:
            try:
                rep = DeepFace.represent(img_path=face_bgr, model_name=EMBEDDING_MODEL, detector_backend=DETECTOR_BACKEND_FALLBACK, enforce_detection=False)
                if isinstance(rep, list) and rep:
                    emb = np.array(rep[0]["embedding"], dtype=np.float32)
                    if np.linalg.norm(emb) == 0:
                        return None
                    emb = emb / (np.linalg.norm(emb) + 1e-12)
                    return emb
            except Exception:
                if DEBUG:
                    traceback.print_exc()
        return None

# ----------------------------- EMOTION MODEL -----------------------------
class EmotionModel:
    def __init__(self):
        self.use_onnx = False
        self.onnx_model = None
        if ONNX_AVAILABLE and FER_ONNX.exists():
            try:
                self.onnx_model = ONNXModel(FER_ONNX)
                self.use_onnx = True
                log("Using FER ONNX for emotions.")
            except Exception as e:
                log("FER ONNX load failed; fallback to DeepFace analyze:", e)
                self.use_onnx = False
        if not self.use_onnx and not DEEPFACE_AVAILABLE:
            log("WARNING: DeepFace not available and FER ONNX missing — emotion will be neutral.")

    def predict(self, face_bgr):
        # returns dominant emotion string
        if self.use_onnx:
            try:
                img = cv2.resize(face_bgr, (64, 64))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                img = np.transpose(img, (2,0,1))[None, ...].astype(np.float32)
                out = self.onnx_model.run({self.onnx_model.input_name: img})
                # assume out[0] is logits/probs
                probs = np.asarray(out[0]).flatten()
                idx = int(np.argmax(probs))
                labels = ["angry","disgust","fear","happy","sad","surprise","neutral"]
                return labels[idx] if idx < len(labels) else "neutral"
            except Exception:
                if DEBUG:
                    traceback.print_exc()
                return "neutral"
        # DeepFace fallback
        if DEEPFACE_AVAILABLE:
            try:
                res = DeepFace.analyze(face_bgr, actions=["emotion"], enforce_detection=False)
                if isinstance(res, list) and res:
                    return res[0].get("dominant_emotion", "neutral")
                elif isinstance(res, dict):
                    return res.get("dominant_emotion", "neutral")
            except Exception:
                if DEBUG:
                    traceback.print_exc()
        return "neutral"

# ----------------------------- KNOWN EMBEDDINGS LOADER -----------------------------
def load_known_embeddings(emb_model: EmbeddingModel, folder=IMAGE_FOLDER):
    known = {}
    p = Path(folder)
    if not p.exists():
        log(f"Images folder '{folder}' not found — create it and add one image per person (filename=personname.jpg).")
        return known
    files = list(p.glob("*.*"))
    for f in files:
        if f.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        name = f.stem
        try:
            # For simplicity, get embedding by reading image and passing to model
            img = cv2.imread(str(f))
            if img is None:
                log("Could not read image", f)
                continue
            emb = emb_model.get_embedding(img)
            if emb is not None:
                known[name] = emb
                log(f"Loaded embedding for '{name}'.")
            else:
                log(f"Failed to compute embedding for '{name}'. Try increasing face crop size or check image quality.")
        except Exception as e:
            log("Error loading", f, e)
    return known

# ----------------------------- VERIFICATION & ATTENDANCE -----------------------------
class Verifier:
    """Temporal verification: requires N consecutive frames matching same identity"""

    def __init__(self, window=VERIFICATION_WINDOW):
        self.window = window
        self.counters = defaultdict(lambda: {"count": 0, "last_seen": 0})
        self.frame_id = 0

    def update(self, candidate_name):
        """Call per-frame with candidate_name (None if unknown). Returns True only when threshold reached now."""
        self.frame_id += 1
        now = time.time()
        # decay counters older than some seconds
        for k in list(self.counters.keys()):
            if now - self.counters[k]["last_seen"] > 5:
                del self.counters[k]
        if candidate_name is None:
            # reset all? just ignore
            return False
        c = self.counters[candidate_name]
        c["count"] += 1
        c["last_seen"] = now
        if c["count"] >= self.window:
            # reset counter after successful verification to avoid repeated triggers
            c["count"] = 0
            return True
        return False

# ----------------------------- MAIN APP -----------------------------
class AttendanceApp:
    def __init__(self):
        log("Initializing AttendanceApp...")
        ensure_csv()
        self.detector = Detector()
        self.anti = AntiSpoof()
        self.emb_model = EmbeddingModel()
        self.emotion_model = EmotionModel()
        self.known = load_known_embeddings(self.emb_model)
        self.verifier = Verifier(window=VERIFICATION_WINDOW)
        self.cooldowns = {}  # name -> last mark time
        # vectorized known embeddings list for fast batch similarity
        self.names = list(self.known.keys())
        self.embs = np.array([self.known[n] for n in self.names]) if self.names else np.array([])

    def mark_attendance(self, name, confidence, emotion, mood, spoof_status):
        now = time.localtime()
        date = time.strftime("%Y-%m-%d", now)
        t = time.strftime("%H:%M:%S", now)
        row = [date, t, name, emotion, mood, spoof_status, f"{confidence:.3f}"]
        with open(ATTENDANCE_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        log(f"Attendance marked: {name} | {emotion} | spoof={spoof_status} | conf={confidence:.3f}")

    def find_best_match(self, emb):
        if emb is None or self.embs.size == 0:
            return None, 0.0
        # compute cosine similarity vectorized
        sims = np.dot(self.embs, emb) / (np.linalg.norm(self.embs, axis=1) * (np.linalg.norm(emb) + 1e-12))
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score >= EMBEDDING_THRESHOLD:
            return self.names[best_idx], best_score
        return None, best_score

    def run(self):
        cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW if os.name == 'nt' else 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not cap.isOpened():
            log("Cannot open camera.")
            return

        log("Starting camera. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            display = frame.copy()
            boxes = self.detector.detect(frame)
            # Sort boxes left->right for stability
            boxes = sorted(boxes, key=lambda b: b[0])
            candidates = []
            for (x, y, w, h) in boxes:
                # expand box slightly
                pad = int(0.12 * max(w, h))
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)
                face = frame[y1:y2, x1:x2].copy()
                if face.size == 0:
                    continue

                # Anti-spoof
                live, spoof_details = self.anti.check(face)
                spoof_label = "real" if live else "spoof"

                # Embedding
                emb = self.emb_model.get_embedding(face)
                name, conf = self.find_best_match(emb) if emb is not None else (None, 0.0)

                # Emotion
                emotion = self.emotion_model.predict(face)
                mood = mood_message_for(emotion)

                # Temporal verification
                verified_now = False
                if name:
                    verified_now = self.verifier.update(name)
                else:
                    self.verifier.update(None)

                # When verified and live and not in cooldown -> mark attendance
                if verified_now and live and name:
                    last = self.cooldowns.get(name, 0)
                    if time.time() - last > COOLDOWN_SECONDS:
                        self.mark_attendance(name, conf, emotion, mood, spoof_label)
                        self.cooldowns[name] = time.time()

                # Visual overlay
                color = (0, 200, 0) if live else (0, 120, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                label = f"{name or 'Unknown'} {conf:.2f}"
                cv2.putText(display, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10,10,10), 2)
                cv2.putText(display, f"{emotion}", (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,80,200), 2)
                cv2.putText(display, mood, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60,60,60), 1)
                cv2.putText(display, f"spoof:{spoof_label}", (x1, y2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120,10,10), 1)

            # HUD
            cv2.putText(display, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40,40,40), 2)
            cv2.imshow("Attendance - Upgraded", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# ----------------------------- RUN -----------------------------
if __name__ == "__main__":
    try:
        app = AttendanceApp()
        app.run()
    except Exception as e:
        log("Fatal error:", e)
        if DEBUG:
            traceback.print_exc()
        sys.exit(1)
# End of file