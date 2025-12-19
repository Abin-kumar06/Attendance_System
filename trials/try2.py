"""
Smart Attendance System using DeepFace for face recognition and emotion detection.

Requirements (Python 3.13):
  pip install opencv-python>=4.12 deepface>=0.0.92 numpy>=2.2

Project layout:
  images/         # place one image per person (file name becomes the label)
  attendance.csv  # auto-created with Name, Time, Emotion
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from deepface import DeepFace

# Paths relative to this script
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
ATTENDANCE_CSV = BASE_DIR / "attendance.csv"

# Model / detection settings
EMBEDDING_MODEL = "Facenet512"  # accurate and available in DeepFace
DETECTOR_BACKEND = "retinaface"  # reliable detector bundled with DeepFace
SIMILARITY_THRESHOLD = 0.38      # lower is stricter; tune if needed


# -------- Utility helpers --------
def ensure_images_dir() -> None:
    """Create images/ placeholder if missing to guide the user."""
    IMAGES_DIR.mkdir(exist_ok=True)
    if not any(IMAGES_DIR.iterdir()):
        print(f"[info] Add reference photos to {IMAGES_DIR} (one file per person).")


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Return cosine similarity between two embedding vectors."""
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-10
    return float(np.dot(vec_a, vec_b) / denom)


def load_known_faces() -> Dict[str, np.ndarray]:
    """
    Load reference embeddings from images/ where each filename is the label.
    Returns a dict {name: embedding}.
    """
    embeddings: Dict[str, np.ndarray] = {}
    ensure_images_dir()

    for img_path in sorted(IMAGES_DIR.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        name = img_path.stem
        try:
            # represent returns a list even for a single image; take the first entry
            reps = DeepFace.represent(
                img_path=str(img_path),
                model_name=EMBEDDING_MODEL,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
                align=True,
            )
            if isinstance(reps, list) and reps:
                embeddings[name] = np.array(reps[0]["embedding"], dtype=np.float32)
                print(f"[loaded] {name} from {img_path.name}")
        except Exception as exc:  # DeepFace raises generic exceptions on detection issues
            print(f"[warn] Could not process {img_path.name}: {exc}")

    if not embeddings:
        print("[warn] No valid reference faces found in images/. Recognition will show 'Unknown'.")
    return embeddings


def match_face(embedding: np.ndarray, known: Dict[str, np.ndarray]) -> Tuple[str, float]:
    """
    Compare an embedding to known embeddings and return (best_name, similarity).
    If no match passes the threshold, returns ("Unknown", best_similarity).
    """
    best_name = "Unknown"
    best_sim = -1.0
    for name, ref_emb in known.items():
        sim = cosine_similarity(embedding, ref_emb)
        if sim > best_sim:
            best_sim = sim
            best_name = name
    if best_sim < SIMILARITY_THRESHOLD:
        return "Unknown", best_sim
    return best_name, best_sim


def mood_message(emotion: str) -> str:
    """Map dominant emotion to a short mood-based message."""
    emotion = (emotion or "").lower()
    messages = {
        "happy": "Great to see you smiling!",
        "neutral": "Hope your day goes well.",
        "sad": "We're here for you.",
        "angry": "Take a breath—you're doing fine.",
        "surprise": "Surprises keep life interesting!",
        "fear": "You're safe here.",
        "disgust": "Let's find something better.",
    }
    return messages.get(emotion, "Welcome!")


def mark_attendance(name: str, emotion: str) -> None:
    """Append attendance to CSV with header if file is new."""
    is_new = not ATTENDANCE_CSV.exists()
    ATTENDANCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with ATTENDANCE_CSV.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if is_new:
            writer.writerow(["Name", "Time", "Emotion"])
        writer.writerow([name, datetime.now().isoformat(timespec="seconds"), emotion])


# -------- Main processing loop --------
def main() -> None:
    known_embeddings = load_known_faces()
    attendance_done: set[str] = set()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[error] Unable to access webcam.")
        sys.exit(1)

    print("[info] Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[warn] Empty frame from webcam.")
                continue

            try:
                # Detect faces and emotions; detect_multiple=True returns list per face
                analyses: List[dict] = DeepFace.analyze(
                    img_path=frame,
                    actions=["emotion"],
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                    align=True,
                    silent=True,
                    detect_multiple=True,
                )
                if not isinstance(analyses, list):
                    analyses = [analyses]
            except Exception as exc:
                print(f"[warn] Detection failed: {exc}")
                analyses = []

            if not analyses:
                cv2.putText(frame, "No face detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Smart Attendance", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            for result in analyses:
                region = result.get("region") or result.get("facial_area") or {}
                x, y, w, h = (
                    int(region.get("x", 0)),
                    int(region.get("y", 0)),
                    int(region.get("w", 0)),
                    int(region.get("h", 0)),
                )

                # Validate bounding box
                if w <= 0 or h <= 0:
                    continue

                # Crop face safely within frame bounds
                y1, y2 = max(y, 0), min(y + h, frame.shape[0])
                x1, x2 = max(x, 0), min(x + w, frame.shape[1])
                face_crop = frame[y1:y2, x1:x2]

                try:
                    rep = DeepFace.represent(
                        img_path=face_crop,
                        model_name=EMBEDDING_MODEL,
                        detector_backend="skip",  # already cropped
                        enforce_detection=False,
                        align=True,
                    )
                    embedding = np.array(rep[0]["embedding"], dtype=np.float32)
                    name, similarity = match_face(embedding, known_embeddings)
                except Exception as exc:
                    print(f"[warn] Embedding failed: {exc}")
                    name, similarity = "Unknown", -1.0

                emotion = result.get("dominant_emotion", "Neutral").capitalize()
                message = mood_message(emotion)

                # Mark attendance once per session per recognized name (skip Unknown)
                if name != "Unknown" and name not in attendance_done:
                    mark_attendance(name, emotion)
                    attendance_done.add(name)

                # Draw bounding box and labels
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{name} ({similarity:.2f})" if similarity >= 0 else name
                cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Emotion: {emotion}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, message, (x1, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Smart Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

