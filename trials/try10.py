import cv2
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN
from fer import FER
import time

detector = MTCNN()
emotion_model = FER()

print("Loading ArcFace model…")
DeepFace.build_model("ArcFace")
print("Model loaded!")

# -----------------------------
# Load known person embedding
# -----------------------------
KNOWN_PATH = "known/person1.jpg"

known_embedding = DeepFace.represent(
    img_path=KNOWN_PATH,
    model_name="ArcFace",
    detector_backend="skip"   # MTCNN used externally
)[0]["embedding"]

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def anti_spoof(frame, box):
    x, y, w, h = box
    face = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap > 60   # threshold

# -----------------------------------
# camera
# -----------------------------------
cap = cv2.VideoCapture(0)

frame_count = 0
verified_count = 0
recognized_name = "UNKNOWN"
emotion_text = "-"

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    detections = detector.detect_faces(frame)

    if len(detections) > 0:
        d = detections[0]
        x, y, w, h = d["box"]

        # Bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # Spoof check
        real = anti_spoof(frame, (x, y, w, h))
        spoof_text = "REAL" if real else "SPOOF"
        cv2.putText(frame, spoof_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0) if real else (0,0,255), 2)

        if real:
            face_crop = frame[y:y+h, x:x+w]

            try:
                rep = DeepFace.represent(
                    img_path=face_crop,
                    model_name="ArcFace",
                    detector_backend="skip"
                )[0]["embedding"]

                sim = cosine_sim(known_embedding, rep)

                if sim > 0.47:
                    verified_count += 1
                    if verified_count > 2:
                        recognized_name = "KNOWN USER"
                else:
                    verified_count = 0
                    recognized_name = "UNKNOWN"

            except:
                pass

            # emotion
            try:
                emo = emotion_model.detect_emotions(face_crop)
                if emo:
                    emotion_text = max(emo[0]["emotions"], key=emo[0]["emotions"].get)
            except:
                emotion_text = "-"
        else:
            recognized_name = "UNKNOWN"
            emotion_text = "-"

        cv2.putText(frame, recognized_name, (x, y+h+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(frame, f"Emotion: {emotion_text}", (x, y+h+45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("MTCNN + ArcFace + FER + AntiSpoof", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
