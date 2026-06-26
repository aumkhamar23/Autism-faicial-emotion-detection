import cv2
import base64
import numpy as np
from pathlib import Path

import keras
import tensorflow as tf

# Must match training/config.py EMOTIONS (label order baked into the model)
EMOTIONS   = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE   = 48
MODEL_PATH = Path(__file__).parent / 'custom_emotion_model.keras'


class EmotionDetector:

    def __init__(self):
        self._model = keras.models.load_model(str(MODEL_PATH))
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.emotion_map = {
            'angry':    'Angry',
            'disgust':  'Disgust',
            'fear':     'Fear',
            'happy':    'Happy',
            'neutral':  'Neutral',
            'sad':      'Sad',
            'surprise': 'Surprise',
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_face(self, bgr_frame: np.ndarray):
        """Return (48×48) grayscale face crop, or None if no face found."""
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])   # largest face
        return cv2.resize(gray[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE))

    def _predict(self, face_gray: np.ndarray):
        """
        Args:
            face_gray: (48, 48) uint8 numpy array
        Returns:
            emotion_key  : str, e.g. 'happy'
            confidence   : float 0-1
            all_emotions : dict {emotion: score_0_to_100}  — same scale as DeepFace
        """
        img    = face_gray.astype(np.float32) / 255.0
        img    = img[np.newaxis, :, :, np.newaxis]           # (1, 48, 48, 1)
        logits = self._model.predict(img, verbose=0)[0]      # model outputs raw logits
        probs  = tf.nn.softmax(logits).numpy()               # numerically stable softmax

        idx         = int(np.argmax(probs))
        emotion_key = EMOTIONS[idx]
        confidence  = float(probs[idx])
        all_emotions = {em: float(p) * 100.0 for em, p in zip(EMOTIONS, probs)}
        return emotion_key, confidence, all_emotions

    # ------------------------------------------------------------------
    # Public API (same contract as the old DeepFace version)
    # ------------------------------------------------------------------

    def analyze_frame(self, frame_data):
        """Analyze a video frame for emotions.

        Accepts base64-encoded JPEG/PNG string or a raw BGR numpy array.
        Returns a dict with keys: success, emotion, confidence, all_emotions.
        """
        try:
            # ---- decode ------------------------------------------------
            if isinstance(frame_data, str):
                if 'base64,' in frame_data:
                    frame_data = frame_data.split('base64,')[1]
                nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                frame = frame_data

            # Resize for faster face detection
            small = cv2.resize(frame, (320, 240))

            # ---- face detection ----------------------------------------
            face = self._detect_face(small)
            if face is None:
                return {
                    'success': False,
                    'error':   'No face detected',
                    'emotion': 'Neutral',
                    'confidence': 0.0,
                }

            # ---- prediction --------------------------------------------
            emotion_key, confidence, all_emotions = self._predict(face)
            emotion_name = self.emotion_map.get(emotion_key, emotion_key.capitalize())

            return {
                'success':     True,
                'emotion':     emotion_name,
                'confidence':  round(confidence, 2),
                'all_emotions': all_emotions,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success':    False,
                'error':      str(e),
                'emotion':    'Neutral',
                'confidence': 0.0,
            }
