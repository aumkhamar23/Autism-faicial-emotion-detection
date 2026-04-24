import cv2
import base64
import numpy as np
from deepface import DeepFace


class EmotionDetector:

    def __init__(self):
        # Maps DeepFace labels → display names used in the quiz & camera UI
        self.emotion_map = {
            'angry':   'Angry',
            'disgust': 'Disgust',
            'fear':    'Fear',
            'happy':   'Happy',
            'sad':     'Sad',
            'surprise':'Surprise',
            'neutral': 'Neutral'
        }

    def analyze_frame(self, frame_data):
        """Analyze a video frame for emotions."""
        try:
            # Decode base64 image if needed
            if isinstance(frame_data, str):
                if 'base64,' in frame_data:
                    frame_data = frame_data.split('base64,')[1]
                img_bytes = base64.b64decode(frame_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                frame = frame_data

            # Resize for faster processing
            small_frame = cv2.resize(frame, (320, 240))

            # Run DeepFace emotion detection
            result = DeepFace.analyze(
                small_frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )

            dominant_emotion = result[0]['dominant_emotion']
            all_emotions = result[0]['emotion']

            # Convert numpy float32 → plain Python float (safe for JSON)
            all_emotions_clean = {k: float(v) for k, v in all_emotions.items()}
            confidence = all_emotions_clean[dominant_emotion] / 100.0
            emotion_name = self.emotion_map.get(dominant_emotion, dominant_emotion.capitalize())

            return {
                'success': True,
                'emotion': emotion_name,
                'confidence': round(confidence, 2),
                'all_emotions': all_emotions_clean
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'emotion': 'Neutral',
                'confidence': 0.0
            }
