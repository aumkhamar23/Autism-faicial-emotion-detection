import cv2
import numpy as np
from deepface import DeepFace
import base64

class EmotionDetector:
    """Wrapper class for DeepFace emotion detection"""
    
    def __init__(self):
        self.emotion_map = {
            "angry": "Angry",
            "disgust": "Disgust",
            "fear": "Fear",
            "happy": "Happy",
            "sad": "Sad",
            "surprise": "Surprise",
            "neutral": "Neutral"
        }
    
    def analyze_frame(self, frame_data):
        """
        Analyze a video frame for emotions
        
        Args:
            frame_data: base64 encoded image string or numpy array
            
        Returns:
            dict: {
                'emotion': str,
                'confidence': float,
                'all_emotions': dict
            }
        """
        try:
            # If base64 string, decode it
            if isinstance(frame_data, str):
                # Remove header if present
                if 'base64,' in frame_data:
                    frame_data = frame_data.split('base64,')[1]
                
                # Decode base64 to image
                img_bytes = base64.b64decode(frame_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                frame = frame_data
            
            # Resize for faster processing
            small_frame = cv2.resize(frame, (320, 240))
            
            # Analyze with DeepFace
            result = DeepFace.analyze(
                small_frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            # Extract emotion data
            dominant_emotion = result[0]['dominant_emotion']
            all_emotions = result[0]['emotion']
            confidence = all_emotions[dominant_emotion] / 100.0
            
            # Map to our emotion names
            emotion_name = self.emotion_map.get(dominant_emotion, dominant_emotion.capitalize())
            
            return {
                'success': True,
                'emotion': emotion_name,
                'confidence': round(confidence, 2),
                'all_emotions': all_emotions
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'emotion': 'Neutral',
                'confidence': 0.0
            }
    
    def analyze_image_path(self, image_path):
        """Analyze an image file for emotions"""
        try:
            img = cv2.imread(image_path)
            return self.analyze_frame(img)
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }