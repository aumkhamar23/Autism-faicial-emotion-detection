# Backend

## Overview

The backend is a **Python Flask** web server (`app.py`) that serves the HTML pages, handles quiz logic using server-side sessions, and exposes a REST API endpoint for real-time emotion detection.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Web framework | Flask 3.x |
| Templating | Jinja2 (built into Flask) |
| Session storage | Flask server-side session (cookie-based) |
| ML inference | TensorFlow 2.21 + Keras 3 |
| Face detection | OpenCV Haar Cascade |
| Image processing | OpenCV + NumPy |

---

## File Structure

```
app.py                        — Flask app, routes, quiz data
models/
  emotion_detector.py         — EmotionDetector class (loads model, runs inference)
  custom_emotion_model.keras  — trained CNN (loaded at startup)
```

---

## Routes

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/` | Home page — clears session, shows level cards |
| GET | `/quiz/<level>` | Starts a quiz — sets up session state |
| GET | `/quiz` | Shows current question from session |
| POST | `/check_answer` | Validates answer, updates session score |
| GET | `/results` | Shows final score and summary |
| GET | `/camera` | Live camera detection page |
| POST | `/analyze_emotion` | Receives a video frame, returns detected emotion |
| GET | `/reset` | Clears session and redirects home |

---

## Quiz System (Session-Based)

Flask sessions store all quiz state between requests. When a user starts a quiz:

```python
session['level']            = 'basic'     # which level is active
session['current_question'] = 0           # index into questions list
session['score']            = 0           # correct answers so far
session['total_questions']  = 5
session['answers']          = []          # full answer history
```

Each `POST /check_answer` call:
1. Reads `session['current_question']` to find the right question
2. Compares the submitted answer to the correct answer
3. Increments `session['score']` if correct
4. Appends the result to `session['answers']`
5. Increments `session['current_question']`
6. Returns JSON: `{ is_correct, correct_answer, emoji, next_question }`

### Quiz content (hardcoded in `app.py`)

Three levels, each with 5 questions:

```python
QUESTIONS = {
    'basic': {
        'questions': [
            { 'image': 'static/images/happy.jpg', 'emotion': 'Happy',
              'options': ['Happy','Sad','Angry','Surprise','Neutral'] },
            ...
        ]
    },
    'intermediate': { ... },
    'complex': { ... }
}
```

---

## Emotion Detection API

### Endpoint

```
POST /analyze_emotion
Content-Type: application/json

{ "frame": "<base64-encoded JPEG string>" }
```

### Response (success)

```json
{
  "success": true,
  "emotion": "Happy",
  "confidence": 0.87,
  "all_emotions": {
    "angry": 1.2,
    "disgust": 0.4,
    "fear": 0.3,
    "happy": 87.0,
    "neutral": 8.5,
    "sad": 1.8,
    "surprise": 0.8
  }
}
```

### Response (no face detected)

```json
{
  "success": false,
  "error": "No face detected",
  "emotion": "Neutral",
  "confidence": 0.0
}
```

---

## EmotionDetector Class (`models/emotion_detector.py`)

Loaded **once at startup** (`detector = EmotionDetector()` at module level in `app.py`), not per-request. This avoids reloading the ~10 MB model on every camera frame.

### Inference pipeline

```
Base64 string
  → decode → NumPy BGR image
  → resize to 320×240
  → OpenCV Haar Cascade face detection
  → crop largest face region
  → convert to grayscale, resize to 48×48
  → normalize: uint8 [0,255] → float32 [0,1]
  → reshape to (1, 48, 48, 1)
  → model.predict() → raw logits (7 values)
  → tf.nn.softmax() → probabilities [0,1]
  → argmax → emotion label + confidence
```

### Face detection

Uses OpenCV's built-in Haar Cascade classifier (`haarcascade_frontalface_default.xml`). When multiple faces are detected, the **largest face** (by area) is used.

If no face is found, the endpoint returns `success: false` — the camera page silently skips that frame.

---

## NumPy JSON Fix

Flask's default JSON encoder cannot serialize NumPy types (common when working with TF/NumPy arrays). A custom JSON provider is registered:

```python
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.ndarray):   return obj.tolist()
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        return super().default(obj)

app.json = NumpyJSONProvider(app)
```

This prevents `TypeError: Object of type float32 is not JSON serializable` errors.

---

## How to Run

```bash
python app.py
```

Open: [http://localhost:5000](http://localhost:5000)

The app loads the Keras model at startup (~10–15 seconds for TensorFlow to initialize). After that, each camera frame analysis takes ~100–200ms.

---

## Dependencies

```
Flask>=3.0.0
tensorflow>=2.15
keras>=3.0
opencv-python-headless>=4.8
numpy<2.0
Pillow>=10.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Security Notes

- `app.secret_key` in `app.py` is a placeholder — must be changed to a random string before any real deployment
- All video frames are processed locally on the server (never sent to a cloud service)
- Session data is stored in a signed cookie — only the score and question index, no image data
