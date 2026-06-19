# Autism Facial Emotion Detection

An interactive web application that helps individuals with autism practice recognizing and understanding facial expressions through a gamified quiz system and real-time emotion detection via webcam.

---

## Overview

The app combines deep learning-powered facial analysis with a structured learning system across three difficulty levels. Users can both learn to identify emotions from static images and practice detecting their own expressions through live camera feedback.

---

## Features

- **Three-level quiz system** — Basic, Intermediate, and Complex emotions (5 questions each)
- **Live emotion detection** — Real-time webcam feed analyzed every 2 seconds using DeepFace
- **Confidence visualization** — Animated confidence bar and per-emotion probability breakdown
- **Supportive UI** — Emoji feedback, descriptive emotion messages, and colorful visual cues
- **Session-based progress tracking** — Score history and performance summary on completion

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask 3.0.0 |
| Emotion Analysis | DeepFace 0.0.91 |
| Deep Learning | TensorFlow 2.12.0, tf-keras 2.12.0 |
| Computer Vision | OpenCV 4.10.0.84 |
| Image Handling | Pillow 10.0.0, NumPy 1.24.3 |
| Frontend | HTML5, CSS3, Vanilla JavaScript, WebRTC |
| Templating | Jinja2 |

---

## Project Structure

```
Autism-faicial-emotion-detection/
├── app.py                     # Flask app, routes, quiz data, session management
├── requirements.txt           # Python dependencies
├── models/
│   └── emotion_detector.py   # DeepFace wrapper — analyzes base64 frames
├── static/
│   ├── css/style.css         # Global styles, animations, responsive layout
│   ├── js/camera.js          # Camera utility script
│   └── images/               # Reference emotion images (happy, sad, angry, etc.)
└── templates/
    ├── base.html             # Base layout (Poppins font, shared blocks)
    ├── home.html             # Landing page with level selection
    ├── quiz.html             # Quiz question and answer interface
    ├── results.html          # Score summary and performance message
    └── camera.html           # Live webcam emotion detection interface
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- A webcam (for live detection feature)
- pip

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd Autism-faicial-emotion-detection

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
```

The app will be available at `http://localhost:5000`.

> **Note:** The first run will download DeepFace model weights (~600 MB). Subsequent runs use the cached models.

---

## Usage

### Quiz Mode

1. Open `http://localhost:5000`
2. Choose a difficulty level:
   - **Basic** — Common emotions: Happy, Sad, Angry, Surprise, Neutral
   - **Intermediate** — Subtle emotions: Confusion, Frustration, Disgust, Fear, Boredom
   - **Complex** — Nuanced emotions: Embarrassment, Guilt, Pride, Envy, Irritation
3. Select the emotion shown in the image
4. See instant feedback and auto-advance after 2 seconds
5. Review your score on the results page

### Live Camera Mode

1. Click **Live Camera Test** on the home page
2. Allow browser camera access
3. Click **Start Camera**
4. The app analyzes your expression every 2 seconds and displays the detected emotion, confidence level, and a supportive description

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/` | Home page |
| GET | `/quiz/<level>` | Start quiz for a level (basic / intermediate / complex) |
| GET | `/quiz` | Current quiz question |
| POST | `/check_answer` | Validate answer, update session score |
| GET | `/results` | Quiz results page |
| GET | `/camera` | Live detection interface |
| POST | `/analyze_emotion` | Accepts base64 frame, returns detected emotion + confidence |
| GET | `/reset` | Clear session and return home |

---

## Emotions Covered

| Level | Emotions |
|---|---|
| Basic | Happy, Sad, Angry, Surprise, Neutral |
| Intermediate | Confusion, Frustration, Disgust, Fear, Boredom |
| Complex | Embarrassment, Guilt, Pride, Envy, Irritation |

---

## Emotion Detection Pipeline

1. Frontend captures a video frame to a `<canvas>` element every 2 seconds
2. Frame is encoded as base64 and POSTed to `/analyze_emotion`
3. `EmotionDetector.analyze_frame()` decodes the image, resizes to 320x240, and passes it to `DeepFace.analyze()`
4. DeepFace returns dominant emotion and confidence scores
5. Results are sent back as JSON and rendered in the UI

---

## Configuration

Before deploying to production, update the secret key in [app.py](app.py):

```python
app.secret_key = 'your-secret-key-change-this-in-production'
```

Replace it with a strong random value, e.g.:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## Requirements

```
Flask==3.0.0
deepface==0.0.91
opencv-python==4.10.0.84
tensorflow==2.12.0
Pillow==10.0.0
numpy==1.24.3
tf-keras==2.12.0
```

---

## License

This project is intended for educational and assistive technology use.
