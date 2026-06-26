# Frontend

## Overview

The frontend is built with **HTML, CSS, and vanilla JavaScript** using Flask's Jinja2 templating engine. No frontend framework (React, Vue, etc.) is used — everything is plain HTML with inline or linked CSS/JS.

---

## File Structure

```
templates/
  base.html       — shared layout (head, font imports, CSS link)
  home.html       — landing page with level selection + camera link
  quiz.html       — quiz question page (image + multiple choice)
  results.html    — score summary after quiz completes
  camera.html     — live webcam emotion detection page

static/
  css/style.css   — global styles (home, quiz, results pages)
  js/camera.js    — (unused; camera JS lives inline in camera.html)
  images/         — emotion reference images used in quiz questions
```

---

## Pages

### 1. Home Page (`home.html`)

The landing page shown when the user visits `/`.

**What it shows:**
- Title: "Emotion Learning Journey"
- 3 level cards (Basic, Intermediate, Complex) — each links to `/quiz/<level>`
- A "Live Camera Test" button linking to `/camera`

**Level cards:**

| Level | Color | Emotions covered |
|-------|-------|-----------------|
| Level 1: Basic | Green | Happy, Sad, Angry, Surprise, Neutral |
| Level 2: Intermediate | Yellow | Confusion, Frustration, Disgust, Fear, Boredom |
| Level 3: Complex | Red | Pride, Guilt, Embarrassment, Envy, Irritation |

---

### 2. Quiz Page (`quiz.html`)

Shown at `/quiz` for each question in the active level.

**What it shows:**
- Progress indicator (e.g. "Question 2/5")
- Current score
- An image of a person showing an emotion
- 5 answer buttons with emoji labels

**How it works (JavaScript):**
1. User clicks an answer button → `selectAnswer(option)` is called
2. All buttons are immediately disabled (prevents double-click)
3. A `POST /check_answer` request is sent with `{ answer: selected }`
4. The server responds with `{ is_correct, correct_answer, emoji, next_question }`
5. Correct answer highlighted green, wrong answer highlighted red
6. After 2 seconds → auto-navigate to next question or results page

---

### 3. Results Page (`results.html`)

Shown at `/results` after all 5 questions.

**What it shows:**
- Score (e.g. "4 / 5")
- Percentage
- Motivational message based on score:
  - 100% → "Perfect Score! Amazing!"
  - ≥80% → "Great job!"
  - ≥60% → "Good effort!"
  - <60% → "Keep practicing!"
- Home button

---

### 4. Camera Page (`camera.html`)

The most interactive page — live webcam emotion detection at `/camera`.

**Layout:** Two-column grid (video | emotion display) with a tips row below.

**Camera flow:**

```
User clicks "Start Camera"
  → Browser requests webcam permission (navigator.mediaDevices.getUserMedia)
  → Video stream appears in the left panel
  → Scanning ring animation starts
  → Every 2 seconds: captureFrame() → POST /analyze_emotion → showEmotion()
User clicks "Stop"
  → Camera stream closed, UI resets
```

**`captureFrame()`** — draws the current video frame onto a hidden `<canvas>` and returns it as a base64 JPEG string (quality 0.8).

**`showEmotion(emotion, confidence)`** — updates the right panel:
- Swaps the large emoji with a pop-in animation
- Shows emotion name and a short description
- Fills a confidence progress bar (0–100%)
- Changes the card's color theme to match the emotion

**Emotion color themes:**

| Emotion | Color | Background |
|---------|-------|-----------|
| Happy | Green (#16a34a) | Light green |
| Sad | Blue (#3b82f6) | Light blue |
| Angry | Red (#dc2626) | Light red |
| Surprise | Amber (#f59e0b) | Light yellow |
| Neutral | Gray (#6b7280) | Light gray |
| Disgust | Olive (#65a30d) | Light green |
| Fear | Purple (#7c3aed) | Light purple |

**Privacy note shown to user:** "Your video stays on your device" — the frame is sent to the local Flask server (localhost), never to a cloud service.

---

## Base Template (`base.html`)

All pages extend `base.html`. It provides:
- `<meta charset="UTF-8">` and viewport tag
- Poppins font from Google Fonts
- Link to `static/css/style.css`
- Jinja2 blocks: `title`, `extra_css`, `content`, `extra_js`

---

## Styling Approach

- **Global styles** (`style.css`): home page cards, quiz layout, results page, buttons
- **Camera page styles**: defined inline in `camera.html` using `{% block extra_css %}` — keeps camera-specific styles isolated
- **Fonts**: Poppins (global) and Nunito (camera page)
- **Responsive**: camera page switches from 2-column to 1-column on screens ≤640px (`@media` query)

---

## Emoji Map

The app maps emotion names to emojis for display across all pages:

```python
EMOTION_EMOJIS = {
    'Happy': '😊',  'Sad': '😢',     'Angry': '😡',
    'Surprise': '😲', 'Neutral': '😐', 'Confusion': '🤔',
    'Frustration': '😣', 'Disgust': '🤢', 'Fear': '😨',
    'Boredom': '😒', 'Embarrassment': '😳', 'Guilt': '😔',
    'Pride': '😏',  'Envy': '😬',    'Irritation': '😤'
}
```

This is passed from Flask to every Jinja2 template via the `emojis` variable.

---

## Key Technical Notes

- No page reloads during a quiz — answer checking is done via `fetch()` (AJAX)
- Camera analysis runs on a `setInterval` timer (every 2000ms)
- A 1200ms delay before the first analysis gives the camera time to warm up
- The `beforeunload` event stops the camera stream automatically when the user navigates away
