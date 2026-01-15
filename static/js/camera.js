// Camera and Emotion Detection JavaScript

const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('startCameraBtn');
const stopBtn = document.getElementById('stopCameraBtn');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const emotionOverlay = document.getElementById('emotionOverlay');
const emotionEmoji = document.getElementById('emotionEmoji');
const emotionName = document.getElementById('emotionName');
const emotionConfidence = document.getElementById('emotionConfidence');

let stream = null;
let analysisInterval = null;

const EMOTION_EMOJIS = {
    'Happy': '😊',
    'Sad': '😢',
    'Angry': '😡',
    'Surprise': '😲',
    'Neutral': '😐',
    'Confusion': '🤔',
    'Frustration': '😣',
    'Disgust': '🤢',
    'Fear': '😨',
    'Boredom': '😒',
    'Embarrassment': '😳',
    'Guilt': '😔',
    'Pride': '😏',
    'Envy': '😬',
    'Irritation': '😤'
};

// Start camera
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        });
        
        video.srcObject = stream;
        
        // Hide placeholder and show video
        videoPlaceholder.classList.add('hidden');
        video.classList.remove('hidden');
        stopBtn.classList.remove('hidden');
        emotionOverlay.classList.remove('hidden');
        
        // Start emotion analysis
        startEmotionAnalysis();
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Could not access camera. Please make sure you have granted camera permissions.');
    }
}

// Stop camera
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    if (analysisInterval) {
        clearInterval(analysisInterval);
        analysisInterval = null;
    }
    
    // Reset UI
    video.classList.add('hidden');
    stopBtn.classList.add('hidden');
    emotionOverlay.classList.add('hidden');
    videoPlaceholder.classList.remove('hidden');
}

// Capture frame from video
function captureFrame() {
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert canvas to base64
    return canvas.toDataURL('image/jpeg', 0.8);
}

// Analyze emotion
async function analyzeEmotion() {
    try {
        const frameData = captureFrame();
        
        const response = await fetch('/analyze_emotion', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ frame: frameData })
        });
        
        const data = await response.json();
        
        if (data.success) {
            updateEmotionDisplay(data.emotion, data.confidence);
        } else {
            console.error('Emotion analysis failed:', data.error);
        }
        
    } catch (error) {
        console.error('Error analyzing emotion:', error);
    }
}

// Update emotion display
function updateEmotionDisplay(emotion, confidence) {
    const emoji = EMOTION_EMOJIS[emotion] || '😐';
    emotionEmoji.textContent = emoji;
    emotionName.textContent = emotion;
    emotionConfidence.textContent = `Confidence: ${(confidence * 100).toFixed(0)}%`;
}

// Start continuous emotion analysis
function startEmotionAnalysis() {
    // Analyze immediately
    setTimeout(() => analyzeEmotion(), 1000);
    
    // Then analyze every 2 seconds
    analysisInterval = setInterval(() => {
        analyzeEmotion();
    }, 2000);
}

// Event listeners
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    stopCamera();
});