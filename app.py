from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import os
from flask.json.provider import DefaultJSONProvider
from models.emotion_detector import EmotionDetector
import numpy as np

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        # Convert numpy arrays to standard Python lists
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Convert numpy integers/floats to standard Python ints/floats
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
            
        # Fall back to the default Flask behavior for everything else
        return super().default(obj)

app.json = NumpyJSONProvider(app)
detector = EmotionDetector()

QUESTIONS = {
    'basic': {
        'name': 'Level 1: Basic Emotions','color': 'green','emoji': '🟢',
        'questions': [
            {'id':1,'image':r'static\images\happy.jpg','emotion':'Happy','options':['Happy','Sad','Angry','Surprise','Neutral']},
            {'id':2,'image':r'static\images\neutral.jpg','emotion':'Neutral','options':['Happy','Sad','Angry','Surprise','Neutral']},
            {'id':3,'image':r'static\images\surprise.jpg','emotion':'Surprise','options':['Happy','Sad','Angry','Surprise','Neutral']},
            {'id':4,'image':r'static\images\sad.jpg','emotion':'Sad','options':['Happy','Sad','Angry','Surprise','Neutral']},
            {'id':5,'image':r'static\images\angry.jpg','emotion':'Angry','options':['Happy','Sad','Angry','Surprise','Neutral']}
        ]
    },
    'intermediate': {
        'name': 'Level 2: Intermediate Emotions','color': 'yellow','emoji': '🟡',
        'questions': [
            {'id':1,'image':r'static\images\confusion.jpg','emotion':'Confusion','options':['Confusion','Frustration','Disgust','Fear','Boredom']},
            {'id':2,'image':r'static\images\boredom.jpg','emotion':'Boredom','options':['Confusion','Frustration','Disgust','Fear','Boredom']},
            {'id':3,'image':r'static\images\fear.jpg','emotion':'Fear','options':['Confusion','Frustration','Disgust','Fear','Boredom']},
            {'id':4,'image':r'static\images\frustration.jpg','emotion':'Frustration','options':['Confusion','Frustration','Disgust','Fear','Boredom']},
            {'id':5,'image':r'static\images\confusion1.jpg','emotion':'Confusion','options':['Confusion','Frustration','Disgust','Fear','Boredom']}
        ]
    },
    'complex': {
        'name': 'Level 3: Complex Emotions','color': 'red','emoji': '🔴',
        'questions': [
            {'id':1,'image':r'static\images\proud.jpg','emotion':'Pride','options':['Embarrassment','Guilt','Pride','Envy','Irritation']},
            {'id':2,'image':r'static\images\Embarrassment.jpg','emotion':'Embarrassment','options':['Embarrassment','Guilt','Pride','Envy','Irritation']},
            {'id':3,'image':r'static\images\irritation.jpg','emotion':'Irritation','options':['Embarrassment','Guilt','Pride','Envy','Irritation']},
            {'id':4,'image':r'static\images\proud.jpg','emotion':'Pride','options':['Embarrassment','Guilt','Pride','Envy','Irritation']},
            {'id':5,'image':r'static\images\guilt.jpg','emotion':'Guilt','options':['Embarrassment','Guilt','Pride','Envy','Irritation']}
        ]
    }
}

EMOTION_EMOJIS = {
    'Happy':'😊','Sad':'😢','Angry':'😡','Surprise':'😲','Neutral':'😐',
    'Confusion':'🤔','Frustration':'😣','Disgust':'🤢','Fear':'😨','Boredom':'😒',
    'Embarrassment':'😳','Guilt':'😔','Pride':'😏','Envy':'😬','Irritation':'😤'
}

@app.route('/')
def home():
    session.clear()
    return render_template('home.html', levels=QUESTIONS)

@app.route('/quiz/<level>')
def start_quiz(level):
    if level not in QUESTIONS:
        return redirect(url_for('home'))
    session['level'] = level
    session['current_question'] = 0
    session['score'] = 0
    session['total_questions'] = len(QUESTIONS[level]['questions'])
    session['answers'] = []
    return redirect(url_for('quiz'))

@app.route('/quiz')
def quiz():
    if 'level' not in session:
        return redirect(url_for('home'))
    level = session['level']
    question_num = session['current_question']
    level_data = QUESTIONS[level]
    if question_num >= len(level_data['questions']):
        return redirect(url_for('results'))
    question = level_data['questions'][question_num]
    return render_template('quiz.html', level=level_data, question=question,
                         question_num=question_num+1, total=session['total_questions'],
                         score=session['score'], emojis=EMOTION_EMOJIS)

@app.route('/check_answer', methods=['POST'])
def check_answer():
    if 'level' not in session:
        return jsonify({'error': 'No active session'}), 400
    selected = request.json.get('answer')
    level = session['level']
    question_num = session['current_question']
    question = QUESTIONS[level]['questions'][question_num]
    correct_answer = question['emotion']
    is_correct = (selected == correct_answer)
    if is_correct:
        session['score'] = session.get('score', 0) + 1
    answers = session.get('answers', [])
    answers.append({'question': question_num+1, 'selected': selected, 'correct': correct_answer, 'is_correct': is_correct})
    session['answers'] = answers
    session['current_question'] = question_num + 1
    return jsonify({'is_correct': is_correct, 'correct_answer': correct_answer,
                    'emoji': EMOTION_EMOJIS.get(correct_answer, ''),
                    'next_question': session['current_question'] < session['total_questions']})

@app.route('/results')
def results():
    if 'level' not in session:
        return redirect(url_for('home'))
    score = session.get('score', 0)
    total = session.get('total_questions', 0)
    percentage = (score / total * 100) if total > 0 else 0
    level_data = QUESTIONS[session['level']]
    if percentage == 100: message = "🎉 Perfect Score! Amazing!"
    elif percentage >= 80: message = "🌟 Great job!"
    elif percentage >= 60: message = "👍 Good effort!"
    else: message = "💪 Keep practicing!"
    return render_template('results.html', score=score, total=total,
                         percentage=percentage, message=message, level=level_data)

@app.route('/camera')
def camera():
    return render_template('camera.html', emojis=EMOTION_EMOJIS)

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    try:
        data = request.json
        frame_data = data.get('frame')
        if not frame_data:
            return jsonify({'error': 'No frame data provided'}), 400
        result = detector.analyze_frame(frame_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'emotion': 'Neutral', 'confidence': 0.0}), 500

@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    print("🚀 Starting Flask app...")
    print("📍 Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
