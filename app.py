from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from flask import Flask, render_template, request, jsonify


app = Flask(__name__, static_folder='static')

@app.route('/')
def emotion():
    return render_template('main.html')

# Load the trained emotion model
model_path = 'models\emotion_model.h5'
emotion_model = load_model(model_path)

# Emotion labels in the same order as your dataset
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Add this feedback dictionary
emotion_feedback = {
    'Angry': 'Take a few deep breaths and try to relax.',
    'Disgust': 'Remember to focus on the positives around you.',
    'Fear': 'Stay calm; face your fears one step at a time.',
    'Happy': 'Keep smiling! Happiness is contagious.',
    'Neutral': 'All set; stay balanced and steady.',
    'Sad': 'Consider talking to a friend or doing something you enjoy.',
    'Surprise': 'Embrace the unexpected; life is full of surprises!'
}


@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # Save and preprocess the uploaded file
    file_path = 'temp.jpg'
    file.save(file_path)

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1) / 255.0

    # Predict emotion
    predictions = emotion_model.predict(img)
    emotion_label = emotion_labels[np.argmax(predictions)]
    feedback_message = emotion_feedback[emotion_label]
    os.remove(file_path)  # Clean up temporary file
    return jsonify({
        'emotion': emotion_label,
        'feedback': feedback_message
    })

if __name__ == '__main__':
    app.run(debug=True)