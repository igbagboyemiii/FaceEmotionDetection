from flask import Flask, render_template, request
import sqlite3
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from datetime import datetime

# --- Initialize Flask app ---
app = Flask(__name__)

# --- Load trained model ---
model = tf.keras.models.load_model('models/face_emotionModel.h5')

# --- Define emotion labels (order must match your training dataset) ---
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- Ensure uploads folder exists ---
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# --- Function to save user info into database ---
def save_to_db(name, email, image_path, emotion):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('INSERT INTO users (name, email, image_path, emotion) VALUES (?, ?, ?, ?)',
              (name, email, image_path, emotion))
    conn.commit()
    conn.close()


# --- Route: Home Page (Form) ---
@app.route('/')
def index():
    return render_template('index.html')


# --- Route: Handle Form Submission ---
@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    email = request.form['email']
    file = request.files['file']

    # --- Save uploaded image ---
    if file:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        file.save(img_path)

        # --- Preprocess image for model ---
        img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # --- Predict emotion ---
        predictions = model.predict(img_array)
        emotion_index = np.argmax(predictions)
        emotion = emotion_labels[emotion_index]

        # --- Save user info and prediction result into DB ---
        save_to_db(name, email, img_path, emotion)

        return f"<h2>Hello {name}!</h2><p>Your emotion looks like: <b>{emotion}</b>.</p><br><a href='/'>Go Back</a>"

    return "No image uploaded!"


# --- Run the app locally ---
if __name__ == '__main__':
    app.run(debug=True)
