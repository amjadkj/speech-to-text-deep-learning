from flask import Flask,render_template, request, jsonify
import pickle 
import numpy as np
from pydub import AudioSegment
import os
import speech_recognition as sr
from hmmlearn import hmm


app = Flask(__name__)

def process_audio(audio_file):
    r=sr.Recognizer()
    if audio_file:
        with sr.AudioFile(audio_file) as source:
            audiodata=r.record(source)
            recognized_text=r.recognize_google(audiodata)
    return recognized_text
def proceess_audio(audio_path):
    model = pickle.load(open("hmm_model2.pkl", "rb"))       
    audio = AudioSegment.from_wav(audio_path)
    samples = np.array(audio.get_array_of_samples())
    train_features, train_labels = extract_features(model, audio)

    hmm_model = hmm.GaussianHMM(n_components=10, covariance_type="diag", n_iter=100)
    hmm_model.fit(train_features.reshape(-1, train_features.shape[-1]))
    features = np.mean(train_features)  # Example: Using mean as a simple feature
    recognizod_text = model.predict(features.reshape(-1, 1))
    return recognizod_text
def extract_features(model, dataset):
    features = []
    labels = []
    for batch in dataset:
        X, y = batch
        batch_predictions = model.predict(X)
        features.extend(batch_predictions)
        labels.extend(tf.keras.backend.eval(y).tolist())  # Convert to Python list
    return np.vstack(features), np.concatenate(labels)
    
@app.route("/",methods=["GET","POST"])
def index():
    return render_template("ui.html")

@app.route('/convert', methods=['POST'])
def convert():

    if 'selected_audio' in request.files:
        selected_audio = request.files['selected_audio']
        upload_folder = 'uploads'
        os.makedirs(upload_folder, exist_ok=True)
        audio_path = os.path.join(upload_folder, 'uploaded_audio.wav')
            
        audio_text=process_audio(selected_audio)
        return jsonify({'text': audio_text})

@app.route('/record', methods=['POST'])
def record_audio():
    if 'recorded_audio' in request.files:
        recorded_audio = request.files['recorded_audio']
        upload_folder = 'uploads'
        os.makedirs(upload_folder, exist_ok=True)
        audio_path = os.path.join(upload_folder, 'recorded_audio.wav')
        recorded_audio.save(audio_path)
        audio_text = process_audio(audio_path)
        return jsonify({'text': audio_text})
    return jsonify({'text': 'No recorded audio received'})
    
    
if __name__ == "__main__":
    app.run(debug=True, threaded=True) 
