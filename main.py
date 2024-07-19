from flask import Flask, render_template, Response, request, redirect, url_for, flash
import cv2

import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from pytube import YouTube
import os
from tensorflow.python.keras.models import model_from_json


app= Flask(__name__)
app.secret_key = 'supersecretkey'

#from keras.models import load_model

import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

emotion_model = tf.keras.models.load_model("models/emotion_model.h5")

print("Loaded model from disk")

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.resize(frame, (1280, 720))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in num_faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  
            
def process_downloaded_video(video):
    #yt = YouTube(url)
    #stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    #stream.download(DOWNLOAD_FOLDER)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * 10)  # 10-second interval
    data = []

    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    while True:
        frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        timestamp = frame_pos / fps
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + interval)
        success, frame = cap.read()
        if not success:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        if len(num_faces) == 0:
            data.append((timestamp, "No face detected"))
        else:
            for (x, y, w, h) in num_faces:
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                data.append((timestamp, emotion_dict[maxindex]))
                
    emotions = pd.DataFrame(data, columns=['timestamp', 'emotion'])
    emotions.set_index('timestamp', inplace=True)
    cap.release()
    return emotions
       

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['GET','POST'])
def video_feed():
    if request.method== 'POST':
        try:
            return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        except Exception as e:
            flash(str(e))
            return redirect(url_for('/video_feed'))
    return render_template('index.html')

@app.route('/analyze_video', methods=['GET', 'POST'])
def downloaded_video_form():
    if request.method == 'POST':
        video = request.form["mp4_file"]
        try:
            emotions = process_downloaded_video(video)
            print(emotions)
            return render_template('index.html', emotions=emotions)
        except Exception as e:
            flash(str(e))
            return redirect(url_for('downloaded_video_form'))
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)


