from flask import Flask , render_template, url_for, redirect,  Response, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import csv 
import random

import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread


import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import speech_recognition as sr

import spacy
#import the phrase matcher
from spacy.matcher import PhraseMatcher
from pandas import *

import pandas as pd
from fer import FER
from fer import Video


app = Flask(__name__)
bcrypt = Bcrypt(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.config['SECRET_KEY'] = 'thisisasecretkey'

app.app_context().push()

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')
    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')

class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('homeafterlogin.html')


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data) #it is used to store password in hashed form
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)


@app.route('/instruction', methods=['GET', 'POST'])
@login_required
def instruction():
    return render_template('instructions.html')


#reading csv file and getting 10 random questions
def read_questions_from_csv(filename):
    questions = []
    with open('interview questions.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            questions.append(row[0])
        return questions

def select_random_questions(questions, count):
    return random.sample(questions, count)

# @app.route('/mock', methods=['GET', 'POST'])
# @login_required
# def mock():
#     questions = read_questions_from_csv('interview questions.csv')
#     random_questions = select_random_questions(questions, 10)
#     return render_template('mock.html', questions=random_questions)


#video screen recording code
global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
face=0
switch=1
rec=0


# camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)


def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame=cv2.flip(frame,1)
            
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if not os.path.isdir("audio_video_files"):
    os.mkdir("audio_video_files")

if not os.path.isdir("result"):
    os.mkdir("result")


questions = read_questions_from_csv('interview_questions.csv')
random_questions = select_random_questions(questions, 3)


@app.route('/question/<int:question_num>', methods=['GET', 'POST'])
def question(question_num):
    # if question_num > len(random_questions):
    #     return redirect(url_for('submit'))
    
    question = random_questions[question_num - 1]
    global switch,camera
    if request.method == 'POST':
        if  request.form.get('start') == 'Start Interview':
            # switch=0
            camera = cv2.VideoCapture(0)
        
        if request.form.get('stop') == 'Stop Video':
            camera.release()
            cv2.destroyAllWindows()
                
            # switch=1
               
               
        if  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out, frequency, recording, duration, stream, video_path, audio_path
            
            rec= not rec
            
            if(rec):
                now=datetime.datetime.now() 
                video_filename = 'vid_q{}.avi'.format(question_num)
                video_path = os.path.join('audio_video_files', video_filename)
                frames_per_second = 24.0
                res = '720p'

                # Set resolution for the video capture
                # Function adapted from https://kirr.co/0l6qmh
                def change_res(cap, width, height):
                    cap.set(3, width)
                    cap.set(4, height)

                # Standard Video Dimensions Sizes
                STD_DIMENSIONS =  {
                    "480p": (640, 480),
                    "720p": (1280, 720),
                    "1080p": (1920, 1080),
                    "4k": (3840, 2160),
                }

                # grab resolution dimensions and set video capture to it.
                def get_dims(cap, res='1080p'):
                    width, height = STD_DIMENSIONS["480p"]
                    if res in STD_DIMENSIONS:
                        width,height = STD_DIMENSIONS[res]
                    ## change the current caputre device
                    ## to the resulting resolution
                    change_res(cap, width, height)
                    return width, height

                # Video Encoding, might require additional installs
                # Types of Codes: http://www.fourcc.org/codecs.php
                VIDEO_TYPE = {
                    'avi': cv2.VideoWriter_fourcc(*'XVID'),
                    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
                    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
                }

                def get_video_type(filename):
                    filename, ext = os.path.splitext(filename)
                    if ext in VIDEO_TYPE:
                        return  VIDEO_TYPE[ext]
                        return VIDEO_TYPE['mp4']


                cap = cv2.VideoCapture(0)
                out = cv2.VideoWriter(video_path, get_video_type(video_path), 25, get_dims(cap, res))
                # fourcc = cv2.VideoWriter_fourcc(*'avc1')
                # out = cv2.VideoWriter('result/vid_{}.mov'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()


                #audio recording 

                # Sampling frequency
                frequency = 44400
            
                # Start recording audio
                # recording = sd.rec(int(frequency*60),samplerate=frequency, channels=1)
                recording = []
                while rec:
                    recording.extend(sd.rec(int(frequency*1),samplerate=frequency, channels=1))
                    sd.wait()
                recording = np.concatenate(recording)
                
            elif(rec==False):
                out.release()

                # Stop recording audio
                sd.stop()

                recording_array = np.array(recording)
                audio_filename = f"question{question_num}_audio.wav"
                audio_path = os.path.join('audio_video_files', audio_filename)
                # Save the recording in .wav format using scipy
                write(f"audio_video_files/record0{question_num}.wav", frequency, recording_array)
                
                # Save the recording in .wav format using wavio
                wv.write(audio_path, recording, frequency, sampwidth=2)

        # return redirect(url_for('question', question_num=question_num+1))
    
    return render_template('mock.html', question=question, question_num=question_num,random_questions=random_questions)


@app.route('/submit', methods=['GET', 'POST'])
@login_required
def submit():
    return render_template('report.html')

@app.route('/report', methods=['GET', 'POST'])
@login_required
def report():
    results = []
    
    # for question_num in range(1, len(random_questions) + 1):
    for i, question in enumerate(random_questions, start=1):
        #speech to text
        global s
        now=datetime.datetime.now() 
        video_filename = 'vid_q{}.avi'.format(i)
        video_path = os.path.join('audio_video_files', video_filename)
        audio_filename = f"question{i}_audio.wav"
        audio_path = os.path.join('audio_video_files', audio_filename)


        r = sr.Recognizer()
        audio = sr.AudioFile(audio_path)
        
        with audio as source:
            audio = r.record(source)
            
        try:
            s = r.recognize_google(audio)
        except Exception as e:
            print("Exception: " + str(e))
            
        with open (f'result/text{i}.txt', 'w') as file:  
            # for s in s:  
            file.write(s) 

        #nlp part
        data = read_csv("words.csv")

        nlp = spacy.load('en_core_web_sm')

        matcher = PhraseMatcher(nlp.vocab)

        phrases = data['negative words'].tolist()

        patterns = [nlp(text) for text in phrases]

        matcher.add('AI', None, *patterns)

        with open(f'result/text{i}.txt') as f:
            contents = f.read().rstrip()

        filtered_sentence = contents.lower()

        sentence = nlp(filtered_sentence)
        matches = matcher(sentence)
        negative_words = []
        for match_id, start, end in matches:
            span = sentence[start:end]
            negative_words.append(span.text)

        # Join the negative words into a single string separated by a comma
        negative_words = ", ".join(negative_words)
        
        # negative_words = [span.text for match_id, start, end in matches for span in sentence[start:end]]

        #fer part
        location_videofile = (video_path)

        face_detector = FER(mtcnn=True)
        input_video = Video(location_videofile)

        processing_data = input_video.analyze(face_detector, display=False)

        vid_df = input_video.to_pandas(processing_data)
        vid_df = input_video.get_first_face(vid_df)
        vid_df = input_video.get_emotions(vid_df)

        # pltfig = vid_df.plot(figsize=(20, 8), fontsize=16).get_figure()
        # pltfig.savefig('data.png')

        angry = sum(vid_df.angry)
        disgust = sum(vid_df.disgust)
        fear = sum(vid_df.fear)
        happy = sum(vid_df.happy)
        sad = sum(vid_df.sad)
        surprise = sum(vid_df.surprise)
        neutral = sum(vid_df.neutral)

        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        emotions_values = [angry, disgust, fear, happy, sad, surprise, neutral]


        score_comparisons = pd.DataFrame(emotions, columns = ['Human Emotions'])
        score_comparisons['Emotion Value from the Video'] = emotions_values
        score_comparisons.to_csv(f'result/question{i}_scores.csv', index=False)
        results.append((f"Question {i}: {question}", s, negative_words, emotions, emotions_values))

    return render_template('report.html', results=results)
#   return render_template('report.html',text=s, negative_words=negative_words,emotions=emotions, emotions_values=emotions_values)


if __name__ == "__main__":
    app.run(debug=True)