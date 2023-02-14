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

#make shots directory to save pics
# try:
#     os.mkdir('./shots')
# except OSError as error:
#     pass


# camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)

def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    return frame

def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            # if(face):                
            #     frame= detect_face(frame)
            # if(grey):
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # if(neg):
            #     frame=cv2.bitwise_not(frame)    
            # if(capture):
            #     capture=0
            #     now = datetime.datetime.now()
            #     p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
            #     cv2.imwrite(p, frame)
            
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


if not os.path.isdir("result"):
    os.mkdir("result")


@app.route('/mock',methods=['POST','GET'])
@login_required
def mock():

    global switch,camera
    if request.method == 'POST':
        # if request.form.get('click') == 'Capture':
        #     global capture
        #     capture=1
        # elif  request.form.get('grey') == 'Grey':
        #     global grey
        #     grey=not grey
        # elif  request.form.get('neg') == 'Negative':
        #     global neg
        #     neg=not neg
        # elif  request.form.get('face') == 'Face Only':
        #     global face
        #     face=not face 
        #     if(face):
        #         time.sleep(4)   
        

        if  request.form.get('start') == 'Start Interview':
            global random_questions
            # switch=0
            camera = cv2.VideoCapture(0)
            questions = read_questions_from_csv('interview questions.csv')
            random_questions = select_random_questions(questions, 10)
        
        if request.form.get('stop') == 'Stop Video':
            camera.release()
            cv2.destroyAllWindows()
                
            # switch=1
               
               
        if  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out, frequency, recording, duration, stream
            
            rec= not rec
            
            if(rec):
                now=datetime.datetime.now() 
                # fourcc = cv2.VideoWriter_fourcc(*'XVID')
                # out = cv2.VideoWriter('result/vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter('result/vid_{}.mov'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                print("Codec:", fourcc)
                print("FPS:", out.get(cv2.CAP_PROP_FPS))
                print("Frame Size:", out.get(cv2.CAP_PROP_FRAME_WIDTH), "x", out.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

                # Save the recording in .wav format using scipy
                write("result/record0.wav", frequency, recording)
                
                # Save the recording in .wav format using wavio
                wv.write("result/record1.wav", recording, frequency, sampwidth=2)
        
    elif request.method=='GET':
        return render_template('mock.html')
    
    return render_template('mock.html', questions=random_questions)


@app.route('/submit', methods=['GET', 'POST'])
@login_required
def submit():
    return render_template('report.html')

@app.route('/report', methods=['GET', 'POST'])
@login_required
def report():
    #speech to text
    global s
    r = sr.Recognizer()
    audio = sr.AudioFile('result/record1.wav')
    
    with audio as source:
        audio = r.record(source)
        
    try:
        s = r.recognize_google(audio)
    except Exception as e:
        print("Exception: " + str(e))
        
    with open ('text.txt', 'w') as file:  
        # for s in s:  
        file.write(s) 

    #nlp part
    data = read_csv("words.csv")

    nlp = spacy.load('en_core_web_sm')

    matcher = PhraseMatcher(nlp.vocab)

    phrases = data['negative words'].tolist()

    patterns = [nlp(text) for text in phrases]

    matcher.add('AI', None, *patterns)

    with open('text.txt') as f:
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

    return render_template('report.html',text=s, negative_words=negative_words)


if __name__ == "__main__":
    app.run(debug=True)