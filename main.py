import numpy as np
import pandas as pd
import warnings
import time
from string import punctuation
from datetime import datetime
import secrets
from PIL import Image
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer

# Importing summarization modules
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from scipy import spatial
import networkx as nx

# Flask packages
from flask import Flask, render_template, request, jsonify, url_for, flash, redirect, Response
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, TextField, SubmitField, TextAreaField, PasswordField, BooleanField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError
from flask_mail import Mail, Message
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from flask_login import login_user, current_user, logout_user, login_required, LoginManager, UserMixin

# General modules
import newspaper
import json
import email_validator
import os
from waitress import serve
from newspaper import fulltext

# Recommendation modules
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors
from sklearn.preprocessing import OneHotEncoder

# Below libraries are for feature representation using sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Below libraries are for similarity matrices using sklearn
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel  
from sklearn.metrics import pairwise_distances
import requests

nltk.download('punkt')
nltk.download('stopwords')
stop_words = stopwords.words('english')

#url="%s" % (url)
def scrape_articles(url):
    url_i = newspaper.Article(url="%s" % (url), language='en')
    url_i.download()
    url_i.parse()
    return url_i.text

def scrape_authors(url):
    url_i = newspaper.Article(url="%s" % (url), language="en")
    url_i.download()
    url_i.parse()
    return url_i.authors

def scrape_publishdate(url):
    url_i = newspaper.Article(url="%s" % (url), language="en")
    url_i.download()
    url_i.parse()
    return url_i.publish_date

def scrape_keywords(url):
    url_i = newspaper.Article(url="%s" % (url), language="en")
    url_i.download()
    url_i.parse()
    return url_i.keywords

def scrape_title(url):
    url_i = newspaper.Article(url="%s" % (url), language="en")
    url_i.download()
    url_i.parse()
    return url_i.title

def scrape_image(url):
    url_i = newspaper.Article(url="%s" % (url), language="en")
    url_i.download()
    url_i.parse()
    return url_i.top_image

df = pd.read_csv('recommendation/output/processed_news_articles.csv')
df = df[pd.isna(df["headline"])==False]
df = df[pd.isna(df["short_description"])==False]

# Accessing word relevancy via term frequency-inverse document frequency
description = df["short_description"]
vector = TfidfVectorizer(max_df=0.3, stop_words="english", lowercase=True, use_idf=True,
    	                norm=u'l2', smooth_idf=True)
tfidf = vector.fit_transform(description)

def search(tfidf_matrix, model, request, top_n=2):
    request_transfrom = model.transform([request])
    similarity = np.dot(request_transfrom, np.transpose(tfidf_matrix))
    x = np.array(similarity.toarray()[0])
    indices = np.argsort(x)[-2:][::-1]
    return indices

def find_similar(tfidf_matrix, index, top_n=2):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return[index for index in related_docs_indices][0:top_n]

def print_result(request_content, indices, X):
    print('\nsearch: ' + request_content)
    print('\nBest Results: ')
    for i in indices:
        yield X['link'].loc[i]

app = Flask(__name__)

# Configurations for mailing service
app.config['SECRET_KEY'] = "0c8973c8a5e001bb0c816a7b56c84f3a"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 465
app.config["MAIL_USE_SSL"] = True
app.config['MAIL_USE_TLS'] = False
app.config["MAIL_USERNAME"] = "EnigmaText.service@gmail.com"
app.config["MAIL_PASSWORD"] = "BuckForBang123"
ADMINS = ['andreliu2004@gmail.com']

db = SQLAlchemy(app)
mail = Mail(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'
mail.init_app(app)
# Dealing with browser caching
response = Response()

class ContactForm(FlaskForm):
    name = StringField("Name", [DataRequired(), Length(max=15, min=2)])
    email = StringField("Email", [DataRequired(), Email(message=('Not a valid email address')), Length(max=30, min=2)])
    subject = StringField("Subject", [DataRequired(), Length(max=30, min=2)])
    message = TextAreaField("Message", [DataRequired(), Length(min=4, message=('Your message is too short.'), max=400)])
    submit = SubmitField("Send")

# Registration form
class RegistrationForm(FlaskForm):
    username = StringField('Username', [DataRequired(), Length(max=15, min=2)])
    email = StringField('Email', [DataRequired(), Email(message=('Not a valid email address')), Length(max=50)])
    password = PasswordField('Password', [DataRequired()])
    confirm_password = PasswordField('Password', [DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

    # If username already exist, raise error
    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('That username is taken. Please choose a different one.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('That email is taken. Please choose a different one.')

# Login form
class LoginForm(FlaskForm):
    email = StringField('Email', [DataRequired(), Email(message=('Not a valid email address')), Length(max=50)])
    password = PasswordField('Password', [DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

class UpdateAccountForm(FlaskForm):
    username = StringField('Username', [DataRequired(), Length(max=15, min=2)])
    email = StringField('Email', [DataRequired(), Email(message=('Not a valid email address')), Length(max=50)])
    picture = FileField('Update Profile Pic', validators=[FileAllowed(['jpg','png', 'jpeg'])])
    submit = SubmitField('Update')

    # If username already exist, raise error
    def validate_username(self, username):
        if username.data !=current_user.username:
            user = User.query.filter_by(username=username.data).first()
            if user:
                raise ValidationError('That username is taken. Please choose a different one.')

    def validate_email(self, email):
        if email.data != current_user.email:
            user = User.query.filter_by(email=email.data).first()
            if user:
                raise ValidationError('That email is taken. Please choose a different one.')

class RequestResetForm(FlaskForm):
    email = StringField('Email', [DataRequired(), Email(message=('Not a valid email address')), Length(max=50)])
    submit = SubmitField('Request Password Reset')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is None:
            raise ValidationError('There is no account with that email. You must register first.')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('Password', [DataRequired()])
    confirm_password = PasswordField('Password', [DataRequired(), EqualTo('password')])
    submit = SubmitField('Reset Password')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    image_file = db.Column(db.String(20), nullable=False, default='user.png')
    password = db.Column(db.String(60), nullable=False)
    # relationship between post and user
    posts = db.relationship('Post', backref='author', lazy=True)

    def get_reset_token(self, expires_sec=1800):
        s = Serializer(app.config['SECRET_KEY'], expires_sec)
        return s.dumps({'user_id': self.id}).decode('utf-8')

    @staticmethod
    def verify_reset_token(token):
        s = Serializer(app.config['SECRET_KEY'])
        try:
            user_id = s.loads(token)['user_id']
        except:
            return None
        return User.query.get(user_id)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}', '{self.image_file}')"

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(200), nullable=False)
    time_taken = db.Column(db.Float, nullable=False)
    content = db.Column(db.String, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"User('{self.url}', '{self.time_taken}', '{self.user_id}')"

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash(f'Your account has been created! You are now able to log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static',  'profile_pics', picture_fn)

    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)
    return picture_fn

def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message('Password Reset Request', sender='EnigmaText.service@gmail.com', recipients=[user.email])
    msg.body = f'''To reset your password, visit the following link:
{url_for('reset_token', token=token, _external=True)}
If you did not make this request then simply ignore this email and no changes will be made.
'''
    mail.send(msg)

# Route to put in email
@app.route('/reset_password', methods=['GET', 'POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        send_reset_email(user)
        flash('An email has been sent with instructions to reset your password.', 'info')
        return redirect(url_for('login'))
    return render_template('reset_request.html', form=form)

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token', 'warning')
        return redirect(url_for('reset_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        flash(f'Your password has been updated! You are now able to log in.', 'success')
        return redirect(url_for('login'))
    return render_template('reset_token.html', form=form)

@app.route('/account', methods=['GET', "POST"])
@login_required
def account():
    form = UpdateAccountForm()
    picture_file = None
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        picture_file= current_user.image_file 
        db.session.commit()
        flash('your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('account.html', title='Account', image_file=image_file, form=form)

@app.route('/results')
def results():
    return render_template("results.html")

@app.errorhandler(404)
def error_404(error):
    return render_template('404.html'), 404

@app.errorhandler(403)
def error_403(error):
    return render_template('404.html'), 403

@app.errorhandler(500)
def error_500(error):
    return render_template('404.html'), 500

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/summarize-text')
def inprogress():
    return render_template('summarize_text.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
  form = ContactForm()
 
  if request.method == 'POST':
    if form.validate() == False:
      flash('All fields are required.')
      return render_template('contact_us.html', form=form)
    else:
      msg = Message(form.subject.data, sender=form.email.data, recipients=['minghong.liu@acsinternational.edu.sg'])
      msg.body = """
      From: %s &lt;%s&gt;
      %s
      """ % (form.name.data, form.email.data, form.message.data)
      mail.send(msg)
 
      return render_template('contact_us.html', success=True)
 
  elif request.method == 'GET':
    return render_template('contact_us.html', form=form)

def textrank(text):
    sentences = sent_tokenize(text)
    sentences_clean = [re.sub(r'[^\w\s]','',sentence.lower()) for sentence in sentences]
    sentence_tokens=[[words for words in sentence.split(' ') if words not in stop_words] for sentence in sentences_clean]

    # Computing word embedding
    w2v = Word2Vec(sentence_tokens, vector_size=1, min_count=1, epochs=1000)
    sentence_embeddings=[[w2v.wv[word][0] for word in words] for words in sentence_tokens]
    max_len=max([len(tokens) for tokens in sentence_tokens])
    sentence_embeddings=[np.pad(embedding,(0,max_len-len(embedding)),'constant') for embedding in sentence_embeddings]

    # Creating Markov Similarity Matrix
    similarity_matrix = np.zeros([len(sentence_tokens), len(sentence_tokens)])
    for i,row_embedding in enumerate(sentence_embeddings):
        for j,column_embedding in enumerate(sentence_embeddings):
            similarity_matrix[i][j] = 1-spatial.distance.cosine(row_embedding,column_embedding)

    # Pagerank Implementation
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    top_sentence={sentence:scores[index] for index,sentence in enumerate(sentences)}
    top=dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:10])
        
    s_list = []
    for sent in sentences:
        if sent in top.keys():
            s_list.append(sent)
    _summary = " ".join(s_list)
    return _summary

# Reading time function
def readingTime(text):
    words = word_tokenize(text)
    total_words = len([word for word in words if word not in punctuation])
    estimated_time = total_words/200.0
    return estimated_time

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    start = time.time()
    if request.method == "POST":
        rawtext = request.form.get('rawtext')
        # Summarization taking place
        _summary = textrank(rawtext)
        # Final reading time
        final_readingTime = readingTime(rawtext)
        summary_reading_time = readingTime(_summary)
        end = time.time()
        final_time = end - start
        return render_template('results.html', summary=_summary, final_time=final_time, final_reading_time=final_readingTime, summary_reading_time = summary_reading_time)
    else:
        return render_template("summarize_text.html")

@app.route('/', methods=['GET', 'POST'])
def index():
    posts=Post.query.all()
    return render_template('index.html')

@app.route('/analyze_url', methods=['GET', 'POST'])
@login_required
def analyze_url():
    start = time.time()
    if request.method == 'POST':
        url = request.form.get("url")
        text = scrape_articles(url)
        publish_date = scrape_publishdate(url)
        authors = scrape_authors(url)
        title = scrape_title(url)
        article = title.lower()
        result = search(tfidf, vector, article, top_n=2)
        recommended_data = print_result(title, result, df)
        msg = Message('Here are some recommended articles to summarize!', sender="EnigmaText.service@gmail.com", recipients=[current_user.email])
        for i in recommended_data:
            msg.html = render_template('email.html', link=i, img_url=scrape_image(i), title=scrape_title(i))
            mail.send(msg)
            print(i)
        # Summarization taking place
        _summary = textrank(text)
        # Final reading time
        final_readingTime = readingTime(text)
        summary_reading_time = readingTime(_summary)
        end = time.time()
        final_time = end - start
        post = Post(url=url, content=text, time_taken=final_readingTime, author=current_user)
        db.session.add(post)
        db.session.commit()
        return render_template('results.html', summary=_summary, final_time=final_time, final_reading_time=final_readingTime, summary_reading_time = summary_reading_time)

if __name__ == '__main__':
    app.run(debug=True)
