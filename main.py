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
# from flask_sitemap import Sitemap

# General modules
import newspaper
import json
import email_validator
import os
from waitress import serve
from newspaper import fulltext
from werkzeug.utils import secure_filename

# Recommendation modules
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors
from sklearn.preprocessing import OneHotEncoder

# Libraries below  are for feature representation using sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Libraries below are for similarity matrices using sklearn
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel  
from sklearn.metrics import pairwise_distances
import requests

# Libraries below are for mining PDF documents to be summarized later
from io import StringIO, BytesIO
import urllib.request

import pdfminer
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage

# python-docx
from docx import Document
from docx.shared import Pt

# Libraries for T5-Inference
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

nltk.download('stopwords')
nltk.download('punkt')
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

app = Flask(__name__)

# Configurations for mailing service
app.config['SECRET_KEY'] = '0c8973c8a5e001bb0c816a7b56c84f3a'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://cmnkmsgxohfmjp:c812819ca44407ae91fb103789b83ebc1f2931e27d28338e781d0ca244a90a93@ec2-52-86-25-51.compute-1.amazonaws.com:5432/d9i7pjrtae42h8'
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 465
app.config["MAIL_USE_SSL"] = True
app.config['MAIL_USE_TLS'] = False
app.config["MAIL_USERNAME"] = "EnigmaText.service@gmail.com"
app.config["MAIL_PASSWORD"] = "vtxrcldpmcwxagjz"
app.config['UPLOAD_EXTENSIONS'] = ['.doc', '.docx']
app.config['UPLOAD_FOLDER'] = "./static/uploads/"
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
# Generating sitemap
# ext = Sitemap(app=app)
@app.route("/sitemap")
@app.route("/sitemap/")
@app.route("/sitemap.xml")
def sitemap():
    """
        Route to dynamically generate a sitemap of your website/application.
        lastmod and priority tags omitted on static pages.
        lastmod included on dynamic content such as blog posts.
    """
    from flask import make_response, request, render_template
    import datetime
    from urllib.parse import urlparse

    host_components = urlparse(request.host_url)
    host_base = host_components.scheme + "://" + host_components.netloc

    # Static routes with static content
    static_urls = list()
    for rule in app.url_map.iter_rules():
        if not str(rule).startswith("/admin") and not str(rule).startswith("/user"):
            if "GET" in rule.methods and len(rule.arguments) == 0:
                url = {
                    "loc": f"{host_base}{str(rule)}"
                }
                static_urls.append(url)

    xml_sitemap = render_template("public/sitemap.xml", static_urls=static_urls, host_base=host_base)
    response = make_response(xml_sitemap)
    response.headers["Content-Type"] = "application/xml"

    return response

class ContactForm(FlaskForm):
    name = StringField("Name", [DataRequired(), Length(max=15, min=2)], render_kw={"placeholder": "Enter your name"})
    email = StringField("Email", [DataRequired(), Email(message=('Not a valid email address')), Length(max=30, min=2)], render_kw={"placeholder": "Enter your email"})
    subject = StringField("Subject", [DataRequired(), Length(max=30, min=2)], render_kw={"placeholder": "Enter the subject of your email"})
    message = TextAreaField("Message", [DataRequired(), Length(min=4, message=('Your message is too short.'), max=400)], render_kw={"placeholder": "Enter your message"})
    submit = SubmitField("Send")

# Registration form
class RegistrationForm(FlaskForm):
    username = StringField('Username', [DataRequired(), Length(max=15, min=2)], render_kw={"placeholder": "Enter your username"})
    email = StringField('Email', [DataRequired(), Email(message=('Not a valid email address')), Length(max=50)], render_kw={"placeholder": "Enter your email"})
    password = PasswordField('Password', [DataRequired()], render_kw={"placeholder": "Enter your password"})
    confirm_password = PasswordField('Password', [DataRequired(), EqualTo('password')], render_kw={"placeholder": "Confirm your password"})
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
    email = StringField('Email', [DataRequired(), Email(message=('Not a valid email address')), Length(max=50)], render_kw={"placeholder": "Enter your email"})
    password = PasswordField('Password', [DataRequired()], render_kw={"placeholder": "Enter your password"})
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
    email = StringField('Email', [DataRequired(), Email(message=('Not a valid email address')), Length(max=50)], render_kw={"placeholder": "Enter your email and we will get back to you shortly"})
    submit = SubmitField('Request Password Reset')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is None:
            raise ValidationError('There is no account with that email. You must register first.')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('Password', [DataRequired()], render_kw={"placeholder": "Enter your password"})
    confirm_password = PasswordField('Confirm Password', [DataRequired(), EqualTo('password')], render_kw={"placeholder": "Confirm your password"})
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

def count_docx(filename):
    document = Document(filename)
    text = ""
    list_lens = []
    for paragraph in document.paragraphs:
        for run in paragraph.runs:
            if ((run.font.size == Pt(12)) and (run.font.name == "Times New Roman") and not (run.bold)):
                text = text + run.text + ""
                words = list(filter(None, text.split(' ')))
                list_lens.append(len(words))
                length = list_lens[-1]
                print(length)
                with open('lengths.txt', 'r') as f:
                    lines = f.readlines()
                lines[0] = str(length)
                with open('lengths.txt', 'w') as f:
                    f.writelines(lines)
                    f.close()
            if ((run.font.size == Pt(12)) and (run.font.name == "Arial") and not (run.bold)):
                text = text + run.text + ""
                words = list(filter(None, text.split(' ')))
                list_lens.append(len(words))
                length = list_lens[-1]
                print(length)
                with open('lengths.txt', 'r') as f:
                    lines = f.readlines()
                lines[0] = str(length)
                with open('lengths.txt', 'w') as f:
                    f.writelines(lines)
                    f.close()
                
@app.route('/wordcounter')
def wordcounter():
    return render_template("wordcounter.html")

@app.route('/wordcount', methods=['GET', "POST"])
def wordcount():
    if request.method == "POST":
        uploaded_file = request.files['file']
        if uploaded_file.filename != "":
            file_ext = os.path.splitext(uploaded_file.filename)[1]
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                return render_template("500.html")
            filename = secure_filename(uploaded_file.filename)
            uploaded_file.save(filename)
            document = filename
            count_docx(document)
            f = open("lengths.txt", 'r')
            length = f.read()
            os.remove(os.path.join(filename))
    return render_template("counted_words.html", wordcount=length)

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
def text_handler():
    return render_template('summarize_text.html')

@app.route('/summarize-pdf')
def pdf_handler():
    return render_template('summarize_pdf.html')

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

"""T5 Inference"""
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": "Bearer api_EwTwRbogIXYiebTJAvPEIxyxUugItvZMhL"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# @login_required
@app.route('/summarized', methods=['GET', 'POST'])
def infer():
    start = time.time()
    if request.method == "POST":
        url = request.form.get("url")
        rawtext = scrape_articles(url)
        publish_date = scrape_publishdate(url)
        authors = scrape_authors(url)
        title = scrape_title(url)
        _summary = query({"inputs": rawtext})
        for i in _summary:
            _summary = i["summary_text"]
        final_readingTime = readingTime(rawtext)
        summary_reading_time = readingTime(_summary)
        end = time.time()
        final_time = end - start
#         post = Post(url=url, content=rawtext, time_taken=final_readingTime, author=current_user)
#         db.session.add(post)
#         db.session.commit()
        return render_template("results.html", summary=_summary, final_time=final_time, final_reading_time=final_readingTime, summary_reading_time=summary_reading_time)
    else:
        return render_template("index.html")


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
#     posts=Post.query.all()
    return render_template('index.html')

# @login_required
@app.route('/analyze_url', methods=['GET', 'POST'])
def analyze_url():
    start = time.time()
    if request.method == 'POST':
        url = request.form.get("url")
        text = scrape_articles(url)
        publish_date = scrape_publishdate(url)
        authors = scrape_authors(url)
        title = scrape_title(url)
        # Summarization taking place
        _summary = textrank(text)
        # Final reading time
        final_readingTime = readingTime(text)
        summary_reading_time = readingTime(_summary)
        end = time.time()
        final_time = end - start
#         post = Post(url=url, content=text, time_taken=final_readingTime, author=current_user)
#         db.session.add(post)
#         db.session.commit()
        return render_template('results.html', summary=_summary, final_time=final_time, final_reading_time=final_readingTime, summary_reading_time = summary_reading_time)

# Online PDF summarizer
laparams = pdfminer.layout.LAParams()
setattr(laparams, 'all_texts', True)

def extract_text_from_pdf_url(url, user_agent=None):
    resource_manager = PDFResourceManager()
    fake_file_handle = StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=laparams)

    if user_agent == None:
        user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36'
    
    headers = {'User-Agent': user_agent}
    request = urllib.request.Request(url, data=None, headers=headers)

    response = urllib.request.urlopen(request).read()
    fb = BytesIO(response)

    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    for page in PDFPage.get_pages(fb,
                                caching=True,
                                check_extractable=True):
        page_interpreter.process_page(page)
    
    text = fake_file_handle.getvalue()

    # close open handles
    fb.close()
    converter.close()
    fake_file_handle.close()

    if text:
        text = text.replace(u'\xa0', u' ')
        return text

@app.route('/analyze_pdf', methods=['GET', 'POST'])
# @login_required
def analyze_pdf():
    start = time.time()
    if request.method == 'POST':
        url = request.form.get("url")
        text = extract_text_from_pdf_url(url)
        # Summarization taking place
        _summary = textrank(text)
        # Final reading time
        final_readingTime = readingTime(text)
        summary_reading_time = readingTime(_summary)
        end = time.time()
        final_time = end - start
        return render_template('results.html', summary=_summary, final_time=final_time, final_reading_time=final_readingTime, summary_reading_time = summary_reading_time)

if __name__ == '__main__':
    app.run(debug=True)
