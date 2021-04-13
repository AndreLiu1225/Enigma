import numpy as np
import pandas as pd
import warnings

# Importing summarization packages
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from scipy import spatial
import networkx as nx

# Flask packages
from flask import Flask, render_template, request, jsonify, url_for, flash, redirect
from flask_wtf import FlaskForm
from wtforms import StringField, TextField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length, Email
from flask_mail import Mail, Message

import newspaper
import json
import email_validator
import os
from waitress import serve
from newspaper import fulltext

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

app = Flask(__name__)
app.secret_key = "Andre1225"
mail = Mail()

UPLOAD_FOLDER = 'C:/Users/liuis/Desktop/g_summary/Enigma/static/styles/upload'

# Configurations for mailing service
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 465
app.config["MAIL_USE_SSL"] = True
app.config['MAIL_USE_TLS'] = False
app.config["MAIL_USERNAME"] = 'andreliu2004@gmail.com'
app.config["MAIL_PASSWORD"] = 'andre1225'
ADMINS = ['andreliu2004@gmail,com']
ALLOWED_EXTENSIONS = ['.csv', 'xls', 'xlsx', 'xlsm']
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

mail.init_app(app)

class ContactForm(FlaskForm):
    name = StringField("Name", [DataRequired(), Length(max=15, min=2)])
    email = StringField("Email", [DataRequired(), Email(message=('Not a valid email address')), Length(max=30, min=2)])
    subject = StringField("Subject", [DataRequired(), Length(max=30, min=2)])
    message = TextAreaField("Message", [DataRequired(), Length(min=4, message=('Your message is too short.'), max=400)])
    submit = SubmitField("Send")

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/in-progress')
def inprogress():
    return render_template('progress.html')

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

global _summary
    
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form.get("url")
        text = scrape_articles(url)
        sentences = sent_tokenize(text)
        sentences_clean = [re.sub(r'[^\w\s]','',sentence.lower()) for sentence in sentences]
        sentence_tokens=[[words for words in sentence.split(' ') if words not in stop_words] for sentence in sentences_clean]

        # Computing word embedding
        w2v = Word2Vec(sentence_tokens, size=1, min_count=1, iter=1000)
        sentence_embeddings=[[w2v[word][0] for word in words] for words in sentence_tokens]
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
        authors = scrape_authors(url)
        publish_date = scrape_publishdate(url)
        return render_template('results.html', summary=_summary, authors=authors, publish_date=publish_date)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)
