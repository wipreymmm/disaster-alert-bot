from flask import Flask, render_template, redirect, url_for
from datetime import datetime

# App
app = Flask(__name__)

# Routes
@app.route('/', endpoint='index') # Chatbot
def index():
    return render_template('index.html', title='DisasterAlertBot')

@app.route('/home') # Landing Page
def home():
    return render_template('home.html', title='DisasterAlertBot | Home')

"""
@app.route('/signin') # Sign up Page
def signup():
    return render_template('signup.html', title='DisasterAlertBot | Sign Up')

@app.route('/login') # Log in Page
def login():
    return render_template('login.html', title='DisasterAlertBot | Log In')
"""

# Context Processors
@app.context_processor
def inject_now():
    contexts = {
        'now': datetime.now()
    }
    return contexts