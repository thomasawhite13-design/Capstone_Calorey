from flask import Flask, jsonify, render_template, request, redirect, url_for, session
from memory import FirestoreCheckpointer
from nutrition_agent import NutritionAgent
import os
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import firestore

CLOUD_SECRET_PATH = "/secrets/firestore-key.json"
LOCAL_SECRET_PATH = "df-trial-487814-b3428a5d8bc5.json"

# 1. Logic to choose the right key
if os.path.exists(CLOUD_SECRET_PATH):
    # This runs when deployed to Cloud Run
    cred = credentials.Certificate(CLOUD_SECRET_PATH)
elif os.path.exists(LOCAL_SECRET_PATH):
    # This runs when you're testing locally
    cred = credentials.Certificate(LOCAL_SECRET_PATH)
else:
    cred = None

# 2. Initialise the app
if not firebase_admin._apps:
    if cred:
        firebase_admin.initialize_app(cred)
    else:
        firebase_admin.initialize_app()

db = firestore.Client(database="nutrition-agent-store")

app = Flask(__name__)
app.secret_key = "super_secret_nutrition_key"


# Initialise the agent once at startup
checkpointer = FirestoreCheckpointer()
agent = NutritionAgent(checkpointer=checkpointer, db=db)

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    user_id = request.form.get('user_id')
    if user_id:
        session['user_id'] = user_id
        return redirect(url_for('home'))
    return redirect(url_for('index'))

@app.route('/home')
def home():
    if 'user_id' not in session: return redirect(url_for('index'))
    return render_template('home.html', user_id=session['user_id'])

@app.route('/chat')
def chat_page():
    """ Renders the actual HTML page with the chat window."""
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return render_template('chat.html')

@app.route('/chat_api', methods=['POST'])
def chat_api():
    if 'user_id' not in session:
        return jsonify({"reply": "Session expired. Please log in again."}), 401
    
    data = request.json
    user_message = data.get('message')
    if not user_message or not user_message.strip():
        return jsonify({"reply": "Please enter a message."}), 400
    
    # Run the agent!
    reply = agent.chat(thread_id=session['user_id'], user_message=user_message, user_id=session['user_id'])

    show_plan_button = agent.meal_generated(session['user_id'])
    
    return jsonify({"reply": reply, "show_plan_button": show_plan_button})

@app.route('/profile')
def profile():
    if 'user_id' not in session: return redirect(url_for('index'))
    
    doc = db.collection("users").document(session['user_id']).get()
    
    if not doc.exists:
        return render_template('profile.html', profile=None)
        
    profile_data = doc.to_dict().get('user_profile')
    return render_template('profile.html', profile=profile_data)

@app.route('/meal-plan')
def meal_plan():
    if 'user_id' not in session: return redirect(url_for('index'))
    
    doc = db.collection("users").document(session['user_id']).get()
    
    if not doc.exists:
        return render_template('meal_plan.html', plan=None)
        
    plan_data = doc.to_dict().get('current_meal_plan')
    return render_template('meal_plan_new.html', plan=plan_data)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
