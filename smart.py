import os
import pickle
import random
import smtplib
import json # Added for AI response parsing
import pandas as pd
from email.message import EmailMessage
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "smart_ai_insurance_key_2024")

# --- Configuration & Model Loading ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

try:
    with open(model_path, "rb") as f:
        insurance_model = pickle.load(f)
except:
    insurance_model = None

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")

DISEASES = ["Diabetes", "Hypertension", "Asthma", "Heart Disease", "Thyroid"]
COMPANIES = ["Star Health", "HDFC Ergo", "ICICI Lombard", "Care Health", "Niva Bupa"]

# ==============================
# AUTHENTICATION ROUTES
# ==============================

@app.route('/')
def root():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        otp = str(random.randint(100000, 999999))
        session['otp'] = otp
        session['temp_email'] = email
        
        try:
            msg = EmailMessage()
            msg.set_content(f"Your Smart AI Insurance Login OTP is: {otp}")
            msg['Subject'] = 'Login Verification Code'
            msg['From'] = SENDER_EMAIL
            msg['To'] = email

            # FIX: Switched to Port 587 + STARTTLS for better reliability
            with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
                smtp.starttls() 
                smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
                smtp.send_message(msg)
            
            return render_template('verify_otp.html', email=email)
        except Exception as e:
            print(f"SMTP Error: {e}")
            flash("Failed to send email. Check your App Password and Internet.")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/verify', methods=['POST'])
def verify():
    user_otp = request.form.get('otp')
    if user_otp == session.get('otp'):
        session['user_email'] = session.get('temp_email')
        session.pop('otp', None)
        return redirect(url_for('index'))
    
    flash("Invalid OTP. Please try again.")
    return render_template('verify_otp.html', email=session.get('temp_email'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ==============================
# CORE APP ROUTES
# ==============================

@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    
    user_name = session['user_email'].split('@')[0]
    tables = None
    ai_outputs = []

    if request.method == 'POST':
        relations = request.form.getlist('relation[]')
        ages = request.form.getlist('age[]')
        diseases = request.form.getlist('disease[]')
        coverage = float(request.form.get('coverage', 500000))
        selected_company = request.form.get('company')

        target_companies = [selected_company] if selected_company else COMPANIES
        
        results = []
        for comp in target_companies:
            avg_age = sum(int(a) for a in ages) / len(ages) if ages else 30
            disease_load = 1.3 if any(d and d != 'None' for d in diseases) else 1.0
            
            premium = (coverage * 0.018) + (avg_age * 150) * disease_load
            results.append({'Company': comp, 'Adjusted Premium': int(premium)})
        
        tables = results

        for plan in tables:
            try:
                # FIX: Explicitly mentioned JSON in prompt for the llama3 model
                prompt = (f"Provide 2 pros and 2 cons for {plan['Company']} insurance "
                          f"for ages {ages} with conditions {diseases}. "
                          "Return only a JSON object with keys 'pros' and 'cons'.")
                
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-8b-8192",
                    response_format={"type": "json_object"}
                )
                ai_outputs.append(json.loads(response.choices[0].message.content))
            except:
                ai_outputs.append({
                    'pros': ["Reliable local network", "Quick claim settlement"],
                    'cons': ["Slightly higher premium", "Limited wellness benefits"]
                })

    return render_template('index.html', 
                           user_name=user_name, 
                           diseases=DISEASES, 
                           companies=COMPANIES, 
                           tables=tables, 
                           ai_outputs=ai_outputs)

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get('message')
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": user_msg}],
            model="mixtral-8x7b-32768",
        )
        return jsonify({'response': chat_completion.choices[0].message.content})
    except:
        return jsonify({'response': "I'm having trouble connecting. Try again in a moment!"})

if __name__ == "__main__":
    # Ensure port is handled for Render deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)