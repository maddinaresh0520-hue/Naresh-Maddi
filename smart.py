import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
import google.generativeai as genai
import pandas as pd

app = Flask(__name__)

# 1. CONFIGURATION
# Replace with your actual API key or set as environment variable
os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY_HERE"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')

# Mock Data for Dropdowns
DISEASES = ["Diabetes", "Hypertension", "Asthma", "Heart Disease", "Thyroid"]
COMPANIES = ["Star Health", "HDFC Ergo", "ICICI Lombard", "Care Health", "Niva Bupa"]

# 2. ROUTES

# THIS FIXES THE "NOT FOUND" ERROR
@app.route('/')
def root():
    # Automatically sends users to the login page when they open the site
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        # Add your own login logic here
        return redirect(url_for('index', user_name=email.split('@')[0]))
    return render_template('login.html')

@app.route('/index')
def index():
    user_name = request.args.get('user_name', 'Guest')
    return render_template('index.html', 
                           diseases=DISEASES, 
                           companies=COMPANIES, 
                           user_name=user_name)

@app.route('/index', methods=['POST'])
def get_recommendations():
    # Get form data
    relations = request.form.getlist('relation[]')
    ages = request.form.getlist('age[]')
    diseases = request.form.getlist('disease[]')
    coverage = request.form.get('coverage')
    
    # Simple Mock Calculation Logic
    results = []
    for comp in COMPANIES:
        base_premium = int(coverage) * 0.02
        # Add age loading
        age_factor = sum([int(a) for a in ages]) * 10
        results.append({
            "Company": comp,
            "Adjusted Premium": int(base_premium + age_factor)
        })

    # AI Insights (Simplified for example)
    ai_outputs = []
    for res in results:
        ai_outputs.append({
            "pros": ["High claim settlement", "Cashless hospitals"],
            "cons": ["Waiting period for pre-existing"],
            "rec": ["Great value for family coverage"]
        })

    return render_template('index.html', 
                           tables=results, 
                           ai_outputs=ai_outputs,
                           diseases=DISEASES, 
                           companies=COMPANIES)

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get('message')
    try:
        response = model.generate_content(f"You are an insurance expert. Answer this: {user_msg}")
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"response": "I'm having trouble connecting to my brain right now!"})

# 3. RUN THE APP
if __name__ == "__main__":
    # This port configuration is required for Render deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)