from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import pickle
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smart-ai-secret-key-123' # Change this in production
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
# ==============================
# 1. AUTHENTICATION SETUP
# ==============================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Mock Database for Users
users = {
    "maddi.naresh0520@gmail.com": {
        "password": generate_password_hash("NareshIT@2026"),
        "name": "Naresh Maddi"
    }
}

class User(UserMixin):
    def __init__(self, id, name):
        self.id = id
        self.name = name

@login_manager.user_loader
def load_user(user_id):
    if user_id not in users:
        return None
    return User(user_id, users[user_id]['name'])

# ==============================
# 2. DATA & MODEL LOADING
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
data_path = os.path.join(BASE_DIR, "insurance_data.csv")

if not os.path.exists(model_path): raise FileNotFoundError("❌ model.pkl not found")
if not os.path.exists(data_path): raise FileNotFoundError("❌ insurance_data.csv not found")

model = pickle.load(open(model_path, "rb"))
df = pd.read_csv(data_path)

# Clean data
df["Diseases Covered"] = df["Diseases Covered"].astype(str).str.lower()
df["Company"] = df["Company"].astype(str)
df["Premium"] = pd.to_numeric(df["Premium"], errors="coerce")
df["Coverage"] = pd.to_numeric(df["Coverage"], errors="coerce")
df["Claim Ratio"] = pd.to_numeric(df["Claim Ratio"], errors="coerce")
df = df.dropna()

# ==============================
# 3. AI LOGIC & CHATBOT SETUP
# ==============================
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

def generate_ai_insight(plan, user, rank):
    prompt = f"""
    Context: Rank {rank} Insurance Plan for a user with profile {user}.
    Plan Details: {plan}
    Task: Provide 2 Pros, 1 Con, and a 1-sentence recommendation in simple English.
    Format:
    Pros: - point 1 - point 2
    Cons: - point 1
    Recommendation: - text
    """
    try:
        return llm.invoke(prompt).content
    except:
        return "Pros: - Budget friendly - Good claim ratio\nCons: - Basic coverage\nRecommendation: - Safe choice."

def parse_ai_output(text):
    sections = {"pros": [], "cons": [], "rec": []}
    current = None
    for line in text.split('\n'):
        if "pros" in line.lower(): current = "pros"
        elif "cons" in line.lower(): current = "cons"
        elif "recommendation" in line.lower(): current = "rec"
        elif "-" in line and current:
            sections[current].append(line.split("-")[-1].strip())
    return sections

# ==============================
# 4. RECOMMENDATION ENGINE
# ==============================
def recommend_plans(user_data, company_filter=None):
    temp_df = df.copy()
    if company_filter:
        temp_df = temp_df[temp_df["Company"] == company_filter]
    
    target_cov = int(user_data.get('coverage', 500000))
    temp_df = temp_df[(temp_df["Coverage"] >= target_cov * 0.5) & (temp_df["Coverage"] <= target_cov * 2.0)]

    scores, premiums = [], []
    for _, row in temp_df.iterrows():
        prem = float(row["Premium"])
        
        # 🟢 FIX: Safely get ages to prevent KeyError
        user_ages = user_data.get('ages', [])
        for age in user_ages:
            if str(age).isdigit() and int(age) > 45: 
                prem += 2000
        
        # 🟢 FIX: Safely get diseases
        user_diseases = user_data.get('diseases', [])
        disease_match = 1 if any(d.lower() in row["Diseases Covered"] for d in user_diseases if d) else 0
        
        features = [[prem, float(row["Coverage"]), float(row["Claim Ratio"]), prem/(row["Coverage"]/100000), disease_match]]
        feat_df = pd.DataFrame(features, columns=["Premium", "Coverage", "Claim Ratio", "premium_per_lakh", "Disease Flag"])
        
        try:
            scores.append(model.predict_proba(feat_df)[0][1])
        except:
            scores.append(0.5)
        premiums.append(int(prem))

    temp_df["ML Score"] = scores
    temp_df["Adjusted Premium"] = premiums
    return temp_df.sort_values("ML Score", ascending=False).head(3)

# ==============================
# 5. FLASK ROUTES
# ==============================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        
        user_data = users.get(email)
        if user_data and check_password_hash(user_data['password'], password):
            user_obj = User(email, user_data['name'])
            login_user(user_obj)
            return redirect(url_for('home'))
        else:
            flash("Invalid credentials. Try admin@smartai.com / password123")
            
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route("/", methods=["GET", "POST"])
@login_required 
def home():
    # Setup Dropdown lists
    all_diseases = sorted(list(set([d.strip().title() for sub in df["Diseases Covered"].dropna() for d in sub.split(",")])))
    companies = sorted(df["Company"].unique().tolist())

    # Only process data IF the user clicks the submit button
    if request.method == "POST":
        user = {
            "ages": request.form.getlist("age[]"),
            "diseases": request.form.getlist("disease[]"),
            "coverage": request.form.get("coverage", 500000)    
        }
        results = recommend_plans(user, request.form.get("company"))
        insights = [parse_ai_output(generate_ai_insight(r, user, i+1)) for i, r in enumerate(results.to_dict('records'))]
        
        return render_template("index.html", 
                               tables=results.to_dict('records'), 
                               ai_outputs=insights, 
                               diseases=all_diseases, 
                               companies=companies, 
                               user_name=current_user.name)

    # Initial page load (GET request)
    return render_template("index.html", diseases=all_diseases, companies=companies, user_name=current_user.name)

@app.route("/chat", methods=["POST"])
@login_required
def chat():
    msg = request.json.get("message")
    try:
        res = llm.invoke(f"Help Assistant Bot: Answer this insurance query shortly: {msg}").content
        return jsonify({"response": res})
    except:
        return jsonify({"response": "I'm having trouble connecting right now. Please try again!"})

if __name__ == "__main__":
    app.run(debug=True)