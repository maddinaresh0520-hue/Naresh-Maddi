from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

# AI imports
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ==============================
# LOAD MODEL
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError("❌ model.pkl not found")

model = pickle.load(open(model_path, "rb"))

# ==============================
# LOAD DATA
# ==============================
data_path = os.path.join(BASE_DIR, "insurance_data.csv")
df = pd.read_csv(data_path)

df["Diseases Covered"] = df["Diseases Covered"].astype(str).str.lower()
df["Company"] = df["Company"].astype(str)

# ==============================
# LLM SETUP
# ==============================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4
)

prompt = PromptTemplate(
    input_variables=["plan", "user", "rank"],
    template="""
You are a friendly insurance advisor.

Use VERY SIMPLE English.

RULES:
- Always use "-"
- Give 2 Pros
- Give 1 or 2 Cons
- Short sentences
- If rank = 1 → say BEST

FORMAT:

Pros:
- text
- text

Cons:
- text

Recommendation:
- text

User:
{user}

Plan:
{plan}

Rank:
{rank}
"""
)

# ==============================
# AI FUNCTION
# ==============================
def generate_ai_insight(plan, user, rank):
    try:
        chain = prompt | llm
        res = chain.invoke({
            "plan": str(plan),
            "user": str(user),
            "rank": rank
        }).content
        return res
    except Exception as e:
        print("AI ERROR:", e)
        return """Pros:
- Good coverage
- Trusted company

Cons:
- Premium slightly high

Recommendation:
- Safe and balanced plan"""


# ==============================
# PARSE AI OUTPUT
# ==============================
def parse_ai_output(text):
    pros, cons, rec = [], [], []
    section = None

    for line in text.split("\n"):
        line = line.strip()

        if line.lower().startswith("pros"):
            section = "pros"
        elif line.lower().startswith("cons"):
            section = "cons"
        elif line.lower().startswith("recommendation"):
            section = "rec"
        elif line.startswith("-"):
            if section == "pros":
                pros.append(line[1:].strip())
            elif section == "cons":
                cons.append(line[1:].strip())
            elif section == "rec":
                rec.append(line[1:].strip())

    return {
        "pros": pros if pros else ["Good coverage"],
        "cons": cons if cons else ["No major issues"],
        "rec": rec if rec else ["This is a decent plan"]
    }


# ==============================
# FEATURE ENGINEERING
# ==============================
def prepare_features(plan, user):

    premium = float(plan["Premium"])

    # Age impact
    for age in user["ages"]:
        if str(age).isdigit():
            age = int(age)
            if age > 50:
                premium += 2000
            elif age > 40:
                premium += 1500
            elif age > 30:
                premium += 1000

    # Family size
    members = len([a for a in user["ages"] if a])
    if members > 1:
        premium += (members - 1) * 1200

    # Disease impact
    disease_flag = 0
    for d in user["diseases"]:
        if d and d.lower() in plan["Diseases Covered"]:
            disease_flag = 1
            break

    coverage = max(float(plan["Coverage"]), 100000)
    premium_per_lakh = premium / (coverage / 100000)

    return premium, [
        premium,
        plan["Coverage"],
        plan["Claim Ratio"],
        premium_per_lakh,
        disease_flag
    ]


# ==============================
# RECOMMENDATION ENGINE
# ==============================
def recommend_plans(user, company_filter=None, claim_filter=None):

    temp_df = df.copy()

    # Company filter
    if company_filter:
        temp_df = temp_df[temp_df["Company"] == company_filter]

    # Claim filter
    if claim_filter:
        claim_filter = int(claim_filter)
        temp_df = temp_df[temp_df["Claim Ratio"] >= claim_filter]

    if temp_df.empty:
        return temp_df

    scores = []
    adjusted_premiums = []

    for _, row in temp_df.iterrows():
        premium, features = prepare_features(row, user)

        try:
            prob = model.predict_proba([features])[0][1]
        except:
            prob = 0.5

        scores.append(prob)
        adjusted_premiums.append(int(premium))

    temp_df["ML Score"] = scores
    temp_df["Adjusted Premium"] = adjusted_premiums

    return temp_df.sort_values(by="ML Score", ascending=False)


# ==============================
# FLASK APP
# ==============================
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    companies = sorted(df["Company"].dropna().unique().tolist())

    if request.method == "POST":

        names = request.form.getlist("name[]")
        ages = request.form.getlist("age[]")
        relations = request.form.getlist("relation[]")
        diseases = request.form.getlist("disease[]")

        coverage = int(request.form.get("coverage", 300000))
        company_filter = request.form.get("company")
        claim_filter = request.form.get("claim_ratio")

        user = {
            "names": names,
            "ages": ages,
            "relations": relations,
            "diseases": diseases,
            "coverage": coverage
        }

        results = recommend_plans(user, company_filter, claim_filter)

        # HANDLE EMPTY RESULT (IMPORTANT FIX)
        if results.empty:
            return render_template(
                "index.html",
                tables=[],
                ai_outputs=[],
                companies=companies
            )

        parsed_outputs = []

        for idx, (_, row) in enumerate(results.iterrows()):
            rank = idx + 1
            raw = generate_ai_insight(row.to_dict(), user, rank)
            parsed_outputs.append(parse_ai_output(raw))

        return render_template(
            "index.html",
            tables=results.to_dict(orient="records"),
            ai_outputs=parsed_outputs,
            companies=companies
        )

    return render_template(
        "index.html",
        tables=None,
        ai_outputs=None,
        companies=companies
    )


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(debug=True)