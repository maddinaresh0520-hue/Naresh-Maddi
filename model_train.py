import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("insurance_data.csv")

# ==============================
# FEATURE ENGINEERING
# ==============================

# Premium per lakh
df["premium_per_lakh"] = df["Premium"] / (df["Coverage"] / 100000)

# Disease flag (generalized)
df["Disease Flag"] = df["Diseases Covered"].apply(
    lambda x: 1 if isinstance(x, str) and x.strip() != "" else 0
)

# Better target (multi-factor)
df["Best Plan"] = (
    (df["Claim Ratio"] > 90) &
    (df["Coverage"] > 500000)
).astype(int)

# ==============================
# FEATURES & TARGET
# ==============================
X = df[["Premium", "Coverage", "Claim Ratio", "premium_per_lakh", "Disease Flag"]]
y = df["Best Plan"]

# ==============================
# TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# MODEL TRAINING
# ==============================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==============================
# EVALUATION
# ==============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {accuracy:.2f}")

# ==============================
# SAVE MODEL
# ==============================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as model.pkl")