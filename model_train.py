import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# ==============================
# LOAD DATA
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "insurance_data.csv")

df = pd.read_csv(data_path)

print("✅ Data Loaded:", df.shape)

# ==============================
# DATA CLEANING
# ==============================
df = df.dropna()

df["Premium"] = pd.to_numeric(df["Premium"], errors="coerce")
df["Coverage"] = pd.to_numeric(df["Coverage"], errors="coerce")
df["Claim Ratio"] = pd.to_numeric(df["Claim Ratio"], errors="coerce")

df = df.dropna()

# ==============================
# FEATURE ENGINEERING
# ==============================

# Premium per lakh (important feature)
df["premium_per_lakh"] = df["Premium"] / (df["Coverage"] / 100000)

# Disease flag (stronger logic)
df["Disease Flag"] = df["Diseases Covered"].apply(
    lambda x: 1 if isinstance(x, str) and len(x.strip()) > 3 else 0
)

# Better target (SMART LOGIC)
df["Best Plan"] = (
    (df["Claim Ratio"] >= 90) &
    (df["Coverage"] >= 500000) &
    (df["premium_per_lakh"] < df["premium_per_lakh"].median())
).astype(int)

# ==============================
# FEATURES & TARGET
# ==============================
features = ["Premium", "Coverage", "Claim Ratio", "premium_per_lakh", "Disease Flag"]

X = df[features]
y = df["Best Plan"]

print("✅ Features Ready")

# ==============================
# TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# MODEL TRAINING
# ==============================
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# EVALUATION
# ==============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {accuracy:.2f}")

# ==============================
# FEATURE IMPORTANCE (VERY USEFUL)
# ==============================
importance = model.feature_importances_

print("\n🔍 Feature Importance:")
for f, imp in zip(features, importance):
    print(f"{f}: {round(imp, 3)}")

# ==============================
# SAVE MODEL
# ==============================
model_path = os.path.join(BASE_DIR, "model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model saved as model.pkl")