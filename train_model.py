# train_model.py – Premier League match-outcome model (22-23 season)
# Streamlined version assuming preprocessed data from preprocess.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# ------------------------------------------------------------------
# 1. Load preprocessed data
# ------------------------------------------------------------------
CSV_PATH = "premier_league_cleaned.csv"
if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError("Cannot locate " + CSV_PATH + ". Run preprocess.py first.")

df = pd.read_csv(CSV_PATH)

# ------------------------------------------------------------------
# 2. Create target variable and select features
# ------------------------------------------------------------------
# Target: 1 if home team wins, else 0 (draw or away win)
df["home_win"] = np.where(df["Goals Home"] > df["Away Goals"], 1, 0)

FEATURES = [
    "home_possessions", "away_possessions",
    "home_shots", "away_shots",
    "attendance"
]
X = df[FEATURES]
y = df["home_win"]

# ------------------------------------------------------------------
# 3. Train/test split
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ------------------------------------------------------------------
# 4. Pipeline + hyperparameter grid search
# ------------------------------------------------------------------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("gbc", GradientBoostingClassifier(random_state=42))
])

param_grid = {
    "gbc__n_estimators":  [200, 300, 400],
    "gbc__learning_rate": [0.05, 0.10, 0.20],
    "gbc__max_depth":     [2, 3, 4]
}

gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0
)

gs.fit(X_train, y_train)
best_model = gs.best_estimator_

# ------------------------------------------------------------------
# 5. Evaluation
# ------------------------------------------------------------------
test_acc = accuracy_score(y_test, best_model.predict(X_test))
print(f"Model accuracy: {test_acc:.3f}")

# ------------------------------------------------------------------
# 6. Save model
# ------------------------------------------------------------------
MODEL_PATH = "premier_league_model.pkl"
joblib.dump(best_model, MODEL_PATH)
print("Model saved to", MODEL_PATH)