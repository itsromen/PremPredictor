# train_model.py  – Premier-League match-outcome model (22-23 season)  
#  
# Re-write from the ground up, focused on stronger accuracy  
# and portability for your GUI.  Requires pandas, scikit-learn, joblib.  
  
import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.pipeline import Pipeline  
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import GradientBoostingClassifier  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  
import joblib  
import os  
  
# ------------------------------------------------------------------  
# 1. Load data  
# ------------------------------------------------------------------  
CSV_PATH = "premier_league_cleaned.csv"  
if not os.path.isfile(CSV_PATH):  
    raise FileNotFoundError("Cannot locate " + CSV_PATH)  
  
df = pd.read_csv(CSV_PATH, encoding="ascii")  
  
# ------------------------------------------------------------------  
# 2. House-keeping / cleaning  
# ------------------------------------------------------------------  
# a)  trim any stray spaces in header names  
df.columns = df.columns.str.strip()  
  
# b) attendance  – ensure numeric  
df["attendance"] = (  
    df["attendance"]  
        .astype(str)          # guarantees .str exists  
        .str.replace(",", "") # remove thousands sep  
)  
df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")  
  
# ------------------------------------------------------------------  
# 3. Remove rows with missing critical fields  
# ------------------------------------------------------------------  
key_cols = [  
    "Goals Home", "Away Goals",  
    "home_possessions", "away_possessions",  
    "home_shots", "away_shots",  
    "attendance"  
]  
df = df.dropna(subset=key_cols)  
  
# ------------------------------------------------------------------  
# 4. Target + features  
#     Target: 1 if home team wins, else 0 (draw or away win)  
# ------------------------------------------------------------------  
df["home_win"] = np.where(df["Goals Home"] > df["Away Goals"], 1, 0)  
  
FEATURES = [  
    "home_possessions", "away_possessions",  
    "home_shots", "away_shots",  
    "attendance"  
]  
X = df[FEATURES]  
y = df["home_win"]  
  
# ------------------------------------------------------------------  
# 5. Train / test split  
# ------------------------------------------------------------------  
X_train, X_test, y_train, y_test = train_test_split(  
    X, y, test_size=0.20, random_state=42, stratify=y  
)  
  
# ------------------------------------------------------------------  
# 6. Pipeline + hyper-parameter grid search  
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
# 7. Evaluation  
# ------------------------------------------------------------------  
train_acc = accuracy_score(y_train, best_model.predict(X_train))  
test_pred = best_model.predict(X_test)  
test_acc  = accuracy_score(y_test, test_pred)  
  
print("--------------------------------------------------------------")  
print(" Best parameters : ", gs.best_params_)  
print(" Training  acc   : ", round(train_acc, 3))  
print(" Test      acc   : ", round(test_acc, 3))  
print(" Confusion matrix\n", confusion_matrix(y_test, test_pred))  
print(" Classification report\n", classification_report(y_test, test_pred))  
print("--------------------------------------------------------------")  
  
# ------------------------------------------------------------------  
# 8. Persist model  
# ------------------------------------------------------------------  
MODEL_PATH = "premier_league_model.pkl"  
joblib.dump(best_model, MODEL_PATH)  
print("Model saved to", MODEL_PATH)  