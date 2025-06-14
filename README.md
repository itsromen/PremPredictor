# Premier League Match-Outcome Predictor

Predicts whether the **home team** will win a Premier-League fixture using five match-day statistics.

---
## Project structure

| File / folder | Purpose |
|---------------|---------|
| `premier_league_cleaned.csv` | Finalised dataset after data‐cleaning – used for model training. |
| `preprocess.py` | Script that loads the raw CSV, cleans it and writes `premier_league_cleaned.csv`. |
| `train_model.py` | Trains a scikit-learn pipeline on the cleaned data and saves `premier_league_model.pkl`. |
| `gui_app.py`    | Tkinter desktop app that loads the model and lets you predict match outcomes interactively. |
| `README.md`     | You are here – quick guide to installation and usage. |

---
## Quick start

1. **Pre-processing**  
   ```bash
   python preprocess.py
   ```
   This reads the raw export, cleans and feature-engineers the table and writes `premier_league_cleaned.csv`.

2. **Training**  
   ```bash
   python train_model.py
   ```
   Produces `premier_league_model.pkl`, a pipeline containing a `StandardScaler` and a gradient-boosting classifier.

3. **Run the GUI**  
   ```bash
   python gui_app.py
   ```
   Enter home-team possession, shots on target, attendance, etc.  The away-team possession is auto-calculated so both sides sum to 100 %.  The app returns the predicted class and the associated probabilities.

---
## Data columns used

- `home_possessions`  
- `away_possessions`  
- `home_shots`  
- `away_shots`  
- `attendance`

These are scaled automatically by the pipeline before classification.