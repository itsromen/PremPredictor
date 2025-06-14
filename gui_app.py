# gui_app.py – Premier League match predictor
# Improved version with better error handling and clearer labels

import joblib, os, numpy as np, pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

MODEL_PATH = "premier_league_model.pkl"

class PremierLeaguePredictor:
    # --------------------------------------------------
    def __init__(self, root):
        self.root = root
        self.root.title("Premier League Predictor")
        self.root.geometry("540x480")
        self.root.configure(bg="#f0f0f0")

        # Load model with error handling
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Run train_model.py first.")
            self.model = joblib.load(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not load model: {str(e)}")
            root.quit()
            return

        self._build_widgets()

    # GUI ------------------------------------------------
    def _build_widgets(self):
        big = ("Segoe UI", 13, "bold")
        tk.Label(self.root, text="Enter match statistics",
                 font=big, bg="#f0f0f0").pack(pady=6)

        frm = tk.Frame(self.root, bg="#f0f0f0"); frm.pack()

        # — home possession (user input) —
        self.hp = self._make_entry(frm, "Home possession (%):", cmd=self._sync_away)

        # — away possession (auto filled / read-only) —
        self.ap = self._make_entry(frm, "Away possession (%):", state="readonly")

        # other features - updated labels to match training data
        self.hs  = self._make_entry(frm, "Home shots:")
        self.as_ = self._make_entry(frm, "Away shots:")
        self.att = self._make_entry(frm, "Attendance:")

        ttk.Button(self.root, text="Predict", command=self._predict
                   ).pack(pady=8)

        ttk.Button(self.root, text="Clear", command=self._clear
                   ).pack()

        # result area
        self.result_var = tk.StringVar(value=" ")
        self.result_lbl = tk.Label(self.root, textvariable=self.result_var,
                                   font=("Segoe UI", 14, "bold"),
                                   bg="#f0f0f0")
        self.result_lbl.pack(pady=12)

        self.prob_frame = tk.Frame(self.root, bg="#f0f0f0"); self.prob_frame.pack()

    # helper to build a labelled entry ------------------
    def _make_entry(self, parent, txt, state="normal", cmd=None):
        row = tk.Frame(parent, bg="#f0f0f0"); row.pack(anchor="w", pady=2)
        tk.Label(row, text=txt, width=22, anchor="w", bg="#f0f0f0").pack(side=tk.LEFT)
        var = tk.StringVar()
        entry = tk.Entry(row, textvariable=var, width=10, state=state)
        entry.pack(side=tk.LEFT)
        if cmd:  # callback when user types
            var.trace_add("write", lambda *_, v=var: cmd(v))
        return entry

    # keep away possession = 100 – home possession ------
    def _sync_away(self, home_var):
        val = home_var.get()
        try:
            hp = float(val)
            if hp < 0 or hp > 100:
                raise ValueError
            ap_val = round(100 - hp, 2)
            self.ap.configure(state="normal")
            self.ap.delete(0, tk.END)
            self.ap.insert(0, str(ap_val))
            self.ap.configure(state="readonly")
        except ValueError:
            # invalid number; clear away possession
            self.ap.configure(state="normal")
            self.ap.delete(0, tk.END)
            self.ap.configure(state="readonly")

    # prediction ----------------------------------------
    def _predict(self):
        try:
            hp  = float(self.hp.get())
            ap  = float(self.ap.get())
            hs  = float(self.hs.get())
            aps = float(self.as_.get())
            att = float(self.att.get())
            
            # Basic validation
            if hp < 0 or hp > 100 or ap < 0 or ap > 100:
                raise ValueError("Possession values must be between 0 and 100")
            if abs(hp + ap - 100) > 0.1:  # Allow small floating point errors
                raise ValueError("Home and away possession must sum to 100%")
            if hs < 0 or aps < 0:
                raise ValueError("Shot counts cannot be negative")
            if att < 0:
                raise ValueError("Attendance cannot be negative")
                
        except ValueError as e:
            messagebox.showerror("Input error", f"Please check your inputs: {str(e)}")
            return

        # Feature vector with proper column names to match training data
        import pandas as pd
        feature_names = ["home_possessions", "away_possessions", "home_shots", "away_shots", "attendance"]
        X = pd.DataFrame([[hp, ap, hs, aps, att]], columns=feature_names)
        
        try:
            probs = self.model.predict_proba(X)[0]
            pred  = self.model.predict(X)[0]
            self._show_result(pred, probs)
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error making prediction: {str(e)}")

    # display result ------------------------------------
    def _show_result(self, pred, probs):
        if pred == 1:
            txt = "Prediction: HOME WIN"
            clr = "green"
        else:
            txt = "Prediction: AWAY WIN"
            clr = "red"
        self.result_var.set(txt)
        self.result_lbl.configure(fg=clr)

        # show class probabilities
        for w in self.prob_frame.winfo_children():
            w.destroy()
        tk.Label(self.prob_frame,
                 text="P(home-win) = " + str(round(probs[1]*100, 1)) + "%",
                 bg="#f0f0f0").pack(anchor="w")
        tk.Label(self.prob_frame,
                 text="P(not)      = " + str(round(probs[0]*100, 1)) + "%",
                 bg="#f0f0f0").pack(anchor="w")

    # clear inputs --------------------------------------
    def _clear(self):
        for e in (self.hp, self.ap, self.hs, self.as_, self.att):
            e.configure(state="normal")
            e.delete(0, tk.END)
            if e is self.ap:
                e.configure(state="readonly")
        self.result_var.set(" ")
        self.result_lbl.configure(fg="black")
        for w in self.prob_frame.winfo_children():
            w.destroy()

# run the app ------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = PremierLeaguePredictor(root)
    root.mainloop()