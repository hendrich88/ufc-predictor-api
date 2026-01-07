import gradio as gr
import pandas as pd
import requests
import os
from datetime import date
from joblib import load
import json

# ======================
# KONFIG: GitHub Releases (v1.0)
# ======================
JSON_URL = "https://github.com/hendrich88/ufc-predictor-api/releases/download/v1.0/df_prep_2025-12-17.json"
MODEL_URL = "https://github.com/hendrich88/ufc-predictor-api/releases/download/v1.0/rf_model5.joblib"
JSON_FILE = "df_prep_2025-12-17.json"
MODEL_FILE = "rf_model5.joblib"

# ======================
# STAHOVÁNÍ SOUBORŮ (cache)
# ======================
def download_file(url, filename):
    if os.path.exists(filename):
        return
    print(f"Stahuji {filename}...")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk: f.write(chunk)
    print(f"{filename} stažen.")

download_file(JSON_URL, JSON_FILE)
download_file(MODEL_URL, MODEL_FILE)

# ======================
# LOAD DATA + MODEL (stejné jako predictor.py)
# ======================
df_stats = pd.read_json(JSON_FILE, lines=True)
df_stats = df_stats.sort_values(by="date")
model = load(MODEL_FILE)

# ... [kopíruj všechny selected_features, stats, DECAY_* konstanty, apply_decay, get_stats_for_fighter, build_input_df funkce z predictor.py] ...

def predict_fight(fighter1: str, fighter2: str) -> dict:
    input_1 = build_input_df(fighter1, fighter2)
    input_2 = build_input_df(fighter2, fighter1)
    prob1 = model.predict_proba(input_1)[0]
    prob2 = model.predict_proba(input_2)[0]
    avg_prob_f1 = (prob1[1] + (1 - prob2[1])) / 2
    avg_prob_f2 = (prob1[0] + (1 - prob2[0])) / 2
    if avg_prob_f1 > avg_prob_f2:
        return f"{fighter1} vyhraje ({avg_prob_f1:.1%})"
    else:
        return f"{fighter2} vyhraje ({avg_prob_f2:.1%})"

# ======================
# GRADIO INTERFACE (HF Spaces)
# ======================
gr.Interface(
    fn=predict_fight,
    inputs=[gr.Textbox(label="Fighter 1"), gr.Textbox(label="Fighter 2")],
    outputs=gr.Textbox(label="Predikce"),
    title="UFC Fight Predictor",
    description="Tvůj 69% model s +16% ROI na 2-fight tiketech"
).launch(server_port=7860)
