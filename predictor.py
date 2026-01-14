import os
import requests
import pandas as pd
from datetime import date
from joblib import load
import shap
import numpy as np

# ======================
# KONFIG
# ======================

JSON_FILE = "df_prep_clean_2026-01-08.json"
MODEL_URL = "https://github.com/hendrich88/ufc-predictor-api/releases/download/v1.0/rf_model5.joblib"
MODEL_FILE = "rf_model5.joblib"

# ======================
# STAHOVÁNÍ MODELU
# ======================

def download_file(url, filename):
    if os.path.exists(filename):
        return
    print(f"Stahuji {filename}...")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"{filename} stažen.")

download_file(MODEL_URL, MODEL_FILE)

# ======================
# LOAD DATA + MODEL
# ======================

if not os.path.exists(JSON_FILE):
    raise FileNotFoundError(f"Missing data file: {JSON_FILE}")

df_stats = pd.read_json(JSON_FILE, lines=True)
df_stats["date"] = pd.to_datetime(df_stats["date"])
df_stats = df_stats.sort_values(by="date")

model = load(MODEL_FILE)
explainer = shap.TreeExplainer(model)

# ======================
# PARAMETRY DECAY
# ======================

DECAY_THRESHOLD = 180
DECAY_RATE = 0.05
MIN_VALUE = 0  # spodní hranice pro všechny statistiky

def apply_decay(value, inactive_days):
    if inactive_days <= DECAY_THRESHOLD:
        return value
    t = (inactive_days - DECAY_THRESHOLD) / 365
    factor = 1 - DECAY_RATE * (t ** 2)
    factor = max(0.7, factor)
    return max(MIN_VALUE, value * factor)

# ======================
# FEATURES
# ======================

selected_features = [
    "diff_age", "diff_elo_before", "diff_ratio_min_sig_strikes_head_lnd_diff", "diff_win_rate",
    "diff_smt_min_sub_att", "diff_ratio_min_sig_strikes_lnd_diff", "diff_smt_sig_strikes_lnd_diff",
    "diff_lose_rate", "diff_ratio_min_sub_att_get", "diff_avg_cplx_kd", "diff_avg_cplx_min_td_thr",
    "diff_ratio_sub_att_diff", "diff_avg_cplx_acc_def_sig_strikes_head_lnd_get", "diff_avg_cplx_sub_att",
    "diff_ratio_sig_strikes_head_lnd_diff", "diff_avg_cplx_sig_strikes_grnd_lnd_get",
    "diff_smt_acc_def_sig_strikes_head_lnd_get", "diff_avg_cplx_kd_get", "diff_ratio_min_kd_diff",
    "diff_avg_cplx_min_sig_strikes_head_lnd_get", "diff_avg_cplx_min_td_lnd", "diff_avg_cplx_sig_strikes_head_lnd",
    "diff_avg_cplx_min_sig_strikes_leg_lnd_get", "diff_avg_cplx_td_thr", "diff_smt_acc_att_strikes_lnd",
    "diff_ratio_lose_ko", "diff_avg_cplx_cntrl_get", "diff_ratio_sig_strikes_grnd_thr_diff",
    "diff_ratio_att_td_lnd", "diff_avg_cplx_acc_def_strikes_lnd_get", "diff_avg_cplx_min_sig_strikes_grnd_thr_get",
    "diff_smt_min_sig_strikes_head_lnd_diff", "diff_ratio_td_thr", "diff_smt_acc_att_td_lnd",
    "diff_avg_cplx_sig_strikes_grnd_thr", "diff_smt_acc_att_sig_strikes_grnd_lnd", "diff_smt_sub_att_diff",
    "diff_smt_acc_def_sig_strikes_grnd_lnd_get", "diff_ratio_reach", "diff_avg_cplx_acc_def_sig_strikes_leg_lnd_get"
]

stats = [f.replace("diff_", "", 1) for f in selected_features if f not in ["diff_age", "diff_elo_before"]]
date_fight_pd = pd.to_datetime(date.today())

# ======================
# SHAP GROUPS
# ======================

groups = {
    'Age': ["diff_age"],
    'Reach': ["diff_ratio_reach"],
    'Win/Lose Rates': ["diff_win_rate","diff_lose_rate"],
    'ELO': ["diff_elo_before"],
    'Boxing Attack': ["diff_ratio_min_sig_strikes_head_lnd_diff","diff_ratio_sig_strikes_head_lnd_diff",
                      "diff_avg_cplx_sig_strikes_head_lnd","diff_smt_min_sig_strikes_head_lnd_diff"],
    'Boxing Defense': ["diff_avg_cplx_acc_def_sig_strikes_head_lnd_get","diff_smt_acc_def_sig_strikes_head_lnd_get",
                       "diff_avg_cplx_kd_get","diff_avg_cplx_min_sig_strikes_head_lnd_get"],
    'Kickboxing Attack': ["diff_ratio_min_kd_diff","diff_avg_cplx_kd","diff_ratio_min_sig_strikes_lnd_diff",
                          "diff_smt_sig_strikes_lnd_diff","diff_smt_acc_att_strikes_lnd"],
    'Kickboxing Defense': ["diff_avg_cplx_min_sig_strikes_leg_lnd_get","diff_ratio_lose_ko",
                           "diff_avg_cplx_acc_def_strikes_lnd_get","diff_avg_cplx_acc_def_sig_strikes_leg_lnd_get"],
    'Wrestling Attack': ["diff_avg_cplx_min_td_thr","diff_avg_cplx_min_td_lnd","diff_avg_cplx_td_thr",
                         "diff_ratio_sig_strikes_grnd_thr_diff","diff_ratio_att_td_lnd","diff_ratio_td_thr",
                         "diff_smt_acc_att_td_lnd","diff_avg_cplx_sig_strikes_grnd_thr","diff_smt_acc_att_sig_strikes_grnd_lnd"],
    'Wrestling Defense': ["diff_avg_cplx_sig_strikes_grnd_lnd_get","diff_avg_cplx_cntrl_get",
                          "diff_avg_cplx_min_sig_strikes_grnd_thr_get","diff_smt_acc_def_sig_strikes_grnd_lnd_get"],
    'Grappling Attack': ["diff_smt_min_sub_att","diff_ratio_sub_att_diff","diff_avg_cplx_sub_att","diff_smt_sub_att_diff"],
    'Grappling Defense': ["diff_ratio_min_sub_att_get"]
}

# ======================
# FUNKCE
# ======================

def get_stats_from_row(row, inactive_days):
    data = {}
    data['age'] = -(row['age'] + inactive_days / 365.25)
    for stat in stats:
        if stat == "ratio_reach":
            data[stat] = row.get(stat, 0)
        else:
            val = row.get(stat, 0)
            data[stat] = apply_decay(val, inactive_days)
    data['elo_before'] = apply_decay(row['elo_before1'], inactive_days)
    return data

def build_diff(row1, row2):
    inactive1 = (date_fight_pd - pd.to_datetime(row1['date'])).days
    inactive2 = (date_fight_pd - pd.to_datetime(row2['date'])).days
    s1 = get_stats_from_row(row1, inactive1)
    s2 = get_stats_from_row(row2, inactive2)
    diffs = {f"diff_{k}": s1[k] - s2[k] for k in s1}
    diffs['diff_age'] = s1['age'] - s2['age']
    diffs['diff_elo_before'] = s1['elo_before'] - s2['elo_before']
    return diffs

def build_input_df(fighter1, fighter2):
    row1_df = df_stats.loc[df_stats['fighter1'] == fighter1]
    if row1_df.empty:
        raise ValueError(f"Fighter not found: {fighter1}")
    row1 = row1_df.iloc[0]

    row2_df = df_stats.loc[df_stats['fighter1'] == fighter2]
    if row2_df.empty:
        raise ValueError(f"Fighter not found: {fighter2}")
    row2 = row2_df.iloc[0]

    diffs = build_diff(row1, row2)
    return pd.DataFrame([{c: diffs.get(c, 0) for c in selected_features}])

# ======================
# PUBLIC API
# ======================

def predict_fight(fighter1: str, fighter2: str) -> dict:
    input_1 = build_input_df(fighter1, fighter2)
    input_2 = build_input_df(fighter2, fighter1)

    prob1 = model.predict_proba(input_1)[0]
    prob2 = model.predict_proba(input_2)[0]

    avg_prob_f1 = (prob1[1] + (1 - prob2[1])) / 2
    avg_prob_f2 = (prob1[0] + (1 - prob2[0])) / 2

    if avg_prob_f1 > avg_prob_f2:
        winner, loser, win_prob = fighter1, fighter2, avg_prob_f1
    else:
        winner, loser, win_prob = fighter2, fighter1, avg_prob_f2

    return {
        "winner": winner,
        "win_prob": f"{round(float(win_prob) * 100, 1)}%",
        "loser": loser,
        "lose_prob": f"{round((1 - float(win_prob)) * 100, 1)}%"
    }

def predict_fight_with_shap(fighter1: str, fighter2: str) -> dict:
    # ======================
    # 1) Inputy
    # ======================
    X1 = build_input_df(fighter1, fighter2)
    X2 = build_input_df(fighter2, fighter1)

    # ======================
    # 2) Pravděpodobnosti
    # ======================
    p1 = model.predict_proba(X1)[0]
    p2 = model.predict_proba(X2)[0]

    prob_f1 = (p1[1] + (1 - p2[1])) / 2
    prob_f2 = (p1[0] + (1 - p2[0])) / 2

    if prob_f1 >= prob_f2:
        winner, loser, win_prob = fighter1, fighter2, prob_f1
        winner_is_f1 = True
    else:
        winner, loser, win_prob = fighter2, fighter1, prob_f2
        winner_is_f1 = False

    # ======================
    # 3) SHAP – SPRÁVNÁ EXTRAKCE
    # ======================
    sv1 = explainer.shap_values(X1)
    sv2 = explainer.shap_values(X2)

    # RF → (1, 40, 2)
    if sv1.ndim == 3:
        sv1 = sv1[:, :, 1]
    if sv2.ndim == 3:
        sv2 = sv2[:, :, 1]

    df1 = pd.DataFrame(sv1, columns=selected_features)
    df2 = pd.DataFrame(sv2, columns=selected_features)

    # ======================
    # 4) SYMETRIZACE
    # ======================
    shap_groups = {}
    for group, feats in groups.items():
        v1 = float(df1[feats].sum(axis=1))
        v2 = float(df2[feats].sum(axis=1))
        shap_groups[group] = (v1 - v2) / 2

    # ======================
    # 5) UKOTVENÍ KE VÍTĚZI
    # ======================
    if not winner_is_f1:
        shap_groups = {k: -v for k, v in shap_groups.items()}

    return {
        "winner": winner,
        "win_prob": f"{round(win_prob * 100, 1)}%",
        "loser": loser,
        "lose_prob": f"{round((1 - win_prob) * 100, 1)}%",
        "shap_groups": shap_groups
    }









