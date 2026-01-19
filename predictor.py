# ======================
# AUTO LOAD INPUT.PY
# ======================

INPUT_URL = "https://raw.githubusercontent.com/hendrich88/input-ufc-predictor-api/main/input.py"
INPUT_FILE = "input.py"

import os
import requests

if not os.path.exists(INPUT_FILE):
    r = requests.get(INPUT_URL, timeout=30)
    r.raise_for_status()
    with open(INPUT_FILE, "w", encoding="utf-8") as f:
        f.write(r.text)

import os
import requests
import pandas as pd
from datetime import date
from joblib import load
import shap
import numpy as np
import json

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
    'MMAI Score': ["diff_elo_before"],
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
# INPUT + BUILD
# ======================

def get_stats_from_row(row, inactive_days):
    data = {}
    data['age'] = -(row['age'] + inactive_days / 365.25)
    for stat in stats:
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
# PREDIKCE S MODELEM
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
        "win_prob": f"{round(win_prob * 100, 1)}%",
        "loser": loser,
        "lose_prob": f"{round((1 - win_prob) * 100, 1)}%"
    }

# ======================
# PREDIKCE S SHAP
# ======================

def predict_fight_with_shap(f1: str, f2: str) -> dict:
    input_1 = build_input_df(f1, f2)
    input_2 = build_input_df(f2, f1)

    prob1 = model.predict_proba(input_1)[0]
    prob2 = model.predict_proba(input_2)[0]

    avg_prob_f1 = (prob1[1] + (1 - prob2[1])) / 2
    avg_prob_f2 = (prob1[0] + (1 - prob2[0])) / 2

    if avg_prob_f1 >= avg_prob_f2:
        winner, loser = f1, f2
        win_prob = avg_prob_f1
        win_input, lose_input = input_1, input_2
    else:
        winner, loser = f2, f1
        win_prob = avg_prob_f2
        win_input, lose_input = input_2, input_1

    # SHAP values
    def extract_class1_shap(x):
        sv = explainer.shap_values(x)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        if sv.ndim != 2:
            raise ValueError(f"Unexpected SHAP shape: {sv.shape}")
        return sv[0]

    shap_win = extract_class1_shap(win_input)
    shap_lose = extract_class1_shap(lose_input)

    shap_win_df = pd.DataFrame([shap_win], columns=selected_features)
    shap_lose_df = pd.DataFrame([shap_lose], columns=selected_features)

    shap_groups = {}
    for group_name, features in groups.items():
        w = shap_win_df[features].sum(axis=1).iloc[0]
        l = shap_lose_df[features].sum(axis=1).iloc[0]
        shap_groups[group_name] = (w - l) / 2

    if sum(shap_groups.values()) < 0:
        shap_groups = {k: -v for k, v in shap_groups.items()}

    shap_groups = {k: round(v * 100, 2) for k, v in shap_groups.items()}
    shap_groups = dict(sorted(shap_groups.items(), key=lambda x: abs(x[1]), reverse=True))

    return {
        "winner": winner,
        "win_prob": f"{round(win_prob * 100, 1)}%",
        "loser": loser,
        "lose_prob": f"{round((1 - win_prob) * 100, 1)}%",
        "shap_groups": shap_groups
    }

# ======================
# PREDIKCE CELÉHO EVENTU
# ======================

from input import event_fighters1, event_fighters2, odds_fighters1, odds_fighters2, hit as default_hit, event_date, event, event_accuracy, event_roi

def predict_event_with_shap_all():
    if len(event_fighters1) != len(event_fighters2):
        raise ValueError("event_fighters1 a event_fighters2 nemají stejnou délku")

    results = {
        "event_date": event_date,
        "event": event,
        "event_accuracy": event_accuracy,
        "event_roi": event_roi,
        "event_fights": len(event_fighters1),
        "fights": []
    }

    for idx, (f1, f2) in enumerate(zip(event_fighters1, event_fighters2)):
        try:
            # 1️⃣ Predikce vítěze
            res = predict_fight_with_shap(f1, f2)
            winner = res["winner"]
            loser = res["loser"]

            # 2️⃣ Přiřazení správných odds podle skutečného vítěze
            if winner == f1:
                win_odds_value = odds_fighters1[idx]
                lose_odds_value = odds_fighters2[idx]
            elif winner == f2:
                win_odds_value = odds_fighters2[idx]
                lose_odds_value = odds_fighters1[idx]
            else:
                # Pokud model vrátil někoho, kdo není ani f1 ani f2 (bezpečnostní kontrola)
                raise ValueError(f"Winner {winner} není ani f1 ani f2")

            # 3️⃣ Převedení odds na procenta
            res["win_odds"] = f"{round((1 / win_odds_value) * 100, 1)}%"
            res["lose_odds"] = f"{round((1 / lose_odds_value) * 100, 1)}%"
            res["hit"] = default_hit[idx]

            results["fights"].append(res)

        except Exception as e:
            results["fights"].append({
                "fighter1": f1,
                "fighter2": f2,
                "error": str(e),
                "hit": default_hit[idx]
            })

    return results

def save_event_to_json(data, filename="event_predictions.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


