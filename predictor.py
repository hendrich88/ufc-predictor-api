import os
import requests
import pandas as pd
from datetime import date
from joblib import load

# ======================
# KONFIG
# ======================

# 🔹 JSON je teď LOKÁLNÍ v repozitáři
JSON_FILE = "df_prep_clean_2026-01-08.json"

# 🔹 Model zůstává z GitHub Releases
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

# stáhnout model při startu
download_file(MODEL_URL, MODEL_FILE)

# ======================
# KONSTANTY MODELU
# ======================

DECAY_THRESHOLD = 180
DECAY_RATE = 0.05
MIN_ELO = 0

# ======================
# LOAD DATA + MODEL
# ======================

if not os.path.exists(JSON_FILE):
    raise FileNotFoundError(f"Missing data file: {JSON_FILE}")

df_stats = pd.read_json(JSON_FILE, lines=True)
df_stats["date"] = pd.to_datetime(df_stats["date"])
df_stats = df_stats.sort_values(by="date")

model = load(MODEL_FILE)

# ======================
# FEATURES
# ======================

selected_features = [
    "diff_age",
    "diff_elo_before",
    "diff_ratio_min_sig_strikes_head_lnd_diff",
    "diff_win_rate",
    "diff_smt_min_sub_att",
    "diff_ratio_min_sig_strikes_lnd_diff",
    "diff_smt_sig_strikes_lnd_diff",
    "diff_lose_rate",
    "diff_ratio_min_sub_att_get",
    "diff_avg_cplx_kd",
    "diff_avg_cplx_min_td_thr",
    "diff_ratio_sub_att_diff",
    "diff_avg_cplx_acc_def_sig_strikes_head_lnd_get",
    "diff_avg_cplx_sub_att",
    "diff_ratio_sig_strikes_head_lnd_diff",
    "diff_avg_cplx_sig_strikes_grnd_lnd_get",
    "diff_smt_acc_def_sig_strikes_head_lnd_get",
    "diff_avg_cplx_kd_get",
    "diff_ratio_min_kd_diff",
    "diff_avg_cplx_min_sig_strikes_head_lnd_get",
    "diff_avg_cplx_min_td_lnd",
    "diff_avg_cplx_sig_strikes_head_lnd",
    "diff_avg_cplx_min_sig_strikes_leg_lnd_get",
    "diff_avg_cplx_td_thr",
    "diff_smt_acc_att_strikes_lnd",
    "diff_ratio_lose_ko",
    "diff_avg_cplx_cntrl_get",
    "diff_ratio_sig_strikes_grnd_thr_diff",
    "diff_ratio_att_td_lnd",
    "diff_avg_cplx_acc_def_strikes_lnd_get",
    "diff_avg_cplx_min_sig_strikes_grnd_thr_get",
    "diff_smt_min_sig_strikes_head_lnd_diff",
    "diff_ratio_td_thr",
    "diff_smt_acc_att_td_lnd",
    "diff_avg_cplx_sig_strikes_grnd_thr",
    "diff_smt_acc_att_sig_strikes_grnd_lnd",
    "diff_smt_sub_att_diff",
    "diff_smt_acc_def_sig_strikes_grnd_lnd_get",
    "diff_ratio_reach",
    "diff_avg_cplx_acc_def_sig_strikes_leg_lnd_get"
]

stats = [f.replace("diff_", "", 1) for f in selected_features if f != "diff_elo_before"]

# ======================
# FUNKCE
# ======================

def apply_decay(elo, inactive_days):
    if inactive_days <= DECAY_THRESHOLD:
        return elo
    t = (inactive_days - DECAY_THRESHOLD) / 365
    factor = 1 - DECAY_RATE * (t ** 2)
    factor = max(0.7, factor)
    return max(MIN_ELO, elo * factor)


def get_stats_for_fighter(df, fighter, inactive_days):
    data = {}
    decay_factor = 1 - DECAY_RATE * max(0, (inactive_days - DECAY_THRESHOLD) / 365) ** 2
    decay_factor = max(0.7, decay_factor)

    fighter_df = df[df["fighter1"] == fighter]

    if fighter_df.empty:
        return {stat: 0 for stat in stats}

    last_row = fighter_df.iloc[-1]

    for stat in stats:
        value = last_row.get(stat, 0)
        data[stat] = value if stat == "ratio_reach" else value * decay_factor

    return data


def build_input_df(fighter1, fighter2):
    today = pd.to_datetime(date.today())

    f1_df = df_stats[df_stats["fighter1"] == fighter1]
    f2_df = df_stats[df_stats["fighter1"] == fighter2]

    if f1_df.empty or f2_df.empty:
        raise ValueError("One or both fighters not found in dataset")

    last_f1 = f1_df.iloc[-1]
    last_f2 = f2_df.iloc[-1]

    inactive_f1 = (today - last_f1["date"]).days
    inactive_f2 = (today - last_f2["date"]).days

    stats_f1 = get_stats_for_fighter(df_stats, fighter1, inactive_f1)
    stats_f2 = get_stats_for_fighter(df_stats, fighter2, inactive_f2)

    age_f1 = -(last_f1["age"] + inactive_f1 / 365.25)
    age_f2 = -(last_f2["age"] + inactive_f2 / 365.25)

    elo_f1 = apply_decay(last_f1["elo_before1"], inactive_f1)
    elo_f2 = apply_decay(last_f2["elo_before1"], inactive_f2)

    diffs = {f"diff_{stat}": stats_f1[stat] - stats_f2[stat] for stat in stats}
    diffs.update({
        "diff_age": age_f1 - age_f2,
        "diff_elo_before": elo_f1 - elo_f2
    })

    input_df = pd.DataFrame(0.0, index=[0], columns=selected_features)

    for col, val in diffs.items():
        if col in input_df.columns:
            input_df.at[0, col] = val

    return input_df

# ======================
# PUBLIC API FUNKCE
# ======================

def predict_fight(fighter1: str, fighter2: str) -> dict:
    input_1 = build_input_df(fighter1, fighter2)
    input_2 = build_input_df(fighter2, fighter1)

    prob1 = model.predict_proba(input_1)[0]
    prob2 = model.predict_proba(input_2)[0]

    avg_prob_f1 = (prob1[1] + (1 - prob2[1])) / 2
    avg_prob_f2 = (prob1[0] + (1 - prob2[0])) / 2

    if avg_prob_f1 > avg_prob_f2:
        return {
            "winner": fighter1,
            "probability": round(float(avg_prob_f1), 3)
        }
    else:
        return {
            "winner": fighter2,
            "probability": round(float(avg_prob_f2), 3)
        }
