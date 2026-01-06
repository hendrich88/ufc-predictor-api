import os
import requests
import pandas as pd
from datetime import date
from joblib import load

# ======================
# KONFIG: Google Drive FILE_ID
# ======================

JSON_FILE_ID = "1iRyPqdNGYGSUR2hyL22qazNOENFBI7GM"   # tvůj df JSON
MODEL_FILE_ID = "1e6WCzLU2rcQdstwHoREnwgzUa5XWN-kU"  # tvůj model joblib

JSON_FILE = "df_prep_2025-12-17.json"
MODEL_FILE = "rf_model5.joblib"

# ======================
# BEZPEČNÉ STAHOVÁNÍ SOUBORU Z DRIVE
# ======================

def download_file(file_id, filename):
    """Stáhne soubor z Google Drive pokud ještě neexistuje."""
    if not os.path.exists(filename):
        print(f"Stahuji {filename} z Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise Exception(f"Chyba při stahování {filename}, status code: {r.status_code}")

        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        # Kontrola, že soubor není HTML stránka
        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            start = f.read(100)
            if "<!DOCTYPE html>" in start or "<html" in start.lower():
                raise Exception(f"{filename} obsahuje HTML místo JSON/modelu! Zkontroluj FILE_ID a přístup.")

        print(f"{filename} stažen.")

# stáhnout soubory
download_file(JSON_FILE_ID, JSON_FILE)
download_file(MODEL_FILE_ID, MODEL_FILE)

# ======================
# KONSTANTY MODELU
# ======================

DECAY_THRESHOLD = 180
DECAY_RATE = 0.05
MIN_ELO = 0

# ======================
# LOAD DATA + MODEL
# ======================

df_stats = pd.read_json(JSON_FILE, lines=True)
df_stats = df_stats.sort_values(by="date")

model = load(MODEL_FILE)

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

stats = [f.replace("diff_", "", 1) for f in selected_features if f not in ["diff_elo_before"]]

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
    for stat in stats:
        if stat in df.columns:
            try:
                value = df[df["fighter1"] == fighter].iloc[-1][stat]
                data[stat] = value if stat == "ratio_reach" else value * decay_factor
            except IndexError:
                data[stat] = 0
        else:
            data[stat] = 0
    return data


def build_input_df(fighter1, fighter2):
    today = pd.to_datetime(date.today().strftime("%Y-%m-%d"))

    last_date_f1 = pd.to_datetime(df_stats[df_stats["fighter1"] == fighter1].iloc[-1]["date"])
    last_date_f2 = pd.to_datetime(df_stats[df_stats["fighter1"] == fighter2].iloc[-1]["date"])

    inactive_f1 = (today - last_date_f1).days
    inactive_f2 = (today - last_date_f2).days

    stats_f1 = get_stats_for_fighter(df_stats, fighter1, inactive_f1)
    stats_f2 = get_stats_for_fighter(df_stats, fighter2, inactive_f2)

    age_f1 = df_stats[df_stats["fighter1"] == fighter1].iloc[-1]["age"]
    age_f2 = df_stats[df_stats["fighter1"] == fighter2].iloc[-1]["age"]

    active_year_f1 = inactive_f1 / 365.25
    active_year_f2 = inactive_f2 / 365.25

    age_f1 = -(age_f1 + active_year_f1)
    age_f2 = -(age_f2 + active_year_f2)

    elo_f1 = df_stats[df_stats["fighter1"] == fighter1].iloc[-1]["elo_before1"]
    elo_f2 = df_stats[df_stats["fighter1"] == fighter2].iloc[-1]["elo_before1"]

    elo_f1 = apply_decay(elo_f1, inactive_f1)
    elo_f2 = apply_decay(elo_f2, inactive_f2)

    diffs = {f"diff_{stat}": stats_f1.get(stat, 0) - stats_f2.get(stat, 0) for stat in stats}
    diffs.update({"diff_age": age_f1 - age_f2, "diff_elo_before": elo_f1 - elo_f2})

    input_df = pd.DataFrame([[0.0] * len(selected_features)], columns=selected_features)
    for col in input_df.columns:
        if col in diffs:
            input_df.loc[0, col] = diffs[col]

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
        return {"winner": fighter1, "probability": round(float(avg_prob_f1), 3)}
    else:
        return {"winner": fighter2, "probability": round(float(avg_prob_f2), 3)}
