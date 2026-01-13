import os
import requests
import pandas as pd
from datetime import date
from joblib import load

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
# KONSTANTY MODELU
# ======================

DECAY_THRESHOLD = 180       # dní bez penalizace
DECAY_RATE = 0.05           # max ~10 % při dlouhé pauze
MIN_VALUE = 0               # spodní hranice pro všechny statistiky

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

stats = [f.replace("diff_", "", 1) for f in selected_features if f not in ["diff_age", "diff_elo_before"]]

# ======================
# FUNKCE
# ======================

def quadratic_decay(value, inactive_days):
    """
    Kvadratický decay pro všechny statistiky kromě age.
    """
    t = max(0, inactive_days - DECAY_THRESHOLD) / 365
    factor = 1 - DECAY_RATE * (t ** 2)
    factor = max(0.7, factor)
    return max(MIN_VALUE, value * factor)

def get_stats_from_row(row, inactive_days):
    """
    Vrací všechny statistiky bojovníka s kvadratickým decay.
    Age není kvadraticky penalizováno (lineární).
    """
    data = {}
    # Age lineární
    data['age'] = -(row['age'] + inactive_days / 365.25)
    # ELO s kvadratickým decay
    data['elo_before'] = quadratic_decay(row['elo_before1'], inactive_days)
    # Ostatní statistiky s kvadratickým decay
    for stat in stats:
        val = row.get(stat, 0)
        data[stat] = quadratic_decay(val, inactive_days)
    return data

def build_diff(row1, row2):
    today = pd.to_datetime(date.today())
    inactive1 = (today - pd.to_datetime(row1['date'])).days
    inactive2 = (today - pd.to_datetime(row2['date'])).days

    s1 = get_stats_from_row(row1, inactive1)
    s2 = get_stats_from_row(row2, inactive2)

    diffs = {f"diff_{k}": s1[k] - s2[k] for k in s1}

    # Age diff lineární
    diffs['diff_age'] = s1['age'] - s2['age']

    # ELO diff s kvadratickým decay
    diffs['diff_elo_before'] = s1['elo_before'] - s2['elo_before']

    return diffs

def build_input_df(fighter1, fighter2):
    row1 = df_stats.loc[df_stats['fighter1'] == fighter1].iloc[0]
    row2 = df_stats.loc[df_stats['fighter1'] == fighter2].iloc[0]

    diffs = build_diff(row1, row2)
    input_df = pd.DataFrame([{c: diffs.get(c, 0) for c in selected_features}])
    return input_df

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
        winner = fighter1
        loser = fighter2
        win_prob = avg_prob_f1
    else:
        winner = fighter2
        loser = fighter1
        win_prob = avg_prob_f2

    return {
        "winner": winner,
        "win_prob": f"{round(float(win_prob) * 100, 1)}%",
        "loser": loser,
        "lose_prob": f"{round((1 - float(win_prob)) * 100, 1)}%"
    }
