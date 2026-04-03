import os
import requests
import pandas as pd
import numpy as np
import json
import importlib  # Nutné pro aktualizaci input.py za běhu
from datetime import date
from joblib import load
import shap

# ======================
# AUTO LOAD & RELOAD INPUT.PY
# ======================
INPUT_URL = "https://raw.githubusercontent.com/hendrich88/data-ufc-predictor/main/input.py"
INPUT_FILE = "input.py"

def refresh_input_file():
    """Stáhne nejnovější verzi input.py a vynutí reload modulu."""
    try:
        r = requests.get(f"{INPUT_URL}?nocache={np.random.random()}", timeout=30)
        r.raise_for_status()
        with open(INPUT_FILE, "w", encoding="utf-8") as f:
            f.write(r.text)
        
        if 'input_mod' in globals():
            importlib.reload(input_mod)
        return True
    except Exception as e:
        print(f"Chyba při stahování input.py: {e}")
        return False

# Prvotní stažení/načtení
if not os.path.exists(INPUT_FILE):
    refresh_input_file()

import input as input_mod

# ======================
# KONFIGURACE CEST
# ======================
JSON_FILE = "df_prep_clean_2026-04-03.json"
AGE_MODEL_URL = "https://github.com/hendrich88/ufc-predictor-api/releases/latest/download/rf_model_age.joblib"
AGE_MODEL_FILE = "rf_model_age.joblib"
MODEL_URL = "https://github.com/hendrich88/ufc-predictor-api/releases/latest/download/rf_calib4.joblib"
MODEL_FILE = "rf_calib4.joblib"

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Stahuji {filename}...")
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"{filename} stažen.")

download_file(MODEL_URL, MODEL_FILE)
download_file(AGE_MODEL_URL, AGE_MODEL_FILE)

# ======================
# NAČTENÍ DAT A MODELŮ
# ======================
if not os.path.exists(JSON_FILE):
    raise FileNotFoundError(f"Missing data file: {JSON_FILE}")

df_stats = pd.read_json(JSON_FILE, lines=True)
df_stats["date"] = pd.to_datetime(df_stats["date"])
df_stats = df_stats.sort_values(by="date")

model = load(MODEL_FILE)
model_age = load(AGE_MODEL_FILE)
model_required_features = list(model.feature_names_in_)
explainer = shap.TreeExplainer(model.estimator)

# ======================
# POMOCNÉ FUNKCE (DECAY & AGE)
# ======================
DECAY_THRESHOLD = 180
DECAY_RATE = 0.05

def apply_decay(value, inactive_days, is_negative=False):
    """
    Upravená logika chřadnutí:
    - Pozitivní stats: klesají k nule (násobení < 1.0)
    - Negativní stats (_get): klesají hlouběji do mínusu (násobení > 1.0)
    - Damage (0-1): roste k jedničce (násobení > 1.0)
    """
    if inactive_days <= DECAY_THRESHOLD:
        return value
    
    t = (inactive_days - DECAY_THRESHOLD) / 365
    factor = max(0.7, 1 - DECAY_RATE * (t ** 2))
    
    if is_negative:
        multiplier = 2 - factor # Např. 1.1 až 1.3
        new_val = value * multiplier
        # Pojistka pro damage 0-1
        if 0 <= value <= 1:
            return min(1.0, new_val)
        return new_val
    else:
        # Pozitivní věci klesají k nule
        return value * factor

def calculate_age_index(fighter, df_stats):
    try:
        f_row = df_stats[df_stats['fighter1'] == fighter].sort_values('date').tail(1)
        if f_row.empty: return 0.5
        last_date = pd.to_datetime(f_row['date'].iloc[0])
        years_passed = (pd.to_datetime(date.today()) - last_date).days / 365.25
        input_data = pd.DataFrame([{
            'weight_kg': float(f_row['weight_kg'].iloc[0]),
            'ufc_age': float(f_row['ufc_age'].iloc[0]) + years_passed,
            'adj_age': (float(f_row['age'].iloc[0]) + years_passed) - float(f_row['glob_avg_age'].iloc[0]),
            'fighter_fight_number': int(f_row['fighter_fight_number'].iloc[0]) + 1
        }])
        return model_age.predict_proba(input_data[['weight_kg', 'ufc_age', 'adj_age', 'fighter_fight_number']])[0][1]
    except:
        return 0.5

stats_to_decay = [f.replace("diff_", "", 1) for f in model_required_features 
                  if f not in ["diff_age_index", "diff_elo_before", "diff_ratio_reach"]]

def get_stats_from_row(row, inactive_days):
    data = {}
    damage_stats = ["avg_self_damage", "avg_balance_damage"]
    for stat in stats_to_decay:
        val = row.get(stat, 0)
        # Identifikujeme negativní metriky (_get jsou záporné, damage 0-1)
        is_neg = stat.endswith('_get') or stat in damage_stats
        data[stat] = apply_decay(val, inactive_days, is_negative=is_neg)
    
    # ELO je pozitivní metrika -> klesá k nule
    data['elo_before'] = apply_decay(row.get('elo_before1', 0), inactive_days, is_negative=False)
    
    if 'ratio_reach' in row: data['ratio_reach'] = row['ratio_reach']
    return data

def build_diff(row1, row2, f1_name, f2_name, df_stats):
    inactive1 = (pd.to_datetime(date.today()) - pd.to_datetime(row1['date'])).days
    inactive2 = (pd.to_datetime(date.today()) - pd.to_datetime(row2['date'])).days
    s1, s2 = get_stats_from_row(row1, inactive1), get_stats_from_row(row2, inactive2)
    diffs = {f"diff_{k}": s1[k] - s2[k] for k in s1}
    diffs['diff_age_index'] = calculate_age_index(f1_name, df_stats) - calculate_age_index(f2_name, df_stats)
    return diffs

def make_input_df(diffs):
    data = {c: diffs.get(c, 0) for c in model_required_features}
    return pd.DataFrame([data])[model_required_features]

# ======================
# SHAP ANALÝZA
# ======================
groups = {
    'Age Index (AI)': ["diff_age_index"],
    'Win/Loss Rates': ["diff_win_rate","diff_lose_rate"],
    'Damage Resistance (AI)': ["diff_avg_balance_damage","diff_avg_self_damage"],
    'Reach': ["diff_ratio_reach"],
    'Ranking (AI)': ["diff_elo_before"],
    'Boxing Attack': ["diff_smt_sig_strikes_head_lnd_diff","diff_ratio_kd_diff","diff_avg_cplx_min_kd"],
    'Boxing Defense': ["diff_avg_cplx_acc_def_sig_strikes_head_lnd_get","diff_ratio_def_sig_strikes_head_lnd_get","diff_avg_cplx_kd_get"],
    'Kickboxing Attack': ["diff_smt_acc_att_sig_strikes_body_lnd","diff_smt_acc_att_sig_strikes_dist_lnd","diff_ratio_att_sig_strikes_body_lnd"],
    'Kickboxing Defense': ["diff_avg_cplx_sig_strikes_body_thr_get","diff_ratio_def_sig_strikes_lnd_get"],
    'Wrestling Attack': ["diff_avg_cplx_min_cntrl","diff_avg_cplx_min_td_lnd"],
    'Wrestling Defense': ["diff_avg_cplx_min_td_thr_get","diff_avg_cntrl_get"],
    'Grappling Attack': ["diff_smt_rev", "diff_ratio_sub_att_diff","diff_ratio_min_rev_diff","diff_avg_cplx_min_rev","diff_avg_cplx_min_sub_att","diff_avg_cplx_sub_att"],
    'Complex Dominance (AI)': ["diff_avg_cplx_dom_total","diff_avg_dom_total"],
    'Striking Dominance (AI)': ["diff_avg_cplx_dom_stance","diff_avg_dom_stance"],
    'Ground Dominance (AI)': ["diff_avg_cplx_dom_ground","diff_avg_dom_ground"],
    'Style Dominance (AI)': ["diff_avg_dom_press"]
}

def extract_shap_impact(input_df):
    sv = explainer.shap_values(input_df)
    s = sv[1][0] if isinstance(sv, list) else (sv[0, :, 1] if sv.ndim == 3 else sv[0])
    group_res = {g_name: round(sum([s[model_required_features.index(f)] for f in g_feats if f in model_required_features]) * 100, 2) 
                 for g_name, g_feats in groups.items()}
    return dict(sorted(group_res.items(), key=lambda x: abs(x[1]), reverse=True))

# ======================
# PREDIKCE ZÁPASU (API)
# ======================
def predict_fight_with_shap(f1, f2, o1=2.0, o2=2.0):
    try:
        r1 = df_stats[df_stats['fighter1'] == f1].sort_values('date').tail(1)
        r2 = df_stats[df_stats['fighter1'] == f2].sort_values('date').tail(1)

        if r1.empty or r2.empty:
            return {"error": f"Bojovník nebyl nalezen: {f1 if r1.empty else f2}"}

        in1 = make_input_df(build_diff(r1.iloc[0], r2.iloc[0], f1, f2, df_stats))
        in2 = make_input_df(build_diff(r2.iloc[0], r1.iloc[0], f2, f1, df_stats))
        
        p1, p2 = model.predict_proba(in1)[0], model.predict_proba(in2)[0]
        avg_p1, avg_p2 = (p1[1] + (1 - p2[1])) / 2, (p1[0] + (1 - p2[0])) / 2

        if avg_p1 > avg_p2:
            winner, win_p, loser, lose_p = f1, avg_p1, f2, avg_p2
            w_odds, l_odds, shap_in = o1, o2, in1
        else:
            winner, win_p, loser, lose_p = f2, avg_p2, f1, avg_p1
            w_odds, l_odds, shap_in = o2, o1, in2

        win_odds_pct = (1 / float(w_odds)) * 100
        lose_odds_pct = (1 / float(l_odds)) * 100
        edge_val = (win_p - (1 / float(w_odds))) / (1 / float(w_odds)) * 100

        return {
            "winner": winner,
            "win_prob": f"{round(win_p * 100, 1)}%",
            "win_odds": f"{round(win_odds_pct, 1)}%",
            "loser": loser,
            "lose_prob": f"{round(lose_p * 100, 1)}%",
            "lose_odds": f"{round(lose_odds_pct, 1)}%",
            "fair_odds": round(1 / win_p, 2),
            "edge": f"{round(edge_val, 1)}%",
            "shap_groups": extract_shap_impact(shap_in),
            "hit": None
        }
    except Exception as e:
        return {"error": f"Interní chyba: {str(e)}"}

# ======================
# HLAVNÍ PREDIKCE EVENTU
# ======================
def predict_event_with_shap_all():
    refresh_input_file()
    importlib.reload(input_mod)

    results = {
        "event_date": input_mod.event_date, "event": input_mod.event,
        "event_accuracy": input_mod.event_accuracy, "event_roi": input_mod.event_roi,
        "event_fights": 0, "fights": []
    }
    
    for idx, (f1, f2) in enumerate(zip(input_mod.event_fighters1, input_mod.event_fighters2)):
        try:
            res = predict_fight_with_shap(f1, f2, input_mod.odds_fighters1[idx], input_mod.odds_fighters2[idx])
            
            if "error" in res: continue

            win_p_float = float(res["win_prob"].replace("%", ""))
            edge_float = float(res["edge"].replace("%", ""))
            
            if win_p_float < input_mod.limit_pred or edge_float < input_mod.edge: 
                continue
            
            res["hit"] = input_mod.hit[idx]
            results["fights"].append(res)
        except: continue

    results["event_fights"] = len(results["fights"])
    return results

if __name__ == "__main__":
    print(json.dumps(predict_event_with_shap_all(), indent=4, ensure_ascii=False))
