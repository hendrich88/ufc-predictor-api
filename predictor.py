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
        # Použití nocache parametru, aby GitHub neposílal starou verzi
        r = requests.get(f"{INPUT_URL}?nocache={np.random.random()}", timeout=30)
        r.raise_for_status()
        with open(INPUT_FILE, "w", encoding="utf-8") as f:
            f.write(r.text)
        
        # Pokud už byl modul načten, provedeme reload
        if 'input' in globals() or 'input' in os.sys.modules:
            importlib.reload(input_mod)
        return True
    except Exception as e:
        print(f"Chyba při stahování input.py: {e}")
        return False

# Prvotní stažení/načtení
if not os.path.exists(INPUT_FILE):
    refresh_input_file()

# Importujeme jako objekt, abychom mohli přistupovat k čerstvým datům přes tečku
import input as input_mod

# ======================
# KONFIGURACE CEST
# ======================
JSON_FILE = "df_prep_clean_2026-03-02.json"
AGE_MODEL_URL = "https://github.com/hendrich88/ufc-predictor-api/releases/latest/download/rf_model_age.joblib"
AGE_MODEL_FILE = "rf_model_age.joblib"
MODEL_URL = "https://github.com/hendrich88/ufc-predictor-api/releases/latest/download/rf_calib3.joblib"
MODEL_FILE = "rf_calib3.joblib"

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

# SHAP explainer
explainer = shap.TreeExplainer(model.estimator)

# ======================
# POMOCNÉ FUNKCE
# ======================
DECAY_THRESHOLD = 180
DECAY_RATE = 0.05
MIN_VALUE = 0

def apply_decay(value, inactive_days):
    if inactive_days <= DECAY_THRESHOLD:
        return value
    t = (inactive_days - DECAY_THRESHOLD) / 365
    factor = max(0.7, 1 - DECAY_RATE * (t ** 2))
    return max(MIN_VALUE, value * factor)

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
        cols = ['weight_kg', 'ufc_age', 'adj_age', 'fighter_fight_number']
        return model_age.predict_proba(input_data[cols])[0][1]
    except:
        return 0.5

stats_to_decay = [f.replace("diff_", "", 1) for f in model_required_features 
                  if f not in ["diff_age_index", "diff_elo_before", "diff_ratio_reach", "diff_avg_self_damage", "diff_avg_balance_damage"]]

def get_stats_from_row(row, inactive_days):
    data = {}
    for stat in stats_to_decay:
        data[stat] = apply_decay(row.get(stat, 0), inactive_days)
    data['elo_before'] = apply_decay(row['elo_before1'], inactive_days)
    if 'ratio_reach' in row: data['ratio_reach'] = row['ratio_reach']
    return data

def build_diff(row1, row2, f1_name, f2_name, df_stats):
    inactive1 = (pd.to_datetime(date.today()) - pd.to_datetime(row1['date'])).days
    inactive2 = (pd.to_datetime(date.today()) - pd.to_datetime(row2['date'])).days
    s1 = get_stats_from_row(row1, inactive1)
    s2 = get_stats_from_row(row2, inactive2)
    diffs = {f"diff_{k}": s1[k] - s2[k] for k in s1}
    diffs['diff_age_index'] = calculate_age_index(f1_name, df_stats) - calculate_age_index(f2_name, df_stats)
    diffs['diff_avg_self_damage'] = apply_decay(row1.get('avg_self_damage', 0), inactive1) - \
                                   apply_decay(row2.get('avg_self_damage', 0), inactive2)
    diffs['diff_avg_balance_damage'] = apply_decay(row1.get('avg_balance_damage', 0), inactive1) - \
                                      apply_decay(row2.get('avg_balance_damage', 0), inactive2)
    return diffs

def make_input_df(diffs):
    data = {c: diffs.get(c, 0) for c in model_required_features}
    return pd.DataFrame([data])[model_required_features]

# ======================
# SHAP ANALÝZA
# ======================
groups = {
    'Age Index (AI)': ["diff_age_index"],
    'Win/Lose Rates': ["diff_avg_self_damage","diff_lose_rate"],
    'Damage Resistance (AI)': ["diff_win_rate","diff_avg_balance_damage"],
    'Reach': ["diff_ratio_reach"],
    'Ranking (AI)': ["diff_elo_before"],
    'Boxing Attack': ["diff_smt_sig_strikes_head_lnd_diff","diff_ratio_kd_diff","diff_avg_cplx_min_kd"],
    'Boxing Defense': ["diff_avg_cplx_acc_def_sig_strikes_head_lnd_get","diff_ratio_def_sig_strikes_head_lnd_get","diff_avg_cplx_kd_get"],
    'Kickboxing Attack': ["diff_smt_acc_att_sig_strikes_body_lnd","diff_smt_acc_att_sig_strikes_dist_lnd","diff_ratio_def_sig_strikes_lnd_get","diff_ratio_att_sig_strikes_body_lnd"],
    'Kickboxing Defense': ["diff_avg_cplx_sig_strikes_body_thr_get"],
    'Wrestling Attack': ["diff_avg_cplx_min_cntrl","diff_avg_cplx_min_td_lnd"],
    'Wrestling Defense': ["diff_avg_cplx_min_td_thr_get","diff_avg_cntrl_get"],
    'Grappling Attack': ["diff_smt_rev", "diff_ratio_sub_att_diff","diff_ratio_min_rev_diff","diff_avg_cplx_min_rev","diff_avg_cplx_min_sub_att","diff_avg_cplx_sub_att"],
    'Complex Dominance (AI)': ["diff_avg_cplx_dom_total","diff_avg_dom_total"],
    'Striking Dominance (AI)': ["diff_avg_cplx_dom_stance","diff_avg_dom_stance"],
    'Ground Dominance (AI)': ["diff_avg_cplx_dom_ground","diff_avg_dom_ground"]
}

def extract_shap_impact(input_df):
    sv = explainer.shap_values(input_df)
    if isinstance(sv, list): 
        s = sv[1][0]
    else: 
        s = sv[0, :, 1] if sv.ndim == 3 else sv[0]
    
    group_res = {}
    for g_name, g_feats in groups.items():
        val = sum([s[model_required_features.index(f)] for f in g_feats if f in model_required_features])
        group_res[g_name] = round(val * 100, 2)
    
    return dict(sorted(group_res.items(), key=lambda x: abs(x[1]), reverse=True))

# ======================
# HLAVNÍ PREDIKCE EVENTU
# ======================
def predict_event_with_shap_all():
    # Vynutíme čerstvá data z input.py před každým výpočtem eventu
    refresh_input_file()
    importlib.reload(input_mod)

    results = {
        "event_date": input_mod.event_date, 
        "event": input_mod.event,
        "event_accuracy": input_mod.event_accuracy, 
        "event_roi": input_mod.event_roi,
        "event_fights": 0,
        "fights": []
    }
    
    valid_fights_count = 0
    # Iterujeme přes data přímo z reloadnutého modulu
    f1_list = input_mod.event_fighters1
    f2_list = input_mod.event_fighters2
    o1_list = input_mod.odds_fighters1
    o2_list = input_mod.odds_fighters2
    hits = input_mod.hit

    for idx, (f1, f2) in enumerate(zip(f1_list, f2_list)):
        try:
            r1_rows = df_stats[df_stats['fighter1'] == f1].sort_values('date').tail(1)
            r2_rows = df_stats[df_stats['fighter1'] == f2].sort_values('date').tail(1)
            
            if r1_rows.empty or r2_rows.empty:
                continue
            
            row1, row2 = r1_rows.iloc[0], r2_rows.iloc[0]
            f1_fights, f2_fights = int(row1['fighter_fight_number'] + 1), int(row2['fighter_fight_number'] + 1)
            
            # Symetrie
            in1 = make_input_df(build_diff(row1, row2, f1, f2, df_stats))
            in2 = make_input_df(build_diff(row2, row1, f2, f1, df_stats))
            p1, p2 = model.predict_proba(in1)[0], model.predict_proba(in2)[0]
            
            avg_p1 = (p1[1] + (1 - p2[1])) / 2
            avg_p2 = (p1[0] + (1 - p2[0])) / 2
            
            # Určení vítěze/poraženého
            if avg_p1 > avg_p2:
                winner, win_prob, loser, lose_prob = f1, avg_p1, f2, (1 - avg_p1)
                w_odds_raw, l_odds_raw = o1_list[idx], o2_list[idx]
                shap_input = in1
            else:
                winner, win_prob, loser, lose_prob = f2, avg_p2, f1, (1 - avg_p2)
                w_odds_raw, l_odds_raw = o2_list[idx], o1_list[idx]
                shap_input = in2

            # Filtry z input.py
            if (win_prob * 100) < input_mod.limit_pred: continue
            if winner == f1 and (f1_fights < input_mod.min_winner_fights or f2_fights < input_mod.min_loser_fights): continue
            if winner == f2 and (f2_fights < input_mod.min_winner_fights or f1_fights < input_mod.min_loser_fights): continue

            # Výpočet ROI klíčů
            win_odds_pct = f"{round((1 / w_odds_raw) * 100, 1)}%"
            lose_odds_pct = f"{round((1 / l_odds_raw) * 100, 1)}%"
            edge_val = (win_prob - (1 / w_odds_raw)) / (1 / w_odds_raw) * 100

            results["fights"].append({
                "winner": winner,
                "win_prob": f"{round(win_prob * 100, 1)}%",
                "win_odds": win_odds_pct,
                "loser": loser,
                "lose_prob": f"{round(lose_prob * 100, 1)}%",
                "lose_odds": lose_odds_pct,
                "fair_odds": round(1 / win_prob, 2),
                "edge": f"{round(edge_val, 1)}%",
                "shap_groups": extract_shap_impact(shap_input),
                "hit": hits[idx]
            })
            valid_fights_count += 1
        except:
            continue

    results["event_fights"] = valid_fights_count
    return results
