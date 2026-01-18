import json
from datetime import datetime
from predictor import predict_fight_with_shap
from input import event_fighters1, event_fighters2

OUTPUT_FILE = "event_predictions.json"

def predict_event_with_shap():
    if len(event_fighters1) != len(event_fighters2):
        raise ValueError("event_fighters1 a event_fighters2 nemají stejnou délku")

    event_results = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_fights": len(event_fighters1),
        "fights": []
    }

    for f1, f2 in zip(event_fighters1, event_fighters2):
        try:
            result = predict_fight_with_shap(f1, f2)
            fight_result = {
                "fighter1": f1,
                "fighter2": f2,
                "winner": result["winner"],
                "win_prob": result["win_prob"],
                "loser": result["loser"],
                "lose_prob": result["lose_prob"],
                "shap_groups": result["shap_groups"]
            }
            event_results["fights"].append(fight_result)
        except Exception as e:
            event_results["fights"].append({
                "fighter1": f1,
                "fighter2": f2,
                "error": str(e)
            })

    return event_results


def save_to_json(data, filename=OUTPUT_FILE):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
