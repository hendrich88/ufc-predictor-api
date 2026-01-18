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

    for idx, (f1, f2) in enumerate(zip(event_fighters1, event_fighters2), start=1):
        print(f"\n[{idx}/{len(event_fighters1)}] {f1} vs {f2}")

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

            print(f"✔ Winner: {result['winner']} ({result['win_prob']})")

        except Exception as e:
            print(f"❌ Chyba: {e}")

            event_results["fights"].append({
                "fighter1": f1,
                "fighter2": f2,
                "error": str(e)
            })

    return event_results


def save_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    results = predict_event_with_shap()
    save_to_json(results, OUTPUT_FILE)

    print(f"\n✅ Event uložen do souboru: {OUTPUT_FILE}")
