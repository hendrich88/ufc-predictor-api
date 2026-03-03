import os
from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI(title="UFC Predictor API")

# Importy z predictor.py dáme DOVNITŘ funkcí, aby se nespouštěly při startu
# To umožní aplikaci okamžitě otevřít port

@app.get("/")
def root():
    return {"message": "API je online. Modely se stahují na pozadí nebo při prvním volání."}

@app.get("/predict-event")
def predict_event(save_json: bool = False):
    try:
        # Import se provede až tady - Render už mezitím aplikaci schválil jako běžící
        from predictor import predict_event_with_shap_all, save_event_to_json
        results = predict_event_with_shap_all()
        if save_json:
            save_event_to_json(results)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chyba při predikci: {str(e)}")

@app.get("/predict_shap")
def predict_shap(fighter1: str, fighter2: str):
    try:
        from predictor import predict_fight_with_shap
        return predict_fight_with_shap(fighter1, fighter2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
