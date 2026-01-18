from fastapi import FastAPI, HTTPException
import event_predictor
from predictor import predict_fight, predict_fight_with_shap

app = FastAPI(
    title="UFC Fight Predictor",
    description="API pro predikci UFC zápasů a eventů včetně SHAP hodnot",
    version="1.4"
)

@app.get("/")
def root():
    return {"message": "UFC Predictor API is running"}

@app.get("/predict")
def predict(fighter1: str, fighter2: str):
    if fighter1 == fighter2:
        raise HTTPException(status_code=400, detail="Fighters must be different")
    try:
        return predict_fight(fighter1, fighter2)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict_shap")
def predict_shap(fighter1: str, fighter2: str):
    if fighter1 == fighter2:
        raise HTTPException(status_code=400, detail="Fighters must be different")
    try:
        return predict_fight_with_shap(fighter1, fighter2)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict-event")
def predict_event(save_json: bool = False):
    """
    Endpoint pro predikci celého eventu.
    Volitelně lze uložit do JSON souboru.
    """
    try:
        results = event_predictor.predict_event_with_shap()
        if save_json:
            event_predictor.save_to_json(results)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
