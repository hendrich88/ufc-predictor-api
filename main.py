from fastapi import FastAPI, HTTPException
from predictor import (
    predict_fight,
    predict_fight_with_shap,
    predict_event_with_shap_all,
    save_event_to_json
)

app = FastAPI(
    title="UFC Fight Predictor",
    description="API pro predikci UFC zápasů a eventů včetně SHAP hodnot",
    version="1.5"
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
    try:
        results = predict_event_with_shap_all()
        if save_json:
            save_event_to_json(results)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
