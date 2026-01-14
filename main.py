from fastapi import FastAPI, HTTPException
from predictor import predict_fight, predict_fight_with_shap

app = FastAPI(
    title="UFC Fight Predictor",
    description="API pro predikci UFC zápasů s možností SHAP hodnot",
    version="1.1"
)

@app.get("/")
def root():
    return {"message": "UFC Predictor API is running"}

@app.get("/predict")
def predict(fighter1: str, fighter2: str):
    """
    Endpoint pro základní predikci zápasu (bez SHAP hodnot)
    """
    if fighter1 == fighter2:
        raise HTTPException(status_code=400, detail="Fighters must be different")

    try:
        return predict_fight(fighter1, fighter2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict_shap")
def predict_shap(fighter1: str, fighter2: str):
    """
    Endpoint pro predikci zápasu včetně SHAP hodnot
    """
    if fighter1 == fighter2:
        raise HTTPException(status_code=400, detail="Fighters must be different")

    try:
        return predict_fight_with_shap(fighter1, fighter2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
