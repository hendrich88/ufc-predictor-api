from fastapi import FastAPI, HTTPException
from predictor import predict_fight, predict_fight_with_shap

app = FastAPI(
    title="UFC Fight Predictor",
    description="API pro predikci UFC zápasů s možností SHAP hodnot",
    version="1.2"
)

@app.get("/")
def root():
    """Základní endpoint pro kontrolu, zda API běží"""
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
    except ValueError as ve:
        # Nenalezený bojovník
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        # Jiné neočekávané chyby
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
    except ValueError as ve:
        # Nenalezený bojovník
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        # Jiné neočekávané chyby
        raise HTTPException(status_code=500, detail=str(e))
