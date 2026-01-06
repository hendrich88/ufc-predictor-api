from fastapi import FastAPI, HTTPException
from predictor import predict_fight

app = FastAPI(title="UFC Fight Predictor")

@app.get("/predict")
def predict(fighter1: str, fighter2: str):
    if fighter1 == fighter2:
        raise HTTPException(status_code=400, detail="Fighters must be different")

    try:
        return predict_fight(fighter1, fighter2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))