import os
import importlib
import traceback
from fastapi import FastAPI, HTTPException, Query
import uvicorn

app = FastAPI(title="UFC Predictor API")

@app.get("/")
def root():
    return {"message": "API je online. Modely se stahují na pozadí nebo při prvním volání."}

@app.get("/predict-event")
def predict_event(save_json: bool = False):
    try:
        # Importy a vynucení reloadu pro aktuální data z GitHubu
        import predictor
        import input as input_mod
        
        importlib.reload(input_mod)
        importlib.reload(predictor)

        # Spuštění kompletní predikce eventu
        results = predictor.predict_event_with_shap_all()
        
        # Volitelné uložení (pokud máš v predictor.py funkci save_event_to_json)
        if save_json and hasattr(predictor, 'save_event_to_json'):
            predictor.save_event_to_json(results)
            
        return results
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chyba při predikci eventu: {str(e)}")

@app.get("/predict_shap")
def get_single_predict(
    fighter1: str, 
    fighter2: str, 
    odds1: float = Query(2.0, description="Kurz na fightera 1"), 
    odds2: float = Query(2.0, description="Kurz na fightera 2")
):
    try:
        import predictor
        importlib.reload(predictor)
        
        # Voláme funkci predict_shap, kterou jsme přidali do predictor.py
        # Pokud jsi ji v predictor.py pojmenoval predict_fight_with_shap, 
        # změň název níže na: predictor.predict_fight_with_shap(...)
        result = predictor.predict_shap(fighter1, fighter2, odds1, odds2)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
            
        return result
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chyba při individuální predikci: {str(e)}")

if __name__ == "__main__":
    # Render port bind
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
