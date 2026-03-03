import os
import importlib
from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI(title="UFC Predictor API")

@app.get("/")
def root():
    return {"message": "API je online. Modely se stahují na pozadí nebo při prvním volání."}

@app.get("/predict-event")
def predict_event(save_json: bool = False):
    try:
        # 1. Importujeme moduly uvnitř
        import predictor
        import input
        
        # 2. KLÍČOVÝ KROK: Vynutíme reload modulu input, 
        # aby se projevily změny v souboru, který jsi přepsal na disku
        importlib.reload(input)
        importlib.reload(predictor) # Reloadujeme i predictor, pokud si bere data z inputu při importu

        # 3. Spustíme predikci
        results = predictor.predict_event_with_shap_all()
        
        if save_json:
            predictor.save_event_to_json(results)
            
        return results
    except Exception as e:
        # Detailnější výpis chyby pro debugování na Renderu
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chyba při predikci: {str(e)}")

@app.get("/predict_shap")
def predict_shap(fighter1: str, fighter2: str):
    try:
        from predictor import predict_fight_with_shap
        return predict_fight_with_shap(fighter1, fighter2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Render port bind
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
