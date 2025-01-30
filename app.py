from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predictor import predict_player
from retrain import retrain_model
import pandas as pd
from dotenv import load_dotenv
import os
import joblib

load_dotenv()

app = FastAPI()
best_model = joblib.load("fantasy_edge_model.pkl")
scaler = joblib.load("scaler.pkl")

data = pd.read_csv("Player_Data.csv")
features = [
    "goalsScored", "assists", "cleanSheets", "penaltiesSaved", "penaltiesMissed",
    "ownGoals", "yellowCards", "redCards", "saves", "bonus", "bonusPointsSystem",
    "dreamTeamCount", "expectedGoals", "expectedAssists", "expectedGoalInvolvements",
    "expectedGoalsConceded", "expectedGoalsPer90", "expectedAssistsPer90",
    "goalsConcededPer90", "startsPer90", "cleanSheetsPer90"
]

class PlayerNameRequest(BaseModel):
    player_name: str

@app.post("/predict/")
def predict_player_endpoint(request: PlayerNameRequest):
    result = predict_player(request.player_name, data, features, best_model, scaler)
    
    if isinstance(result, str):
        return None

    return result 

@app.get("/retrain/")
def retrain_endpoint():
    api_url = os.getenv("API_URL")
    print(f"API_URL: {api_url}")
    
    if not api_url:
        raise HTTPException(status_code=400, detail="API_URL not found")

    try:
        result = retrain_model(api_url)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))