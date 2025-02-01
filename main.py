# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import requests
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

app = FastAPI()

# Global variables to hold model and data
model = None
scaler = StandardScaler()
data = pd.DataFrame()

# Define request models
class PlayerRequest(BaseModel):
    player_name: str

# Initialize model and data on startup
@app.on_event("startup")
async def startup_event():
    global model, scaler, data
    try:
        model = joblib.load("fantasy_edge_rf_model.pkl")
        data = pd.read_csv('Player_Data.csv')
        # Fit scaler with existing data (assuming data is available)
        features = get_features()
        X = data[features]
        scaler.fit(X)
    except:
        print("Initial model not found, please retrain first")

def get_features():
    return [
        "goalsScored", "assists", "cleanSheets", "penaltiesSaved", "penaltiesMissed",
        "ownGoals", "yellowCards", "redCards", "saves", "bonus", "bonusPointsSystem",
        "dreamTeamCount", "expectedGoals", "expectedAssists", "expectedGoalInvolvements",
        "expectedGoalsConceded", "expectedGoalsPer90", "expectedAssistsPer90",
        "goalsConcededPer90", "startsPer90", "cleanSheetsPer90",
        "avgPointsLast3", "maxPointsLast5", "daysSinceLastGame"
    ]

def fetch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Data fetch error: {e}")

def preprocess_data(df):
    df["playerName"] = df["firstName"] + " " + df["secondName"]
    df = df.sort_values(by=["playerName", "gameWeek"])
    
    # Create features
    df["previousPoints"] = df.groupby("playerName")["totalPoints"].shift(1)
    df["avgPointsLast3"] = df.groupby("playerName")["totalPoints"].rolling(3).mean().reset_index(0, drop=True)
    df["maxPointsLast5"] = df.groupby("playerName")["totalPoints"].rolling(5).max().reset_index(0, drop=True)
    
    # Handle datetime
    if 'gameWeek' in df.columns:
        df['gameWeek'] = pd.to_datetime(df['gameWeek'], errors='coerce')
        df['daysSinceLastGame'] = (datetime.datetime.now() - df['gameWeek']).dt.days
    
    df = df.dropna(subset=["previousPoints", "avgPointsLast3", "maxPointsLast5"])
    return df

@app.post("/retrain")
async def retrain_model():
    global model, scaler, data
    
    # Fetch new data
    url = 'http://fantasyedgeai.runasp.net/api/player/data'
    data = fetch_data(url)
    
    # Preprocess data
    data = preprocess_data(data)
    data.to_csv('Player_Data.csv', index=False)
    
    # Prepare features/target
    features = get_features()
    X = data[features]
    y = data["totalPoints"]
    
    # Feature Scaling
    X_scaled = scaler.fit_transform(X)
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Hyperparameter Tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']  # Removed 'auto' to prevent errors
    }
    
    rf_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    
    # Train final model
    model = RandomForestRegressor(random_state=42, **best_params)
    model.fit(X_train, y_train)
    
    # Save updated model and scaler
    joblib.dump(model, "fantasy_edge_rf_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    return {"message": "Model retrained successfully", "best_params": best_params}

@app.post("/predict")
async def predict(player_request: PlayerRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    
    player_name = player_request.player_name
    player_data = data[data["playerName"] == player_name]
    
    if player_data.empty:
        raise HTTPException(status_code=404, detail="Player not found")
    
    try:
        features = get_features()
        player_features = player_data[features].iloc[-1:]
        player_features_scaled = scaler.transform(player_features)
        
        predicted_points = model.predict(player_features_scaled)[0]
        previous_points = player_data["previousPoints"].iloc[-1]
        
        percentage_change = ((predicted_points - previous_points) / previous_points * 100) if previous_points != 0 else 0
        trend = "Increasing" if percentage_change > 0 else "Decreasing"
        
        position = player_data["position"].values[0]
        result = {
            "playerName": player_name,
            "predictedPoints": round(float(predicted_points), 2),
            "percentageChange": round(float(percentage_change), 2),
            "trend": trend
        }
        
        if position != 1:  # Not goalkeeper
            result.update({
                "assistsLast5": int(player_data["assists"].tail(5).sum()),
                "goalsLast5": int(player_data["goalsScored"].tail(5).sum())
            })
        else:
            result["cleanSheetsLast5"] = int(player_data["cleanSheets"].tail(5).sum())
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)