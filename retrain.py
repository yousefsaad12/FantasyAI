import requests
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def fetch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def retrain_model(api_url):
    try:
        data = fetch_data(api_url)

        if data.empty:
            raise Exception("No data fetched from API.")

        data["playerName"] = data["firstName"] + " " + data["secondName"]
        data = data.sort_values(by=["playerName", "gameWeek"])
        
        data["previousPoints"] = data.groupby("playerName")["totalPoints"].shift(1)
        data["avgPointsLast3"] = data.groupby("playerName")["totalPoints"].rolling(3).mean().reset_index(0, drop=True)
        data["maxPointsLast5"] = data.groupby("playerName")["totalPoints"].rolling(5).max().reset_index(0, drop=True)

        data = data.dropna(subset=["previousPoints", "avgPointsLast3", "maxPointsLast5"])

        features = [
            "goalsScored", "assists", "cleanSheets", "penaltiesSaved", "penaltiesMissed",
            "ownGoals", "yellowCards", "redCards", "saves", "bonus", "bonusPointsSystem",
            "dreamTeamCount", "expectedGoals", "expectedAssists", "expectedGoalInvolvements",
            "expectedGoalsConceded", "expectedGoalsPer90", "expectedAssistsPer90",
            "goalsConcededPer90", "startsPer90", "cleanSheetsPer90"
        ]
        target = "totalPoints"
        
        X = data[features + ["avgPointsLast3", "maxPointsLast5"]]
        y = data[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
        }

        xgb_model = xgb.XGBRegressor(objective="reg:squarederror")

        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring='neg_mean_squared_error'
        )

        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_

        best_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            **best_params
        )

        best_model.fit(X_train, y_train)

        best_model.save_model("fantasy_edge_model.json")
        joblib.dump(scaler, "scaler.pkl")

        cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
        print(f"Cross-Validation MSE: {-cv_scores.mean():.2f}")

        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        data.to_csv('Player_Data.csv')
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R2 Score: {r2:.2f}")

        return {"message": "Model retrained and saved successfully."}

    except Exception as e:
        return {"error": str(e)}