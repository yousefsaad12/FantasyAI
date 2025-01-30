import pandas as pd
from sklearn.preprocessing import StandardScaler

def predict_player(player_name, data, features, best_model, scaler):
    player_data = data[data["playerName"] == player_name]
    if player_data.empty:
        return f"Player '{player_name}' not found in the dataset."

    player_features = player_data[features + ["avgPointsLast3", "maxPointsLast5"]].iloc[-1:]
    player_features_scaled = scaler.transform(player_features)
    
    predicted_points = best_model.predict(player_features_scaled)[0]
    predicted_points = float(predicted_points)
    
    previous_points = player_data["previousPoints"].iloc[-1]
    previous_points = float(previous_points)
    
    if previous_points != 0:
        percentage_change = ((predicted_points - previous_points) / previous_points) * 100
    else:
        percentage_change = 0
    
    percentage_change = float(percentage_change)
    trend = "Increasing" if percentage_change > 0 else "Decreasing"

    position = player_data["position"].values[0]
    is_goalkeeper = position == 1

    if not is_goalkeeper:
        assists_percentage = player_data["assists"].iloc[-1] / player_data["totalPoints"].iloc[-1] * 100 if player_data["totalPoints"].iloc[-1] > 0 else 0
        goals_percentage = player_data["goalsScored"].iloc[-1] / player_data["totalPoints"].iloc[-1] * 100 if player_data["totalPoints"].iloc[-1] > 0 else 0
    else:
        clean_sheet_percentage = player_data["cleanSheets"].iloc[-1] / player_data["totalPoints"].iloc[-1] * 100 if player_data["totalPoints"].iloc[-1] > 0 else 0

    if not is_goalkeeper:
        assists_percentage = float(assists_percentage)
        goals_percentage = float(goals_percentage)
    else:
        clean_sheet_percentage = float(clean_sheet_percentage)
    
    avg_bonus_points = player_data["bonus"].mean()
    points_per_week = player_data["totalPoints"].mean()

    result = {
        "playerName": player_name,
        "predictedPoints": round(predicted_points, 2),
        "percentageChange": f"{round(percentage_change, 2)}%",
        "trend": trend,
        "avgBonusPoints": round(avg_bonus_points, 2),
        "pointsPerWeek": round(points_per_week, 2),
    }

    if not is_goalkeeper:
        result["assistsPercentage"] = f"{round(assists_percentage, 2)}%"
        result["goalsPercentage"] = f"{round(goals_percentage, 2)}%"
    else:
        result["cleanSheetPercentage"] = f"{round(clean_sheet_percentage, 2)}%"

    return result