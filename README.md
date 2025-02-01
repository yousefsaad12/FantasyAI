# Fantasy Edge - AI Model

## Overview

The **AI Model** in Fantasy Premier Edge is designed to analyze historical player performance data, predict future scores, and recommend optimal team changes. It leverages **XGBoost**, a powerful gradient-boosting algorithm, to make data-driven predictions.

## Features

- **Fantasy Points Prediction**: Predicts the expected fantasy points for each player based on past performances, upcoming fixtures, and opponent difficulty.
- **Percentage Change in Performance**: Calculates the percentage change in predicted points compared to the previous game week.
- **Gameweek Insights**: Provides expected player contributions for the upcoming matches based on statistical and AI-driven forecasts.
- **Continuous Learning**: Updates models with the latest data every week to refine predictions and improve accuracy.
- **Injury & Suspension Awareness**: Flags players with injuries or suspensions that might impact their performance.

## Machine Learning Approach

- **Data Collection**: Fetches and processes player statistics from the Fantasy Premier League API.
- **Data Processing**: Uses **Pandas** and **NumPy** to clean and prepare historical player statistics.
- **Feature Engineering**: Constructs relevant features such as form, fixture difficulty, and expected goal involvement.
- **XGBoost Model Training**: Trains on past fantasy data, optimizing hyperparameters for better accuracy.
- **Prediction API**: Serves predictions through a **FastAPI** endpoint, allowing seamless integration with the backend.

### Example API Response:

```json
{
  "playerName": "Mohamed Salah",
  "predictedPoints": 9.2,
  "percentageChange": "5.4%",
  "trend": "Upward",
  "avgBonusPoints": 1.5,
  "pointsPerWeek": 6.8
}
```

## Tech Stack

- **Python** (Primary language)
- **XGBoost** (Machine learning model for player performance prediction)
- **Pandas / NumPy** (Data manipulation)
- **FastAPI** (Serving AI predictions as an API)
- **Docker** (Containerization)

## Installation

```sh
git clone https://github.com/YousefSaad25/FantasyPremierEdge-AI.git
cd FantasyPremierEdge-AI
pip install -r requirements.txt
uvicorn app:main --host 0.0.0.0 --port 8000
```

## Configuration

- Set up database connection and environment variables for data sources.

## Deployment

```sh
docker build -t fantasy-ai .
docker run -d -p 8000:8000 fantasy-ai
```

---
