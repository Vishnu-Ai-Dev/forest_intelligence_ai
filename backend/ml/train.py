import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample", "synthetic_fire_risk_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "forest_fire_model.pkl")

def train_model():
    """
    Trains a RandomForestRegressor on the synthetic dataset and saves it to disk.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run data_generator.py first.")
        
    X_list = []
    y_list = []
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X_list.append([
                float(row["temperature"]),
                float(row["humidity"]),
                float(row["rainfall"]),
                float(row["wind_speed"]),
                float(row["vegetation_dryness"])
            ])
            y_list.append(float(row["risk_score"]))
            
    X = np.array(X_list)
    y = np.array(y_list)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Model trained successfully.")
    print(f"Evaluation Metrics (on synthetic test set):")
    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
