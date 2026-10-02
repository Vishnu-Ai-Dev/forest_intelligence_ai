import os
import csv
import random

def generate_synthetic_data(output_path: str, num_samples: int = 1000):
    """
    Generates a synthetic dataset for forest fire risk prediction.
    NOTE: This is a hackathon prototype dataset, not real-world observations.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["temperature", "humidity", "rainfall", "wind_speed", "vegetation_dryness", "risk_score"])
        
        for _ in range(num_samples):
            # Reasonable environmental ranges
            temperature = random.uniform(10.0, 45.0)  # Celsius
            humidity = random.uniform(10.0, 90.0)     # Percentage
            rainfall = random.uniform(0.0, 50.0)      # mm (last 24h)
            wind_speed = random.uniform(0.0, 60.0)    # km/h
            vegetation_dryness = random.uniform(0.0, 1.0) # 0 to 1 index
            
            # Synthetic heuristic function to calculate risk score
            # High temp, low humidity, low rain, high wind, high dryness = HIGH RISK
            
            base_risk = (temperature / 45.0) * 30 + \
                        ((100 - humidity) / 100.0) * 20 + \
                        (wind_speed / 60.0) * 20 + \
                        (vegetation_dryness) * 30
            
            # Reduce risk sharply if it rained recently
            if rainfall > 5.0:
                base_risk *= max(0.1, (1.0 - (rainfall / 50.0)))
                
            risk_score = max(0.0, min(100.0, base_risk + random.uniform(-5, 5)))
            
            writer.writerow([
                round(temperature, 1),
                round(humidity, 1),
                round(rainfall, 1),
                round(wind_speed, 1),
                round(vegetation_dryness, 2),
                round(risk_score, 1)
            ])

if __name__ == "__main__":
    output = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample", "synthetic_fire_risk_data.csv")
    generate_synthetic_data(output)
    print(f"Generated synthetic dataset at {output}")
