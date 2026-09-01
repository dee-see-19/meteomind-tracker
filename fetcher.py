import requests
import pandas as pd
import json
import os
from datetime import datetime

LAT = 45.63706
LON = 8.88147
DATA_FILE = "runs_history.json"

def fetch_and_save():
    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ["temperature_850hPa", "temperature_500hPa", "temperature_2m", "precipitation"],
        "models": "gfs_seamless",
        "timezone": "UTC"
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Errore API: {response.status_code}")
        return
        
    data = response.json()["hourly"]
    df_time = pd.to_datetime(data["time"])
    
    t850_cols = [k for k in data if "temperature_850hPa" in k]
    t500_cols = [k for k in data if "temperature_500hPa" in k]
    t2m_cols = [k for k in data if "temperature_2m" in k]
    precip_cols = [k for k in data if "precipitation" in k]
    
    df_t850 = pd.DataFrame({k: data[k] for k in t850_cols}, index=df_time)
    df_t500 = pd.DataFrame({k: data[k] for k in t500_cols}, index=df_time)
    df_t2m = pd.DataFrame({k: data[k] for k in t2m_cols}, index=df_time)
    df_precip = pd.DataFrame({k: data[k] for k in precip_cols}, index=df_time)
    
    now_utc = datetime.utcnow()
    hour_slot = f"{(now_utc.hour // 6) * 6:02d}Z"
    run_id = f"GEFS_{now_utc.strftime('%Y-%m-%d')}_{hour_slot}"
    
    run_payload = {
        "run_id": run_id,
        "timestamp": now_utc.isoformat(),
        "times": [t.strftime("%Y-%m-%d %H:%M") for t in df_time],
        "t850_mean": df_t850.mean(axis=1).round(2).tolist(),
        "t850_std": df_t850.std(axis=1).round(2).tolist(),
        "t500_mean": df_t500.mean(axis=1).round(2).tolist(),
        "t2m_mean": df_t2m.mean(axis=1).round(2).tolist(),
        "precip_mean": df_precip.mean(axis=1).round(2).tolist(),
    }
    
    history = {}
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                history = json.load(f)
        except Exception:
            history = {}
            
    history[run_id] = run_payload
    
    # Mantiene archiviati gli ultimi 20 run operativi (ultimi 5 giorni di emissioni)
    if len(history) > 20:
        oldest = sorted(history.keys())[0]
        del history[oldest]
        
    with open(DATA_FILE, "w") as f:
        json.dump(history, f, indent=2)
    print(f"✅ Run {run_id} archiviato correttamente.")

if __name__ == "__main__":
    fetch_and_save()
  
