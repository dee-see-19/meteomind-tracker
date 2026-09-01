import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime

LAT = 45.63706
LON = 8.88147
DATA_FILE = "runs_history.json"
MAX_STORED_RUNS = 24

def fetch_and_save():
    """
    Estrae il pacchetto ensemble completo GEFS a 384 ore (16 giorni)
    per l'analisi sinottica, termodinamica e cinematica su Olgiate Olona.
    """
    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": [
            "temperature_850hPa",
            "temperature_500hPa",
            "temperature_2m",
            "dew_point_2m",
            "cape",
            "precipitation",
            "wind_speed_10m",
            "wind_speed_500hPa"
        ],
        "models": "gfs_seamless",
        "forecast_days": 16,
        "timezone": "UTC"
    }
    
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')} UTC] Connessione API Open-Meteo Ensemble (16 Giorni)...")
    response = requests.get(url, params=params, timeout=35)
    
    if response.status_code != 200:
        print(f"❌ Errore API ({response.status_code}): {response.text}")
        return None
        
    raw = response.json()["hourly"]
    df_time = pd.to_datetime(raw["time"])
    
    def extract_ensemble_df(var_prefix):
        cols = [c for c in raw if c.startswith(var_prefix)]
        if not cols:
            return pd.DataFrame(0.0, index=df_time, columns=["dummy"])
        return pd.DataFrame({c: raw[c] for c in cols}, index=df_time).fillna(0.0)

    df_t850 = extract_ensemble_df("temperature_850hPa")
    df_t500 = extract_ensemble_df("temperature_500hPa")
    df_t2m = extract_ensemble_df("temperature_2m")
    df_dp = extract_ensemble_df("dew_point_2m")
    df_cape = extract_ensemble_df("cape")
    df_precip = extract_ensemble_df("precipitation")
    df_w10 = extract_ensemble_df("wind_speed_10m")
    df_w500 = extract_ensemble_df("wind_speed_500hPa")

    # Deep Layer Shear (DLS 0-6 km approssimato in nodi)
    dls_series = ((df_w500.mean(axis=1) - df_w10.mean(axis=1)).clip(lower=0) * 0.54).round(1)

    now_utc = datetime.utcnow()
    hour_slot = f"{(now_utc.hour // 6) * 6:02d}Z"
    run_id = f"GEFS_{now_utc.strftime('%Y-%m-%d')}_{hour_slot}"

    payload = {
        "run_id": run_id,
        "fetched_at": now_utc.isoformat(),
        "total_hours": len(df_time),
        "times": [t.strftime("%Y-%m-%d %H:%M") for t in df_time],
        "t850_mean": df_t850.mean(axis=1).round(2).tolist(),
        "t850_std": df_t850.std(axis=1).round(2).tolist(),
        "t500_mean": df_t500.mean(axis=1).round(2).tolist(),
        "t2m_mean": df_t2m.mean(axis=1).round(2).tolist(),
        "dewpoint_mean": df_dp.mean(axis=1).round(2).tolist(),
        "cape_mean": df_cape.mean(axis=1).round(1).tolist(),
        "cape_max": df_cape.max(axis=1).round(1).tolist(),
        "precip_mean": df_precip.mean(axis=1).round(2).tolist(),
        "precip_max": df_precip.max(axis=1).round(2).tolist(),
        "precip_accum": df_precip.mean(axis=1).cumsum().round(1).tolist(),
        "dls_knots": dls_series.tolist()
    }

    history = {}
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = {}

    history[run_id] = payload

    if len(history) > MAX_STORED_RUNS:
        for k in sorted(history.keys())[:-MAX_STORED_RUNS]:
            del history[k]

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"✔ Run {run_id} archiviato con successo ({len(df_time)} ore / ~16 giorni).")
    return run_id

if __name__ == "__main__":
    fetch_and_save()
