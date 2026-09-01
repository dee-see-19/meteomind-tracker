import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# Coordinate baricentriche: Olgiate Olona (VA) - Quota media 225m s.l.m.
LAT = 45.63706
LON = 8.88147
DATA_FILE = "runs_history.json"
MAX_STORED_RUNS = 24  # Mantiene in archivio gli ultimi 6 giorni di run completi (4 run/giorno)

def fetch_and_save():
    """
    Estrae il pacchetto ensemble completo GEFS a 384 ore (16 giorni)
    per l'analisi sinottica, termodinamica e cinematica su Olgiate Olona.
    """
    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    
    # Parametri fisici richiesti a tutti i membri ensemble
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": [
            "temperature_850hPa",      # Advezione termica nei bassi strati (~1500m)
            "temperature_500hPa",      # Aria fredda in quota e asse saccatura (~5500m)
            "temperature_2m",          # Temperatura al suolo
            "dew_point_2m",            # Contenuto di vapore acqueo / Mixing ratio al suolo
            "cape",                    # Convective Available Potential Energy (J/kg)
            "precipitation",           # Millimetri di pioggia oraria
            "wind_speed_10m",          # Vento al suolo per inflow padano (km/h)
            "wind_direction_10m",      # Direzione vento al suolo (gradi)
            "wind_speed_500hPa",       # Vento portante in quota per Shear 0-6km (km/h)
            "wind_direction_500hPa"    # Direzione getto/flusso a 500 hPa
        ],
        "models": "gfs_seamless",
        "timezone": "UTC"
    }
    
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')} UTC] Connessione API Open-Meteo Ensemble...")
    response = requests.get(url, params=params, timeout=30)
    
    if response.status_code != 200:
        print(f"❌ Errore API ({response.status_code}): {response.text}")
        return None
        
    raw_data = response.json()["hourly"]
    df_time = pd.to_datetime(raw_data["time"])
    
    # Funzione ausiliaria per isolare ed estrarre i dataframe di ciascun set di membri ensemble
    def extract_ensemble_df(var_prefix):
        member_cols = [col for col in raw_data if col.startswith(var_prefix)]
        return pd.DataFrame({col: raw_data[col] for col in member_cols}, index=df_time)

    # Estrazione matrici dei membri
    df_t850 = extract_ensemble_df("temperature_850hPa")
    df_t500 = extract_ensemble_df("temperature_500hPa")
    df_t2m = extract_ensemble_df("temperature_2m")
    df_dp = extract_ensemble_df("dew_point_2m")
    df_cape = extract_ensemble_df("cape")
    df_precip = extract_ensemble_df("precipitation")
    df_w10 = extract_ensemble_df("wind_speed_10m")
    df_wdir10 = extract_ensemble_df("wind_direction_10m")
    df_w500 = extract_ensemble_df("wind_speed_500hPa")
    df_wdir500 = extract_ensemble_df("wind_direction_500hPa")

    # Identificazione canonica del Run Operativo (00Z, 06Z, 12Z, 18Z)
    now_utc = datetime.utcnow()
    slot_hour = (now_utc.hour // 6) * 6
    run_slot_str = f"{slot_hour:02d}Z"
    run_id = f"GEFS_{now_utc.strftime('%Y-%m-%d')}_{run_slot_str}"

    # Calcolo proxy cinematica: Deep Layer Shear (DLS 0-6 km approssimato)
    # Differenza vettoriale scalare tra 500 hPa e 10 m in nodi (1 km/h = 0.539957 nodi)
    dls_mean_knots = ((df_w500.mean(axis=1) - df_w10.mean(axis=1)).clip(lower=0) * 0.54).round(1)

    # Costruzione del payload compresso con metriche statistiche aggregate
    run_payload = {
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
        "wind_speed_10m_mean": df_w10.mean(axis=1).round(1).tolist(),
        "wind_dir_10m_mean": df_wdir10.mean(axis=1).round(0).tolist(),
        "dls_0_6km_knots": dls_mean_knots.tolist()
    }

    # Gestione storico su file JSON
    history = {}
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = {}

    # Sovrascrive o aggiunge il run corrente
    history[run_id] = run_payload

    # Pruning per non appesantire il file (conserva gli ultimi MAX_STORED_RUNS)
    if len(history) > MAX_STORED_RUNS:
        oldest_keys = sorted(history.keys())[:-MAX_STORED_RUNS]
        for k in oldest_keys:
            del history[k]

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"✔ Snapshot {run_id} registrato con successo ({len(df_time)} ore / ~16 giorni).")
    return run_id

if __name__ == "__main__":
    fetch_and_save()
    
