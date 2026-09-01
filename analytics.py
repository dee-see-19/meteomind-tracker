import os
import json
import pandas as pd
import numpy as np
import google.generativeai as genai
import streamlit as st

def compute_run_delta(df_base, df_target):
    common = df_base.index.intersection(df_target.index)
    df_b = df_base.loc[common]
    df_t = df_target.loc[common]
    delta = pd.DataFrame(index=common)
    delta["delta_t850"] = df_t["t850"] - df_b["t850"]
    delta["delta_t500"] = df_t["t500"] - df_b["t500"]
    delta["delta_t2m"] = df_t["t2m"] - df_b["t2m"]
    delta["delta_precip"] = df_t["precip"] - df_b["precip"]
    return delta

@st.cache_data(show_spinner=False)
def get_ai_stormchaser_briefing(target_id, base_id, horizon_mode, stats_summary_json):
    """
    Interroga il modello LLM Gemini per generare un bollettino ultra-dettagliato
    tarato sul microclima di Olgiate Olona, memorizzandolo in cache per ogni run.
    """
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    if not api_key:
        return "⚠️ **API Key non configurata.** Inserisci `GEMINI_API_KEY` nei Secrets di Streamlit per attivare l'assistente AI live."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        system_instruction = (
            "Sei 'MeteoMind', un meteorologo senior e stormchaser di altissimo livello specializzato nella dinamica "
            "dei temporali severi (supercelle, squall line, tornado, grandine gigante) e nel microclima del Nord Italia "
            "(in particolare la pianura lombarda, la Valle Olona e la pedemontana del Campo dei Fiori / Olgiate Olona). "
            "Il tuo stile è autorevole, appassionato, scientificamente rigoroso e con occhio clinico da cacciatore di temporali. "
            "Analizza i dati del modello meteo forniti ed elabora una diagnosi approfondita strutturata in 3 punti: "
            "1. Assetto Sinottico & Deriva del Modello (analisi run-to-run, asse delle saccature, rischio falla iberica o blocco anticiclonico). "
            "2. Termodinamica & Carburante Convettivo (Lapse Rate 850-500, CAPE max, Dew Point a terra, velocità stimata updraft w_max, quota base nubi LCL). "
            "3. Focus Microclima & Rischio Severe Weather su Olgiate Olona (interazione con il massiccio del Campo dei Fiori, richiamo sciroccale padano, shear 0-6km e tipo di struttura convettiva attesa)."
        )

        prompt = f"""
{system_instruction}

Dati operativi del run attivo:
- Run Target: {target_id} | Run Base di confronto: {base_id}
- Orizzonte temporale visualizzato: {horizon_mode}
- Metriche quantitative rilevate:
{stats_summary_json}

Genera un'analisi ricca, dettagliata e scorrevole in formato Markdown per il bollettino operativo della dashboard.
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Errore durante la generazione del bollettino AI: {str(e)}"
