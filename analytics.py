import os
import json
import pandas as pd
import numpy as np
import streamlit as st

def compute_run_delta(df_base, df_target):
    """Calcola la differenza punto su punto tra il run Target e quello Base."""
    common = df_base.index.intersection(df_target.index)
    df_b = df_base.loc[common]
    df_t = df_target.loc[common]
    delta = pd.DataFrame(index=common)
    delta["delta_t850"] = df_t["t850"] - df_b["t850"]
    delta["delta_t500"] = df_t["t500"] - df_b["t500"]
    delta["delta_t2m"] = df_t["t2m"] - df_b["t2m"]
    delta["delta_precip"] = df_t["precip"] - df_b["precip"]
    return delta

def generate_local_fallback_briefing(stats_dict, horizon_mode):
    """Motore deterministico di riserva per il microclima di Olgiate Olona."""
    lr = stats_dict.get("max_lapse_rate_850_500", 0.0)
    cape = stats_dict.get("max_cape_ensemble", 0.0)
    delta_7d = stats_dict.get("delta_t850_trend", 0.0)
    dls = stats_dict.get("max_dls_knots", 0.0)
    w_max = np.sqrt(2 * cape) if cape > 0 else 0
    
    # 1. Sinottica e Deriva
    if delta_7d >= 1.5:
        synoptic_txt = f"🔴 **Deriva in Riscaldamento (+{delta_7d:.1f}°C a medio termine):** Saccatura atlantica in rallentamento. Possibile isolamento a ovest (falla iberica) con richiamo caldo sciroccale e accumulo di umidità nei bassi strati lungo la Valle Olona."
    elif delta_7d <= -1.5:
        synoptic_txt = f"🔵 **Deriva in Raffreddamento ({delta_7d:.1f}°C a medio termine):** Ingresso anticipato del cavo d'onda atlantico, con fronte freddo più diretto verso le Alpi occidentali."
    else:
        synoptic_txt = f"⚪ **Assetto Stabile ({delta_7d:+.1f}°C):** Buona convergenza tra i run operativi, evoluzione confermata."

    # 2. Termodinamica & Severità
    if cape >= 2000 and lr >= 28.5:
        conv_txt = f"⚡ **Potenziale Convettivo Severo:** CAPE di picco a **{cape:.0f} J/kg** e Lapse Rate $\Delta T = {lr:.1f}^\circ\\text{{C}}$. Velocità ascensionale stimata $w_{{max}} \\approx {w_max:.0f}\\text{{ m/s}}$ ({w_max*3.6:.0f} km/h). Rischio grandine media/grossa e downburst se attivato il trigger orografico sul Campo dei Fiori."
    elif cape >= 1000 or lr >= 27.5:
        conv_txt = f"🌦️ **Instabilità Moderata:** CAPE a **{cape:.0f} J/kg** con Lapse Rate $\Delta T = {lr:.1f}^\circ\\text{{C}}$ e DLS a **{dls:.0f} kts**. Possibili rovesci temporaleschi organizzati in multicelle lungo la fascia pedemontana."
    else:
        conv_txt = f"🟢 **Colonna d'Aria Stabile:** CAPE confinato sotto i 1000 J/kg o gradiente termico ordinario ($\Delta T = {lr:.1f}^\circ\\text{{C}}$). Basso rischio di fenomeni violenti."

    return f"{synoptic_txt}\n\n{conv_txt}"

@st.cache_data(show_spinner=False)
def get_ai_stormchaser_briefing(target_id, base_id, horizon_mode, stats_summary_json):
    """Interroga Gemini API per il bollettino da stormchaser o attiva il fallback locale."""
    stats_dict = json.loads(stats_summary_json)
    
    # Recupero chiave API dai Secrets di Streamlit o variabili d'ambiente
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    except Exception:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return generate_local_fallback_briefing(stats_dict, horizon_mode) + "\n\n> 💡 *Nota:* Per sbloccare l'analisi testuale avanzata dell'assistente AI Gemini, aggiungi `GEMINI_API_KEY` nei Secrets di Streamlit."

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        system_instruction = (
            "Sei MeteoMind, meteorologo senior e stormchaser esperto di dinamica convettiva e mesoscala nel Nord Italia, "
            "con focus specifico sulla pianura lombarda, la Valle Olona e la pedemontana del Campo dei Fiori (Olgiate Olona). "
            "Spiega i dati in modo chiaro, autorevole, scientifico ma accessibile. "
            "Struttura la risposta in 3 paragrafi senza elenchi puntati: "
            "1. Quadro Sinottico e Trend Run-to-Run (saccature, blocchi anticiclonici, asse del getto). "
            "2. Termodinamica e Carburante (CAPE, Lapse Rate 850-500, Dew Point al suolo, stima velocità updraft w_max). "
            "3. Focus Microclima Olgiate Olona (inflow sciroccale padano, forzante orografica del Campo dei Fiori, shear e tipo di struttura: pulse, multicella o supercella)."
        )

        prompt = f"""
{system_instruction}

Dati operativi del run:
- Target Run: {target_id} | Base Run: {base_id}
- Orizzonte visualizzato: {horizon_mode}
- Parametri quantitativi:
{stats_summary_json}
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return generate_local_fallback_briefing(stats_dict, horizon_mode)
        
