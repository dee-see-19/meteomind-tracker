import os
import json
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="MeteoMind Severe Weather Lab", layout="wide")

# ==========================================
# MOTORE ANALITICO & DIAGNOSTICO METEOMIND
# ==========================================

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

def generate_local_expert_briefing(stats_dict, horizon_mode):
    """Generatore deterministico di analisi meteorologica avanzata per Olgiate Olona."""
    lr = stats_dict.get("max_lapse_rate_850_500", 0.0)
    cape = stats_dict.get("max_cape_ensemble", 0.0)
    delta_7d = stats_dict.get("delta_t850_trend", 0.0)
    dls = stats_dict.get("max_dls_knots", 0.0)
    dp = stats_dict.get("max_dewpoint", 0.0)
    w_max = np.sqrt(2 * cape) if cape > 0 else 0
    lcl = max(200, 125 * (stats_dict.get("t850_max", 20.0) - dp))

    # Analisi Sinottica e Deriva
    if delta_7d >= 1.5:
        p1 = f"🔴 **QUADRO SINOTTICO & DERIVA RUN (+{delta_7d:.1f}°C):** La saccatura atlantica sta subendo una decisa frenata rispetto alle corse precedenti. Il modello ipotizza uno sprofondamento troppo occidentale verso la penisola iberica ('falla iberica'). Questa dinamica attiva un richiamo sciroccale prefrontale molto caldo dal Nord Africa, posticipando l'irruzione fredda e aumentando l'accumulo di calore sensibile e vapore acqueo nei bassi strati della Pianura Padana occidentale."
    elif delta_7d <= -1.5:
        p1 = f"🔵 **QUADRO SINOTTICO & DERIVA RUN ({delta_7d:.1f}°C):** I cluster modellistici accelerano la discesa dell'aria polare-marittima. Il cavo d'onda entra più franco sul Golfo del Leone: l'ingresso frontale sul Nord-Ovest risulterà anticipato e più incisivo rispetto a quanto simulato nei run passati."
    else:
        p1 = f"⚪ **QUADRO SINOTTICO & DERIVA RUN ({delta_7d:+.1f}°C):** Elevata stabilità e convergenza tra i membri ensemble. La traiettoria e la tempistica dell'asse della saccatura sono confermate."

    # Termodinamica della Colonna d'Aria
    if cape >= 2000 and lr >= 28.5:
        p2 = f"⚡ **TERMODINAMICA & ENERGIA CONVETTIVA:** Profilo verticale iper-instabile. Registriamo un picco di CAPE a **{cape:.0f} J/kg** e un Lapse Rate tra 1500m e 5500m di **{lr:.1f}°C** (gradiente verticale $> 7.2^\circ\\text{{C/km}}$). Con un Dew Point al suolo fino a **{dp:.1f}°C**, l'updraft teorico massimo tocca **$w_{{max}} \\approx {w_max:.0f}\\text{{ m/s}}$ ({w_max*3.6:.0f} km/h)**. Valori di questa magnitudo sostengono idrometeore di grandi dimensioni all'interno della nube."
    elif cape >= 1000 or lr >= 27.5:
        p2 = f"🌦️ **TERMODINAMICA & ENERGIA CONVETTIVA:** Instabilità moderata. CAPE di picco a **{cape:.0f} J/kg** con Lapse Rate $\Delta T = {lr:.1f}^\circ\\text{{C}}$ e Dew Point a **{dp:.1f}°C**. Velocità ascensionale potenziale nell'ordine dei **{w_max:.0f} m/s**, adatta per convezione organizzata a multicella."
    else:
        p2 = f"🟢 **TERMODINAMICA & ENERGIA CONVETTIVA:** Colonna atmosferica prevalentemente stabile o inibita da subsidenza anticiclonica. CAPE limitato e gradiente termico ordinario ($\Delta T = {lr:.1f}^\circ\\text{{C}}$). Assenza di forzanti convettive significative."

    # Microclima Olgiate Olona & Pedemontana
    if cape >= 1500 and dls >= 30:
        p3 = f"🧭 **FOCUS MICROCLIMA OLGIATE OLONA & VARESOTTO:** Rischio di strutture temporalesche rotazionali. La risalita di aria umida da Sud-Est lungo la Valle Olona impatta contro il baluardo orografico del **Campo dei Fiori**, fornendo il sollevamento forzato necessario a rompere il tappo (CIN). Il Deep-Layer Shear stimato a **{dls:.0f} kts** garantisce la separazione tra correnti ascensionali e discendenti: potenziale per **supercelle isolate o grandinate severe** lungo la direttrice Malpensa-Busto Arsizio-Saronno."
    elif cape >= 800:
        p3 = f"🧭 **FOCUS MICROCLIMA OLGIATE OLONA & VARESOTTO:** Possibile sviluppo di sistemi convettivi multicellari (MCS) o linee di groppo (squall line) in discesa dalle valli prealpine verso l'alta pianura, con rischio di improvvise raffiche di vento lineare (*downburst*)."
    else:
        p3 = f"🧭 **FOCUS MICROCLIMA OLGIATE OLONA & VARESOTTO:** Regime di ventilazione debole o secca. Monitorare eventuali ingressi di correnti da Nord (Favonio) che sterilizzerebbero istantaneamente la colonna d'aria abbattendo il Dew Point."

    return f"{p1}\n\n{p2}\n\n{p3}"

@st.cache_data(show_spinner=False)
def get_meteomind_briefing(target_id, base_id, horizon_mode, stats_json):
    """Chiama l'API LLM Gemini oppure attiva il generatore esperto locale."""
    stats_dict = json.loads(stats_json)
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    except Exception:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return generate_local_expert_briefing(stats_dict, horizon_mode)

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        prompt = f"""
Sei 'MeteoMind', meteorologo senior e stormchaser specializzato nella dinamica dei temporali severi e nel microclima del Nord Italia (Olgiate Olona, Valle Olona, fascia pedemontana del Campo dei Fiori).
Analizza i seguenti parametri estratti dal modello GEFS ed elabora un bollettino scientifico ma accessibile, dettagliato e strutturato in 3 sezioni chiare:
1. Analisi Sinottica & Trend Run-to-Run (traiettoria saccature, getto, rischio falla iberica o blocco anticiclonico).
2. Termodinamica della Colonna (Lapse Rate 850-500, CAPE max, Dew Point, velocità updraft w_max).
3. Focus Microclima Olgiate Olona & Rischio Strutture (inflow da Sud-Est, forzante orografica del Campo dei Fiori, shear 0-6km, classificazione tra pulse storm, multicelle, squall line o supercelle).

Dati:
Target: {target_id} | Base: {base_id} | Orizzonte: {horizon_mode}
Metriche: {stats_json}
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        return generate_local_expert_briefing(stats_dict, horizon_mode)

# ==========================================
# INTERFACCIA STREAMLIT DASHBOARD
# ==========================================

st.title("⚡ MeteoMind: AI Synoptic Intelligence & Severe Weather Desk")
st.caption("Coordinate Target: Olgiate Olona (45.64°N, 8.88°E) | Copilota Meteorologico di Mesoscala")

DATA_FILE = "runs_history.json"
if not os.path.exists(DATA_FILE):
    st.info("In attesa del primo salvataggio dati. Esegui la GitHub Action per popolare il database.")
    st.stop()

with open(DATA_FILE, "r", encoding="utf-8") as f:
    history = json.load(f)

run_keys = sorted(list(history.keys()), reverse=True)
if not run_keys:
    st.warning("Nessun run presente nel database.")
    st.stop()

c_r1, c_r2, c_r3 = st.columns([2, 2, 3])
with c_r1:
    target_id = st.selectbox("🎯 Run Attivo (TARGET):", run_keys, index=0)
with c_r2:
    base_idx = 1 if len(run_keys) > 1 else 0
    base_id = st.selectbox("⚖️ Run di Riferimento (BASE):", run_keys, index=base_idx)
with c_r3:
    horizon_mode = st.radio(
        "🔭 Orizzonte Temporale:",
        ["3 Giorni (Nowcasting)", "7 Giorni (Medio Raggio)", "14 Giorni (Trend Ensemble)"],
        index=2,
        horizontal=True
    )

horizon_hours = 72 if "3" in horizon_mode else (168 if "7" in horizon_mode else 336)

t_data = history[target_id]
b_data = history[base_id]

df_target_full = pd.DataFrame({
    "time": pd.to_datetime(t_data["times"]),
    "t850": t_data["t850_mean"],
    "t850_std": t_data.get("t850_std", [0]*len(t_data["times"])),
    "t500": t_data["t500_mean"],
    "t2m": t_data["t2m_mean"],
    "dewpoint": t_data.get("dewpoint_mean", [0]*len(t_data["times"])),
    "cape_mean": t_data.get("cape_mean", [0]*len(t_data["times"])),
    "cape_max": t_data.get("cape_max", [0]*len(t_data["times"])),
    "precip": t_data["precip_mean"],
    "dls_knots": t_data.get("dls_knots", [0]*len(t_data["times"]))
}).set_index("time")

df_base_full = pd.DataFrame({
    "time": pd.to_datetime(b_data["times"]),
    "t850": b_data["t850_mean"],
    "t500": b_data["t500_mean"],
    "t2m": b_data["t2m_mean"],
    "precip": b_data["precip_mean"]
}).set_index("time")

limit_time = df_target_full.index[0] + pd.Timedelta(hours=horizon_hours)
df_target = df_target_full[df_target_full.index <= limit_time]
df_base = df_base_full[df_base_full.index <= limit_time]

delta_df = compute_run_delta(df_base, df_target)

# Calcolo Metriche per il Bollettino
lapse_rate = df_target["t850"] - df_target["t500"]
stats_summary = {
    "t850_max": float(df_target["t850"].max()),
    "t850_min": float(df_target["t850"].min()),
    "t500_min": float(df_target["t500"].min()),
    "max_lapse_rate_850_500": float(lapse_rate.max()),
    "max_cape_ensemble": float(df_target["cape_max"].max()),
    "max_dewpoint": float(df_target["dewpoint"].max()),
    "max_rain_rate": float(df_target["precip"].max()),
    "max_dls_knots": float(df_target["dls_knots"].max()),
    "delta_t850_trend": float(delta_df["delta_t850"].iloc[min(len(delta_df)-1, int(horizon_hours/2))]) if not delta_df.empty else 0.0
}

# SEZIONE: IL BOLLETTINO METEOROLOGICO DINAMICO
st.markdown("---")
st.subheader("🧭 MeteoMind Desk: Bollettino Sinottico & Mesoscala")

with st.spinner("Analisi dei campi termodinamici in corso..."):
    briefing_text = get_meteomind_briefing(target_id, base_id, horizon_mode, json.dumps(stats_summary, indent=2))

st.info(briefing_text)

st.markdown("---")

# SCHEDE GRAFICHE
tab1, tab2, tab3 = st.tabs([
    f"📈 Profilo Termico Multi-Livello ({horizon_mode})",
    f"⚡ Energia CAPE & Dew Point ({horizon_mode})",
    f"📊 Deriva Differenziale Run-to-Run ({horizon_mode})"
])

with tab1:
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=df_target.index, y=df_target["t850"], name="T850 (~1500m)", line=dict(color="#ff3333", width=2.5)))
    fig_t.add_trace(go.Scatter(x=df_target.index, y=df_target["t500"], name="T500 (~5500m)", line=dict(color="#3399ff", width=2)))
    fig_t.add_trace(go.Scatter(x=df_target.index, y=df_target["t2m"], name="T2m (Suolo)", line=dict(color="#ffa31a", width=1.5, dash="dot")))
    fig_t.update_layout(title=f"Evoluzione Termica Multi-Livello su Olgiate Olona [{horizon_mode}]", template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig_t, use_container_width=True)

with tab2:
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(x=df_target.index, y=df_target["cape_max"], name="CAPE Max Membri (J/kg)", line=dict(color="#00ffcc", width=2)))
    fig_c.add_trace(go.Scatter(x=df_target.index, y=df_target["cape_mean"], name="CAPE Medio (J/kg)", fill='tozeroy', fillcolor='rgba(0, 255, 204, 0.12)', line=dict(color="#00b386", width=1.5)))
    fig_c.add_trace(go.Scatter(x=df_target.index, y=df_target["dewpoint"], name="Dew Point Suolo (°C)", yaxis="y2", line=dict(color="#ff00ff", width=1.8, dash="dash")))
    fig_c.update_layout(
        title=f"Energia Potenziale Convettiva (CAPE) & Dew Point [{horizon_mode}]",
        template="plotly_dark", hovermode="x unified",
        yaxis=dict(title="CAPE (J/kg)"),
        yaxis2=dict(title="Punto di Rugiada (°C)", overlaying="y", side="right")
    )
    st.plotly_chart(fig_c, use_container_width=True)

with tab3:
    fig_d = go.Figure()
    common = df_target.index.intersection(df_base.index)
    delta_sub = delta_df.loc[common]
    colors = ['#ff4d4d' if v >= 0 else '#3399ff' for v in delta_sub["delta_t850"]]
    fig_d.add_trace(go.Bar(x=common, y=delta_sub["delta_t850"], marker_color=colors, name="Δ T850"))
    fig_d.update_layout(title=f"Deriva Termica Netta a 850 hPa (Target vs Base) [{horizon_mode}]", template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig_d, use_container_width=True)
    
