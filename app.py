import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import os
from analytics import compute_run_delta, generate_dynamic_explanations

st.set_page_config(page_title="MeteoMind Stormchaser Lab", layout="wide")

st.title("⚡ MeteoMind: Synoptic & Convective Intelligence")
st.caption("Coordinate: Olgiate Olona (45.64°N, 8.88°E) | Microclima: Alta Pianura Padana & Pedemontana Varesina")

DATA_FILE = "runs_history.json"
if not os.path.exists(DATA_FILE):
    st.info("In attesa del primo salvataggio. Esegui la GitHub Action per popolare il database.")
    st.stop()

with open(DATA_FILE, "r") as f:
    history = json.load(f)

run_keys = sorted(list(history.keys()), reverse=True)
if not run_keys:
    st.warning("Database vuoto.")
    st.stop()

# Selettore Run
col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    target_id = st.selectbox("🎯 Run Attivo (TARGET):", run_keys, index=0)
with col_sel2:
    base_idx = 1 if len(run_keys) > 1 else 0
    base_id = st.selectbox("⚖️ Run di Confronto (BASE):", run_keys, index=base_idx)

t_data = history[target_id]
b_data = history[base_id]

df_target = pd.DataFrame({
    "time": pd.to_datetime(t_data["times"]),
    "t850": t_data["t850_mean"],
    "t850_std": t_data.get("t850_std", [0]*len(t_data["times"])),
    "t500": t_data["t500_mean"],
    "t2m": t_data["t2m_mean"],
    "dewpoint": t_data.get("dewpoint_mean", [0]*len(t_data["times"])),
    "cape_mean": t_data.get("cape_mean", [0]*len(t_data["times"])),
    "cape_max": t_data.get("cape_max", [0]*len(t_data["times"])),
    "precip": t_data["precip_mean"]
}).set_index("time")

df_base = pd.DataFrame({
    "time": pd.to_datetime(b_data["times"]),
    "t850": b_data["t850_mean"],
    "t500": b_data["t500_mean"],
    "t2m": b_data["t2m_mean"],
    "precip": b_data["precip_mean"]
}).set_index("time")

common_idx = df_target.index.intersection(df_base.index)
delta_df = compute_run_delta(df_base, df_target)

# Esecuzione Motore Didattico Dinamico
thermo_exp, cape_exp, delta_exp, local_alerts = generate_dynamic_explanations(df_target, delta_df)

# SEZIONE 1: LIVE STORMCHASER BRIEFING SUL TARGET RUN
st.markdown("---")
st.subheader("🧭 Live Stormchaser Desk: Olgiate Olona & Varesotto")

if local_alerts:
    for al in local_alerts:
        with st.expander(f"{al['structure']} ➔ {al['time_str']}"):
            st.markdown(al["micro_note"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ΔT (850-500 hPa)", f"{al['lr']:.1f} °C")
            c2.metric("CAPE Max Stimato", f"{al['cape']:.0f} J/kg")
            c3.metric("Dew Point al Suolo", f"{al['dp']:.1f} °C")
            c4.metric("Rain Rate Modellato", f"{al['rain']:.1f} mm/h")
else:
    st.success("✅ Nessun innesco convettivo severo rilevato. Colonna atmosferica stabile o inibita da subsidenza.")

st.markdown("---")

# SEZIONE 2: GRAFICI CON SPIEGAZIONE DINAMICA SOTTOSTANTE
tab_thermo, tab_cape, tab_delta = st.tabs([
    "📈 Profilo Termico & Lapse Rate",
    "⚡ Carburante Convettivo (CAPE / Td)",
    "📊 Deriva Run-to-Run (Delta Trend)"
])

with tab_thermo:
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=df_target.index, y=df_target["t850"], name="T850 (~1500m)", line=dict(color="#ff3333", width=2.5)))
    fig_t.add_trace(go.Scatter(x=df_target.index, y=df_target["t500"], name="T500 (~5500m)", line=dict(color="#3399ff", width=2)))
    fig_t.add_trace(go.Scatter(x=df_target.index, y=df_target["t2m"], name="T2m (Suolo)", line=dict(color="#ffa31a", width=1.5, dash="dot")))
    fig_t.update_layout(title="Profilo Termico Multi-Livello (850 hPa vs 500 hPa)", template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig_t, use_container_width=True)
    
    # Spiegazione Dinamica per il Grafico Termico
    st.info(thermo_exp)
    st.caption("💡 **Cosa guardare:** Il divario tra la linea rossa (1500m) e la linea blu (5500m). Più si allontanano, più l'atmosfera accelera i moti verticali.")

with tab_cape:
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(x=df_target.index, y=df_target["cape_max"], name="CAPE Max Membri (J/kg)", line=dict(color="#00ffcc", width=2)))
    fig_c.add_trace(go.Scatter(x=df_target.index, y=df_target["cape_mean"], name="CAPE Medio (J/kg)", fill='tozeroy', fillcolor='rgba(0, 255, 204, 0.15)', line=dict(color="#00b386", width=1.5)))
    fig_c.add_trace(go.Scatter(x=df_target.index, y=df_target["dewpoint"], name="Dew Point Suolo (°C)", yaxis="y2", line=dict(color="#ff00ff", width=1.8, dash="dash")))
    fig_c.update_layout(
        title="Energia Convettiva Disponibile (CAPE) & Punto di Rugiada (Dew Point)",
        template="plotly_dark", hovermode="x unified",
        yaxis=dict(title="CAPE (J/kg)"),
        yaxis2=dict(title="Punto di Rugiada (°C)", overlaying="y", side="right")
    )
    st.plotly_chart(fig_c, use_container_width=True)
    
    # Spiegazione Dinamica per il Grafico CAPE
    st.info(cape_exp)
    st.caption("💡 **Cosa guardare:** Se il Dew Point (linea viola tratteggiata) crolla sotto i 12°C senza pioggia, sta entrando il Favonio (Föhn da Nord), che sterilizza la convezione.")

with tab_delta:
    fig_d = go.Figure()
    colors = ['#ff4d4d' if v >= 0 else '#3399ff' for v in delta_df["delta_t850"]]
    fig_d.add_trace(go.Bar(x=common_idx, y=delta_df["delta_t850"], marker_color=colors, name="Δ T850"))
    fig_d.update_layout(title="Variazione Netta Run-to-Run (Nuovo Run vs Run di Riferimento)", template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig_d, use_container_width=True)
    
    # Spiegazione Dinamica per la Deriva
    st.info(delta_exp)
    st.caption("💡 **Cosa guardare:** Le barre rosse indicano che il run appena uscito sta scaldando/posticipando il fronte rispetto a prima; le barre blu indicano accelerazione dell'aria fredda.")
