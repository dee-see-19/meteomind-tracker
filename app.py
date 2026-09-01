import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import os
from analytics import compute_run_delta, generate_stormchaser_briefing

st.set_page_config(page_title="MeteoMind Stormchaser Lab", layout="wide")

st.title("⚡ MeteoMind: Synoptic & Convective Diagnostic Engine")
st.caption("Coordinate Target: Olgiate Olona (45.64°N, 8.88°E) | Microclima: Bassa Pedemontana Varesina")

DATA_FILE = "runs_history.json"

if not os.path.exists(DATA_FILE):
    st.info("In attesa del primo salvataggio dati. Esegui la GitHub Action per popolare il database.")
    st.stop()

with open(DATA_FILE, "r") as f:
    history = json.load(f)

run_keys = sorted(list(history.keys()), reverse=True)
if not run_keys:
    st.warning("Nessun run disponibile.")
    st.stop()

col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    target_id = st.selectbox("🎯 Run Attivo (TARGET):", run_keys, index=0)
with col_sel2:
    base_idx = 1 if len(run_keys) > 1 else 0
    base_id = st.selectbox("⚖️ Run di Riferimento (BASE):", run_keys, index=base_idx)

target_data = history[target_id]
base_data = history[base_id]

df_target = pd.DataFrame({
    "time": pd.to_datetime(target_data["times"]),
    "t850": target_data["t850_mean"],
    "t850_std": target_data["t850_std"],
    "t500": target_data["t500_mean"],
    "t2m": target_data["t2m_mean"],
    "precip": target_data["precip_mean"]
}).set_index("time")

df_base = pd.DataFrame({
    "time": pd.to_datetime(base_data["times"]),
    "t850": base_data["t850_mean"],
    "t500": base_data["t500_mean"],
    "t2m": base_data["t2m_mean"],
    "precip": base_data["precip_mean"]
}).set_index("time")

common_idx = df_target.index.intersection(df_base.index)
delta_df = compute_run_delta(df_base, df_target)

# Generazione Diagnostica Stormchaser
synoptic_summary, severe_windows, target_std = generate_stormchaser_briefing(df_target, delta_df, target_id, base_id)

# SEZIONE 1: BRIEFING STORCHASER
st.markdown("---")
st.subheader("🧭 Stormchaser Diagnostic Desk")
st.info(synoptic_summary)

if severe_windows:
    st.markdown("#### ⚠️ Finestre di Rischio Convezione Severa Rilevate")
    for event in severe_windows:
        with st.expander(f"{event['threat']} ➔ {event['timestamp']}"):
            st.write(event["desc"])
            cols = st.columns(4)
            cols[0].metric("ΔT Verticale (850-500)", f"{event['lapse_rate']:.1f} °C")
            cols[1].metric("Isoterma 500 hPa (~5500m)", f"{event['t500']:.1f} °C")
            cols[2].metric("Isoterma 850 hPa (~1500m)", f"{event['t850']:.1f} °C")
            cols[3].metric("Pioggia Modellata Media", f"{event['rain']:.1f} mm/h")
else:
    st.success("✅ Nessun innesco convettivo severo rilevato. Colonna d'aria stabile o convezione inibita da subsidenza/mancanza di forzanti.")

st.markdown("---")

# SEZIONE 2: GRAFICI COMPARATIVI
tab1, tab2 = st.tabs(["📈 Analisi Termica e Spread", "📊 Differenziale Delta Run-to-Run"])

with tab1:
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(
        x=df_base.index, y=df_base["t850"],
        mode='lines', name=f'Base T850 ({base_id})',
        line=dict(color='gray', dash='dash', width=2)
    ))
    fig_comp.add_trace(go.Scatter(
        x=df_target.index, y=df_target["t850"],
        mode='lines', name=f'Target T850 ({target_id})',
        line=dict(color='#ff3333', width=2.5)
    ))
    fig_comp.add_trace(go.Scatter(
        x=df_target.index.tolist() + df_target.index.tolist()[::-1],
        y=(df_target["t850"] + df_target["t850_std"]).tolist() + (df_target["t850"] - df_target["t850_std"]).tolist()[::-1],
        fill='toself', fillcolor='rgba(255, 50, 50, 0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", name='Spread Incertezza (±1σ)'
    ))
    fig_comp.update_layout(title="Sovrapposizione Isoterme 850 hPa (~1500 m s.l.m.)", template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig_comp, use_container_width=True)

with tab2:
    fig_delta = go.Figure()
    colors = ['#ff4d4d' if v >= 0 else '#3399ff' for v in delta_df["delta_t850"]]
    fig_delta.add_trace(go.Bar(x=common_idx, y=delta_df["delta_t850"], marker_color=colors, name="Δ T850"))
    fig_delta.update_layout(title="Variazione Netta Run-to-Run (Rosso = Il nuovo run scalda | Blu = Il nuovo run raffredda)", template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig_delta, use_container_width=True)
