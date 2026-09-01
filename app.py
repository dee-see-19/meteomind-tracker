import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import os

st.set_page_config(page_title="MeteoMind Varesotto Severe Tracker", layout="wide")

st.title("⚡ MeteoMind: Convective & Synoptic Run Tracker")
st.caption("Coordinate: Olgiate Olona (45.64°N, 8.88°E) | Monitoraggio Deriva Sinottica & Severe Weather")

DATA_FILE = "runs_history.json"

if not os.path.exists(DATA_FILE):
    st.info("Nessun dato ancora sincronizzato. Il primo run automatico verrà generato a breve da GitHub Actions.")
    st.stop()

with open(DATA_FILE, "r") as f:
    history = json.load(f)

run_keys = sorted(list(history.keys()), reverse=True)
if not run_keys:
    st.warning("Database vuoto.")
    st.stop()

col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    target_id = st.selectbox("Seleziona Run Più Recente (TARGET):", run_keys, index=0)
with col_sel2:
    base_idx = 1 if len(run_keys) > 1 else 0
    base_id = st.selectbox("Seleziona Run di Confronto (BASE):", run_keys, index=base_idx)

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
    "precip": base_data["precip_mean"]
}).set_index("time")

common_idx = df_target.index.intersection(df_base.index)
delta_t850 = df_target.loc[common_idx, "t850"] - df_base.loc[common_idx, "t850"]

# Indicatori di Deriva Sinottica
horizon_idx = min(168, len(df_target) - 1)
target_7d = df_target["t850"].iloc[horizon_idx]
delta_7d = delta_t850.iloc[min(horizon_idx, len(delta_t850)-1)]

col1, col2, col3 = st.columns(3)
col1.metric("T850 Prevista a +7 giorni", f"{target_7d:.1f} °C", f"{delta_7d:+.2f} °C vs run base")
col2.metric("Spread Incertezza (±1σ)", f"±{df_target['t850_std'].iloc[horizon_idx]:.2f} °C")

trend_msg = "Assetto Stabile"
if delta_7d >= 1.2:
    trend_msg = "Riscaldamento / Ritardo Cavo d'Onda 🔴"
elif delta_7d <= -1.2:
    trend_msg = "Raffreddamento / Anticipo Fronte 🔵"
col3.metric("Trend Sinottico", trend_msg)

# Grafico Comparativo Termico a 850 hPa
fig_comp = go.Figure()
fig_comp.add_trace(go.Scatter(
    x=df_base.index, y=df_base["t850"],
    mode='lines', name=f'Base ({base_id})',
    line=dict(color='gray', dash='dash', width=2)
))
fig_comp.add_trace(go.Scatter(
    x=df_target.index, y=df_target["t850"],
    mode='lines', name=f'Target ({target_id})',
    line=dict(color='#ff3333', width=2.5)
))
fig_comp.add_trace(go.Scatter(
    x=df_target.index.tolist() + df_target.index.tolist()[::-1],
    y=(df_target["t850"] + df_target["t850_std"]).tolist() + (df_target["t850"] - df_target["t850_std"]).tolist()[::-1],
    fill='toself', fillcolor='rgba(255, 50, 50, 0.1)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip", name='Spread Confidenza (±1σ)'
))
fig_comp.update_layout(title="Confronto Isoterme 850 hPa (~1500 m s.l.m.)", template="plotly_dark", hovermode="x unified")
st.plotly_chart(fig_comp, use_container_width=True)

# Grafico Differenziale Run-to-Run
fig_delta = go.Figure()
colors = ['#ff4d4d' if v >= 0 else '#3399ff' for v in delta_t850]
fig_delta.add_trace(go.Bar(x=common_idx, y=delta_t850, marker_color=colors, name="Δ T850"))
fig_delta.update_layout(title="Differenziale Run-to-Run (Rosso = Scalda | Blu = Raffredda)", template="plotly_dark", hovermode="x unified")
st.plotly_chart(fig_delta, use_container_width=True)

# Scanner Convettivo per il Microclima Locale
st.subheader("⚡ Finestre di Innesco Convettivo Severo Rilevate (Target Run)")
lapse_rate = df_target["t850"] - df_target["t500"]
alerts = []
for t, lr in lapse_rate.items():
    rain = df_target.loc[t, "precip"]
    if lr >= 27.5 and rain >= 1.0:
        risk = "Multicelle / Linee di Groppo"
        if lr >= 30.0 and rain >= 3.0:
            risk = "Severo: Elevato Rischio Supercellare / Grandine Grossa"
        alerts.append({
            "Data/Ora (UTC)": t.strftime("%Y-%m-%d %H:%M"),
            "ΔT Verticale (850-500)": f"{lr:.1f} °C",
            "T500": f"{df_target.loc[t, 't500']:.1f} °C",
            "Pioggia Oraria Media": f"{rain:.1f} mm/h",
            "Rischio Convezione": risk
        })

if alerts:
    st.dataframe(pd.DataFrame(alerts), use_container_width=True)
else:
    st.info("Nessuna finestra di instabilità severa rilevata nel run selezionato.")
  
