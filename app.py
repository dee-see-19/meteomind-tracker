import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import os
from analytics import compute_run_delta, get_ai_stormchaser_briefing

st.set_page_config(page_title="MeteoMind Stormchaser Lab", layout="wide")

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
    st.warning("Database vuoto.")
    st.stop()

c_r1, c_r2, c_r3 = st.columns([2, 2, 3])
with c_r1:
    target_id = st.selectbox("🎯 Run Attivo (TARGET):", run_keys, index=0)
with c_r2:
    base_idx = 1 if len(run_keys) > 1 else 0
    base_id = st.selectbox("⚖️ Run di Confronto (BASE):", run_keys, index=base_idx)
with c_r3:
    horizon_mode = st.radio(
        "🔭 Orizzonte Temporale:",
        ["3 Giorni (Nowcasting)", "7 Giorni (Medio Raggio)", "14 Giorni (Trend Ensemble)"],
        index=1,
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

# Preparazione sintesi fisica per l'AI
lapse_rate = df_target["t850"] - df_target["t500"]
stats_summary = {
    "t850_max": float(df_target["t850"].max()),
    "t850_min": float(df_target["t850"].min()),
    "t500_min": float(df_target["t500"].min()),
    "max_lapse_rate_850_500": float(lapse_rate.max()),
    "max_lapse_rate_time": lapse_rate.idxmax().strftime("%Y-%m-%d %H:%M UTC"),
    "max_cape_ensemble": float(df_target["cape_max"].max()),
    "max_cape_time": df_target["cape_max"].idxmax().strftime("%Y-%m-%d %H:%M UTC"),
    "max_dewpoint": float(df_target["dewpoint"].max()),
    "max_rain_rate": float(df_target["precip"].max()),
    "max_dls_knots": float(df_target["dls_knots"].max()),
    "delta_t850_trend": float(delta_df["delta_t850"].iloc[min(len(delta_df)-1, int(horizon_hours/2))]) if not delta_df.empty else 0.0
}

# SEZIONE: IL METEOROLOGO PERSONALE AI (IL BOX BLU DINAMICO)
st.markdown("---")
st.subheader("🤖 MeteoMind AI: Analisi Sinottica & Diagnosi di Mesoscala")

with st.spinner("MeteoMind sta analizzando le carte isobariche e i profili termodinamici per Olgiate Olona..."):
    ai_briefing = get_ai_stormchaser_briefing(
        target_id, base_id, horizon_mode, json.dumps(stats_summary, indent=2)
    )

st.info(ai_briefing)

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
    fig_c.add_trace(go.Scatter(x=df_target.index, y=df_target["cape_max"], name="CAPE Max (J/kg)", line=dict(color="#00ffcc", width=2)))
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
