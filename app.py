import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="MeteoMind Severe Weather Lab", layout="wide")

# ==========================================
# MOTORE ANALITICO & BOLLETTINO DINAMICO
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

def generate_expert_briefing_by_horizon(stats_dict, horizon_mode):
    """
    Genera un bollettino specialistico che adatta l'analisi fisica
    in base alla scala temporale selezionata (3, 7 o 16 giorni).
    """
    lr = stats_dict.get("max_lapse_rate_850_500", 0.0)
    cape = stats_dict.get("max_cape_ensemble", 0.0)
    dp = stats_dict.get("max_dewpoint", 0.0)
    dls = stats_dict.get("max_dls_knots", 0.0)
    rain_max = stats_dict.get("max_rain_rate", 0.0)
    delta = stats_dict.get("delta_t850_trend", 0.0)
    w_max = np.sqrt(2 * cape) if cape > 0 else 0

    if "3" in horizon_mode:
        # FOCUS NOWCASTING / MESOSCALA (0-72h)
        title = "⚡ **BOLLETTINO NOWCASTING & MESOSCALA (3 GIORNI):**"
        if cape >= 1500 and lr >= 28.0:
            core = (
                f"Condizioni di elevata instabilità sul Varesotto. Dew Point a **{dp:.1f}°C** e picco CAPE a **{cape:.0f} J/kg** "
                f"con updraft potenziale $w_{{max}} \\approx {w_max:.0f}\\text{{ m/s}}$ ({w_max*3.6:.0f} km/h). "
                f"Con Deep-Layer Shear stimato a **{dls:.0f} kts**, il flusso sciroccale spinto contro il massiccio del Campo dei Fiori "
                "fornirà la forzante meccanica per innescare **supercelle isolate o multicelle grandinigene**."
            )
        else:
            core = (
                f"Fase a prevalente stabilità anticiclonica o convezione termica diurna ordinaria. CAPE confinato a **{cape:.0f} J/kg** "
                f"e Dew Point a **{dp:.1f}°C**. Assenza di forzanti dinamiche severe nelle 72 ore."
            )
        return f"{title}\n\n{core}"

    elif "7" in horizon_mode:
        # FOCUS MEDIO RAGGIO / SINOTTICA FRONTALE (3-7 giorni)
        title = "🛰️ **BOLLETTINO SINOTTICO FRONTALE (7 GIORNI):**"
        if delta >= 1.5:
            trend = f"🔴 **Trend in Riscaldamento (+{delta:.1f}°C):** La saccatura rallenta e sprofonda a ovest verso la penisola iberica (falla iberica), richiamando Scirocco caldo e umido con accumulo di calore latente nei bassi strati."
        elif delta <= -1.5:
            trend = f"🔵 **Trend in Raffreddamento ({delta:.1f}°C):** Ingresso anticipato del fronte freddo atlantico sul Nord-Ovest, con contrasti termici anticipati."
        else:
            trend = f"⚪ **Assetto Stabile ({delta:+.1f}°C):** Alta convergenza modellistica sull'evoluzione a medio raggio."

        details = f"Lapse Rate di picco atteso a **{lr:.1f}°C** tra 1500m e 5500m con piogge orarie modellate fino a **{rain_max:.1f} mm/h**."
        return f"{title}\n\n{trend}\n\n{details}"

    else:
        # FOCUS ENSEMBLE ESTESO (16 GIORNI)
        title = "🔭 **BOLLETTINO CIRCOLAZIONE GENERALE & ROSSBY (16 GIORNI):**"
        synop = (
            f"Analisi d'insieme sull'evoluzione a lungo termine. Si valuta la tenuta del promontorio subtropicale contro le ondulazioni del getto polare. "
            f"La deriva termica a medio-lungo termine segna **{delta:+.1f}°C**. "
            "Superate le 168-200 ore la dispersione ensemble consiglia di monitorare i cambi di regime barico generale (blocco a Omega vs treno di perturbazioni atlantiche) piuttosto che il singolo orario di pioggia."
        )
        return f"{title}\n\n{synop}"

@st.cache_data(show_spinner=False)
def get_meteomind_ai_briefing(target_id, base_id, horizon_mode, stats_json):
    stats_dict = json.loads(stats_json)
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    except Exception:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return generate_expert_briefing_by_horizon(stats_dict, horizon_mode)

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        prompt = f"""
Sei 'MeteoMind', meteorologo senior e stormchaser specializzato nella dinamica dei temporali severi nel Nord Italia (Olgiate Olona, Valle Olona, fascia pedemontana del Campo dei Fiori).
Adatta la tua diagnosi all'orizzonte temporale selezionato ({horizon_mode}):
- Se 3 Giorni: Focus su Nowcasting, convezione di mesoscala, CAPE, Dew Point, trigger orografico Campo dei Fiori, shear 0-6km e tipo di temporale (pulse, multicella, supercella).
- Se 7 Giorni: Focus su Sinottica, asse della saccatura, richiamo sciroccale, rischio falla iberica e timing del fronte freddo.
- Se 16 Giorni: Focus su Onde di Rossby, circolazione generale, tenuta anticiclonica vs rottura stagionale e spread ensemble.

Dati: Target: {target_id} | Base: {base_id}
Metriche: {stats_json}
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        return generate_expert_briefing_by_horizon(stats_dict, horizon_mode)

# ==========================================
# INTERFACCIA DASHBOARD
# ==========================================

st.title("⚡ MeteoMind: Synoptic Intelligence & Severe Weather Desk")
st.caption("Coordinate Target: Olgiate Olona (45.64°N, 8.88°E) | Copilota Meteorologico di Mesoscala")

DATA_FILE = "runs_history.json"
if not os.path.exists(DATA_FILE):
    st.info("In attesa del primo salvataggio dati. Esegui la GitHub Action per estrarre l'archivio a 16 giorni.")
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
    base_id = st.selectbox("⚖️ Run di Confronto (BASE):", run_keys, index=base_idx)
with c_r3:
    horizon_mode = st.radio(
        "🔭 Orizzonte Temporale:",
        ["3 Giorni (Nowcasting)", "7 Giorni (Medio Raggio)", "16 Giorni (Trend Ensemble)"],
        index=2,
        horizontal=True
    )

horizon_hours = 72 if "3" in horizon_mode else (168 if "7" in horizon_mode else 384)

t_data = history[target_id]
b_data = history[base_id]

df_target_full = pd.DataFrame({
    "time": pd.to_datetime(t_data["times"]),
    "t850": t_data.get("t850_mean", []),
    "t850_std": t_data.get("t850_std", [0]*len(t_data["times"])),
    "t500": t_data.get("t500_mean", []),
    "t2m": t_data.get("t2m_mean", []),
    "dewpoint": t_data.get("dewpoint_mean", [0]*len(t_data["times"])),
    "cape_mean": t_data.get("cape_mean", [0]*len(t_data["times"])),
    "cape_max": t_data.get("cape_max", [0]*len(t_data["times"])),
    "precip": t_data.get("precip_mean", [0]*len(t_data["times"])),
    "precip_max": t_data.get("precip_max", [0]*len(t_data["times"])),
    "precip_accum": t_data.get("precip_accum", [0]*len(t_data["times"])),
    "dls_knots": t_data.get("dls_knots", [0]*len(t_data["times"]))
}).set_index("time")

df_base_full = pd.DataFrame({
    "time": pd.to_datetime(b_data["times"]),
    "t850": b_data.get("t850_mean", []),
    "t500": b_data.get("t500_mean", []),
    "t2m": b_data.get("t2m_mean", []),
    "precip": b_data.get("precip_mean", [0]*len(b_data["times"]))
}).set_index("time")

limit_time = df_target_full.index[0] + pd.Timedelta(hours=horizon_hours)
df_target = df_target_full[df_target_full.index <= limit_time]
df_base = df_base_full[df_base_full.index <= limit_time]

delta_df = compute_run_delta(df_base, df_target)

# Calcolo Metriche
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

# BOLLETTINO DINAMICO METEOMIND
st.markdown("---")
st.subheader("🧭 MeteoMind Desk: Bollettino Sinottico & Mesoscala")

with st.spinner("Analisi dei profili termodinamici in corso..."):
    briefing_text = get_meteomind_ai_briefing(target_id, base_id, horizon_mode, json.dumps(stats_summary, indent=2))

st.info(briefing_text)
st.markdown("---")

# SCHEDE GRAFICHE A 5 PANNELLI
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    f"📈 Profilo Termico & Lapse Rate",
    f"⚡ Energia CAPE & Dew Point",
    f"🌧️ Precipitazioni & Accumulo",
    f"💨 Cinematica & Wind Shear",
    f"📊 Deriva Run-to-Run (Delta)"
])

with tab1:
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=df_target.index, y=df_target["t850"], name="T850 (~1500m)", line=dict(color="#ff3333", width=2.5)))
    fig_t.add_trace(go.Scatter(x=df_target.index, y=df_target["t500"], name="T500 (~5500m)", line=dict(color="#3399ff", width=2)))
    fig_t.add_trace(go.Scatter(x=df_target.index, y=df_target["t2m"], name="T2m (Suolo)", line=dict(color="#ffa31a", width=1.5, dash="dot")))
    fig_t.update_layout(title=f"Profilo Termico Multi-Livello [{horizon_mode}]", template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig_t, use_container_width=True)

with tab2:
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(x=df_target.index, y=df_target["cape_max"], name="CAPE Max Membri (J/kg)", line=dict(color="#00ffcc", width=2)))
    fig_c.add_trace(go.Scatter(x=df_target.index, y=df_target["cape_mean"], name="CAPE Medio (J/kg)", fill='tozeroy', fillcolor='rgba(0, 255, 204, 0.12)', line=dict(color="#00b386", width=1.5)))
    fig_c.add_trace(go.Scatter(x=df_target.index, y=df_target["dewpoint"], name="Dew Point Suolo (°C)", yaxis="y2", line=dict(color="#ff00ff", width=1.8, dash="dash")))
    fig_c.update_layout(
        title=f"Instabilità Convettiva: CAPE (J/kg) & Dew Point (°C) [{horizon_mode}]",
        template="plotly_dark", hovermode="x unified",
        yaxis=dict(title="CAPE (J/kg)"),
        yaxis2=dict(title="Punto di Rugiada (°C)", overlaying="y", side="right")
    )
    st.plotly_chart(fig_c, use_container_width=True)

with tab3:
    fig_p = go.Figure()
    fig_p.add_trace(go.Bar(x=df_target.index, y=df_target["precip_max"], name="Pioggia Max Scenario (mm/h)", marker_color="rgba(0, 180, 255, 0.4)"))
    fig_p.add_trace(go.Bar(x=df_target.index, y=df_target["precip"], name="Pioggia Media Ensemble (mm/h)", marker_color="#0099ff"))
    fig_p.add_trace(go.Scatter(x=df_target.index, y=df_target["precip_accum"], name="Accumulo Cumulato (mm)", yaxis="y2", line=dict(color="#ffff66", width=2)))
    fig_p.update_layout(
        title=f"Precipitazioni Orarie & Accumulo Progressivo [{horizon_mode}]",
        template="plotly_dark", hovermode="x unified",
        yaxis=dict(title="Pioggia Oraria (mm/h)"),
        yaxis2=dict(title="Accumulo Cumulato (mm)", overlaying="y", side="right")
    )
    st.plotly_chart(fig_p, use_container_width=True)

with tab4:
    fig_w = go.Figure()
    fig_w.add_trace(go.Scatter(x=df_target.index, y=df_target["dls_knots"], name="Deep-Layer Shear 0-6km (kts)", line=dict(color="#ff9933", width=2.5)))
    fig_w.add_hline(y=35, line_dash="dash", line_color="red", annotation_text="Soglia Supercellare (>35 kts)")
    fig_w.add_hline(y=20, line_dash="dot", line_color="yellow", annotation_text="Soglia Multicellare (>20 kts)")
    fig_w.update_layout(title=f"Cinematica: Deep-Layer Shear 0-6 km (kts) [{horizon_mode}]", template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig_w, use_container_width=True)

with tab5:
    fig_d = go.Figure()
    common = df_target.index.intersection(df_base.index)
    delta_sub = delta_df.loc[common]
    colors = ['#ff4d4d' if v >= 0 else '#3399ff' for v in delta_sub["delta_t850"]]
    fig_d.add_trace(go.Bar(x=common, y=delta_sub["delta_t850"], marker_color=colors, name="Δ T850"))
    fig_d.update_layout(title=f"Deriva Termica Run-to-Run a 850 hPa [{horizon_mode}]", template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig_d, use_container_width=True)
