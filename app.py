import os
import json
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="MeteoMind Stormchaser Lab", layout="wide")

# =========================================================
# MOTORE DIDATTICO E DIAGNOSTICO SPECIALISTICO METEOMIND
# =========================================================

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

def build_tab_expert_briefing(tab_name, stats, horizon_mode):
    """Generatore di spiegazioni didattiche e scientifiche ultra-approfondite."""
    lr = stats.get("max_lapse_rate_850_500", 0.0)
    cape = stats.get("max_cape_ensemble", 0.0)
    dp = stats.get("max_dewpoint", 0.0)
    t2m_max = stats.get("t2m_max", 30.0)
    dls = stats.get("max_dls_knots", 0.0)
    rain_max = stats.get("max_rain_rate", 0.0)
    accum = stats.get("total_accum", 0.0)
    delta = stats.get("delta_t850_trend", 0.0)
    t500_min = stats.get("t500_min", -12.0)
    t850_max = stats.get("t850_max", 18.0)
    w_max = np.sqrt(2 * cape) if cape > 0 else 0
    lcl = max(200, 125 * (t2m_max - dp))

    if tab_name == "thermo":
        return (
            f"### 🔬 Analisi Termodinamica & Profilo Verticale\n"
            f"* **Gradiente Termico Verticale (Lapse Rate $\\Gamma$):** Il divario termico massimo tra $850\\text{{ hPa}}$ (~1500m, **{t850_max:.1f}°C**) "
            f"e $500\\text{{ hPa}}$ (~5500m, **{t500_min:.1f}°C**) tocca i **{lr:.1f}°C** sull'orizzonte selezionato ({horizon_mode}). "
            f"Su uno spessore atmosferico di 4000 metri, questo corrisponde a un tasso di raffreddamento reale di **{lr/4:.2f}°C/km**.\n"
            f"* **Significato Fisico:** Un'atmosfera standard si raffredda a circa $6.5^\circ\\text{{C/km}}$. Quando il gradiente supera i **$7.0\\text{{–}}7.5^\circ\\text{{C/km}}$** "
            f"(come in questo caso con $\\Delta T \\ge 28.5^\circ\\text{{C}}$), l'aria si trova in uno stato di **instabilità condizionale super-adiabatica**. "
            f"Qualsiasi massa d'aria umida scalzata dal suolo risulterà costantemente più calda e leggera rispetto all'aria fredda circostante, accelerando spontaneamente verso l'alto per galleggiamento termico.\n"
            f"* **Dinamica Locale (Valle Olona):** La presenza di isoterme a 850 hPa superiori a 16-18°C testimonia un forte carico termico nei primi strati padani. "
            f"L'ingresso di aria fredda a 500 hPa fa da detonatore: se l'inversione termica al suolo viene rotta, il galleggiamento convettivo si attiva in modo esplosivo."
        )

    elif tab_name == "cape":
        return (
            f"### ⚡ Carburante Convettivo, Punti di Rugiada & Dinamica dell'Updraft\n"
            f"* **CAPE (Convective Available Potential Energy):** Il valore di picco simulato dai membri ensemble è di **{cape:.0f} J/kg**.\n"
            f"* **Velocità Ascensionale Teorica ($w_{{max}}$):** Convertendo l'energia termica in energia cinetica pura ($w_{{max}} = \\sqrt{{2 \\cdot \\text{{CAPE}}}}$), "
            f"l'aria all'interno dell'updraft può raggiungere una velocità ascensionale di **{w_max:.1f} m/s ({w_max*3.6:.0f} km/h)**. "
            f"Valori superiori a 40–50 m/s sono in grado di sospendere e far accrescere idrometeore congelate fino a calibri da **grandine media o gigante (>3–5 cm)**.\n"
            f"* **Dew Point al Suolo ($T_d$ = {dp:.1f}°C):** Il punto di rugiada rappresenta la concentrazione assoluta di vapore acqueo nei primi 500 metri. "
            f"Valori $\\ge 20^\circ\\text{{C}}$ indicano un catino padano saturo di calore latente. "
            f"La quota stimata della base delle nubi (**LCL**) si attesta a **~{lcl:.0f} metri**. Una base nube bassa riduce l'evaporazione sotto la nube e favorisce l'organizzazione tornadica.\n"
            f"* **Allerta Favonio (Föhn da Nord):** Se il Dew Point crollasse bruscamente sotto i 10°C senza precipitazioni, significherebbe l'ingresso di correnti alpine settentrionali compresse adiabaticamente, capaci di sterilizzare all'istante l'instabilità atmosferica."
        )

    elif tab_name == "rain":
        return (
            f"### 🌧️ Microfisica delle Precipitazioni, Downdrafts & Cold Pools\n"
            f"* **Intensità Oraria di Picco:** Previsti fino a **{rain_max:.1f} mm/h** con un accumulo cumulato progressivo di **{accum:.1f} mm**.\n"
            f"* **La Dinamica del Downdraft:** Quando le idrometeore (pioggia e grandine) precipitano attraverso gli strati intermedi, causano un raffreddamento per evaporazione (*evaporative cooling*). "
            f"L'aria raffreddata, diventando più pesante dell'aria calda circostante, precipita violentemente verso il terreno (*downdraft*).\n"
            f"* **Rischio Downburst al Suolo:** Toccando terra, il flusso si espande a ventaglio creando una sacca d'aria fredda (*cold pool*). "
            f"Se il contrasto termico con l'aria preesistente supera i 10–12°C, si generano raffiche lineari di **downburst oltre gli 80–110 km/h** lungo la direttrice Gallarate-Busto Arsizio-Legnano.\n"
            f"* **Outflow Boundary:** Il bordo d'avanzamento della bolla fredda funge da cuneo di sollevamento, costringendo l'aria umida a salire e rigenerando continuamente nuove celle temporalesche lungo l'alta pianura."
        )

    elif tab_name == "shear":
        structure = "Supercella Mesociclonica" if dls >= 35 else ("Multicella / Squall Line Organizzata" if dls >= 20 else "Temporale a Cella Singola / Pulse Storm")
        return (
            f"### 💨 Cinematica dei Venti & Struttura Convettiva Prevista\n"
            f"* **Deep-Layer Shear ($DLS_{{0-6\\text{{km}}}}$):** Differenziale di velocità tra il vento al suolo e a 5500 metri stimato a **{dls:.0f} nodi (kts)**.\n"
            f"* **Classificazione Morfologica:** In base ai parametri attuali, la tipologia temporalesca dominante è: **{structure}**.\n"
            f"* **Meccanismo di Separazione:** Con $DLS \\ge 35\\text{{ kts}}$, il vento in quota inclina l'asse del cumulonembo. "
            f"Questo impedisce alla colonna di pioggia di collassare sopra l'updraft (cosa che soffocherebbe il temporale in 30 minuti), garantendo una vita autonoma di svariate ore al sistema convettivo.\n"
            f"* **La Forzante del Campo dei Fiori:** A Olgiate Olona il vento al suolo da Sud-Est (Scirocco padano) converge contro il massiccio prealpino a nord. "
            f"La rotazione del vento con la quota (*Storm-Relative Helicity*) inietta vorticità orizzontale che l'updraft solleva in verticale, generando il **mesociclone rotante tipico delle supercelle**."
        )

    elif tab_name == "delta":
        if delta >= 1.5:
            trend_desc = f"🔴 **Riscaldamento / Rallentamento Cavo d'Onda (+{delta:.1f}°C a 850 hPa):** Il run attuale ritarda la perturbazione. Scenario tipico da **falla iberica**: la saccatura affonda troppo a ovest e richiama un'ondata di calore sciroccale che carica ulteriormente il serbatoio padano."
        elif delta <= -1.5:
            trend_desc = f"🔵 **Raffreddamento / Accelerazione Fronte ({delta:.1f}°C a 850 hPa):** Il modello anticipa la discesa della goccia fredda sul Nord-Ovest, stringendo i tempi per l'innesco dei primi temporali prealpini."
        else:
            trend_desc = f"⚪ **Assetto Stabile e Confermato ({delta:+.1f}°C a 850 hPa):** Elevata coerenza probabilistica tra i cluster ensemble."
        return (
            f"### 📊 Deriva Modellistica Run-to-Run & Onde Planetarie\n"
            f"* **Variazione Differenziale Netta:** {trend_desc}\n"
            f"* **Interpretazione Sinottica:** Il confronto diretto tra l'emissione attiva e quella precedente evidenzia lo spostamento spaziale delle onde di Rossby nel medio-lungo raggio ({horizon_mode}). "
            f"Monitorare le barre blu/rosse permette di capire se l'evento convettivo sta guadagnando intensità, se il fronte viene posticipato o se l'anticiclone subtropicale riuscirà a bloccare l'avanzata atlantica."
        )

@st.cache_data(show_spinner=False)
def get_tab_ai_explanation(tab_name, target_id, base_id, horizon_mode, stats_json):
    """Interroga l'AI Gemini per un approfondimento custom o genera l'analisi specialistica locale."""
    stats_dict = json.loads(stats_json)
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    except Exception:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return build_tab_expert_briefing(tab_name, stats_dict, horizon_mode)

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
Sei 'MeteoMind', meteorologo senior e stormchaser di altissimo livello specializzato nella dinamica dei temporali severi nel Nord Italia (Olgiate Olona, Valle Olona, fascia pedemontana del Campo dei Fiori).
Spiega in modo approfondito, scientifico ma accessibile cosa stiamo osservando nella scheda grafica '{tab_name}' per l'orizzonte '{horizon_mode}'.
SE UTILIZZI TERMINI TECNICI (es. Lapse Rate, CAPE, CIN, Dew Point, LCL, Downdraft, Cold Pool, Downburst, Deep-Layer Shear, Elicità SRH, Mesociclone, Supercella, Falla Iberica, Outflow), DEVI SPIEGARE CHIARAMENTE COSA SIGNIFICANO DAL PUNTO DI VISTA FISICO E COME SI APPLICANO A OLGIATE OLONA.

Dati numerici del run:
{stats_json}
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        return build_tab_expert_briefing(tab_name, stats_dict, horizon_mode)

# =========================================================
# INTERFACCIA UTENTE STREAMLIT
# =========================================================

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
    base_id = st.selectbox("⚖️ Run di Riferimento (BASE):", run_keys, index=base_idx)
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

# Matrice Statistica Completa
lapse_rate = df_target["t850"] - df_target["t500"]
stats_summary = {
    "t850_max": float(df_target["t850"].max()),
    "t850_min": float(df_target["t850"].min()),
    "t500_min": float(df_target["t500"].min()),
    "t2m_max": float(df_target["t2m"].max()),
    "max_lapse_rate_850_500": float(lapse_rate.max()),
    "max_cape_ensemble": float(df_target["cape_max"].max()),
    "max_dewpoint": float(df_target["dewpoint"].max()),
    "max_rain_rate": float(df_target["precip"].max()),
    "total_accum": float(df_target["precip_accum"].iloc[-1]) if not df_target.empty else 0.0,
    "max_dls_knots": float(df_target["dls_knots"].max()),
    "delta_t850_trend": float(delta_df["delta_t850"].iloc[min(len(delta_df)-1, int(horizon_hours/2))]) if not delta_df.empty else 0.0
}
stats_json_str = json.dumps(stats_summary, indent=2)

st.markdown("---")

# 5 SCHEDE GRAFICHE INTERATTIVE CON TUTOR DINAMICO DEDICATO
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
    
    st.markdown("#### 🧭 La Lezione del Meteorologo (Termodinamica)")
    st.info(get_tab_ai_explanation("thermo", target_id, base_id, horizon_mode, stats_json_str))

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
    
    st.markdown("#### 🧭 La Lezione del Meteorologo (Carburante Convettivo)")
    st.info(get_tab_ai_explanation("cape", target_id, base_id, horizon_mode, stats_json_str))

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
    
    st.markdown("#### 🧭 La Lezione del Meteorologo (Dinamica Precipitativa)")
    st.info(get_tab_ai_explanation("rain", target_id, base_id, horizon_mode, stats_json_str))

with tab4:
    fig_w = go.Figure()
    fig_w.add_trace(go.Scatter(x=df_target.index, y=df_target["dls_knots"], name="Deep-Layer Shear 0-6km (kts)", line=dict(color="#ff9933", width=2.5)))
    fig_w.add_hline(y=35, line_dash="dash", line_color="red", annotation_text="Soglia Supercellare (>35 kts)")
    fig_w.add_hline(y=20, line_dash="dot", line_color="yellow", annotation_text="Soglia Multicellare (>20 kts)")
    fig_w.update_layout(title=f"Cinematica: Deep-Layer Shear 0-6 km (kts) [{horizon_mode}]", template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig_w, use_container_width=True)
    
    st.markdown("#### 🧭 La Lezione del Meteorologo (Wind Shear & Mesocicloni)")
    st.info(get_tab_ai_explanation("shear", target_id, base_id, horizon_mode, stats_json_str))

with tab5:
    fig_d = go.Figure()
    common = df_target.index.intersection(df_base.index)
    delta_sub = delta_df.loc[common]
    colors = ['#ff4d4d' if v >= 0 else '#3399ff' for v in delta_sub["delta_t850"]]
    fig_d.add_trace(go.Bar(x=common, y=delta_sub["delta_t850"], marker_color=colors, name="Δ T850"))
    fig_d.update_layout(title=f"Deriva Termica Run-to-Run a 850 hPa [{horizon_mode}]", template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig_d, use_container_width=True)
    
    st.markdown("#### 🧭 La Lezione del Meteorologo (Deriva Sinottica)")
    st.info(get_tab_ai_explanation("delta", target_id, base_id, horizon_mode, stats_json_str))
