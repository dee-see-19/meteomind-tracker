import os
import json
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="MeteoMind Stormchaser Lab", layout="wide")

# =========================================================
# FEATURE EXTRACTOR: ANALISI PUNTUALE DELLA SERIE TEMPORALE
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

def extract_chronological_features(df, delta_df):
    """
    Estrae cronologicamente eventi salienti, cali termici, picchi e overlap convettivi.
    """
    lapse_rate = df["t850"] - df["t500"]
    features = {
        "start_time": df.index[0].strftime("%d/%m %H:%M UTC"),
        "end_time": df.index[-1].strftime("%d/%m %H:%M UTC"),
        "t850_max": float(df["t850"].max()),
        "t850_max_time": df["t850"].idxmax().strftime("%a %d/%m %H:%M UTC"),
        "t850_min": float(df["t850"].min()),
        "t850_min_time": df["t850"].idxmin().strftime("%a %d/%m %H:%M UTC"),
        "t500_min": float(df["t500"].min()),
        "t500_min_time": df["t500"].idxmin().strftime("%a %d/%m %H:%M UTC"),
        "t2m_max": float(df["t2m"].max()),
        "t2m_max_time": df["t2m"].idxmax().strftime("%a %d/%m %H:%M UTC"),
        "max_lapse_rate": float(lapse_rate.max()),
        "max_lapse_rate_time": lapse_rate.idxmax().strftime("%a %d/%m %H:%M UTC"),
        "max_cape": float(df["cape_max"].max()),
        "max_cape_time": df["cape_max"].idxmax().strftime("%a %d/%m %H:%M UTC"),
        "max_dp": float(df["dewpoint"].max()),
        "max_dp_time": df["dewpoint"].idxmax().strftime("%a %d/%m %H:%M UTC"),
        "min_dp": float(df["dewpoint"].min()),
        "max_rain": float(df["precip"].max()),
        "max_rain_time": df["precip"].idxmax().strftime("%a %d/%m %H:%M UTC"),
        "total_accum": float(df["precip_accum"].iloc[-1]) if "precip_accum" in df else 0.0,
        "max_dls": float(df["dls_knots"].max()),
        "max_dls_time": df["dls_knots"].idxmax().strftime("%a %d/%m %H:%M UTC"),
        "supercell_overlap_windows": [],
        "foehn_signals": [],
        "thermal_drops": []
    }

    # 1. Rilevamento Overlap Supercellare (CAPE elevato + Forte Shear)
    for t in df.index:
        c = df.loc[t, "cape_max"]
        s = df.loc[t, "dls_knots"]
        lr = lapse_rate.loc[t]
        if c >= 1400 and s >= 32:
            features["supercell_overlap_windows"].append({
                "time": t.strftime("%a %d/%m ore %H:%M UTC"),
                "cape": float(c),
                "dls": float(s),
                "lr": float(lr)
            })

    # 2. Rilevamento Segnali di Fohn (Aria secca con crollo Dew Point sotto 10 C)
    for t in df.index:
        dp_val = df.loc[t, "dewpoint"]
        t2m_val = df.loc[t, "t2m"]
        rain_val = df.loc[t, "precip"]
        if dp_val <= 9.5 and t2m_val >= 20.0 and rain_val == 0.0:
            features["foehn_signals"].append(t.strftime("%a %d/%m %H:%M UTC"))

    # 3. Rilevamento Crollo Termico Frontale (Delta 24h a 850 hPa)
    if len(df) >= 24:
        t850_diff_24h = df["t850"].diff(24)
        min_drop = t850_diff_24h.min()
        if min_drop <= -5.0:
            drop_time = t850_diff_24h.idxmin()
            features["thermal_drops"].append({
                "drop_val": float(min_drop),
                "time": drop_time.strftime("%a %d/%m %H:%M UTC")
            })

    # 4. Deriva Trend Run-to-Run
    if not delta_df.empty:
        features["delta_t850_mean"] = float(delta_df["delta_t850"].mean())
        features["delta_t850_max_pos"] = float(delta_df["delta_t850"].max())
        features["delta_t850_max_neg"] = float(delta_df["delta_t850"].min())

    return features

# =========================================================
# MOTORE GENERATORE DI ANALISI DETTAGLIATA METEOMIND
# =========================================================

def build_dynamic_expert_analysis(tab_name, feat, horizon_mode):
    """
    Costruisce una diagnosi puntuale e ricca di dettagli fisici e microclimatici.
    """
    if tab_name == "thermo":
        drop_txt = ""
        if feat["thermal_drops"]:
            d = feat["thermal_drops"][0]
            drop_txt = f"\n* 📉 **Firma del Fronte Freddo:** Registrato un crollo termico marcato a 850 hPa di **{d['drop_val']:.1f} °C in 24h** attorno a **{d['time']}**, che identifica l'irruzione della massa d'aria polare-marittima."

        return (
            f"### 🔬 Diagnosi Termodinamica & Profilo Verticale ({horizon_mode})\n"
            f"* ⏱️ **Intervallo Esaminato:** Da {feat['start_time']} a {feat['end_time']}.\n"
            f"* 🌡️ **Escursione Termica a 850 hPa (~1500m):** Picco massimo di **+{feat['t850_max']:.1f} °C** ({feat['t850_max_time']}) che scende fino a un minimo di **+{feat['t850_min']:.1f} °C** ({feat['t850_min_time']}).\n"
            f"* ❄️ **Minimo Termico in Quota (500 hPa / ~5500m):** Il termometro in media troposfera sprofonda a **{feat['t500_min']:.1f} °C** ({feat['t500_min_time']}).\n"
            f"* ⚡ **Picco del Lapse Rate (Delta T 850-500):** Il gradiente termico verticale massimo tocca **{feat['max_lapse_rate']:.1f} °C** ({feat['max_lapse_rate_time']}), corrispondente a un tasso reale di **{feat['max_lapse_rate']/4:.2f} °C/km**.\n"
            f"* 🔎 **Lettura Fisica da Stormchaser:** Un gradiente termico verticale superiore a 7.0 °C/km indica che l'aria fredda atlantica in quota scivola sopra lo strato limite padano caldo. "
            f"La colonna d'aria si trova in uno stato di **instabilità super-adiabatica condizionale**: qualsiasi sollevamento forzato dal suolo trasforma l'energia latente in violenta accelerazione ascensionale.{drop_txt}"
        )

    elif tab_name == "cape":
        w_max = np.sqrt(2 * feat["max_cape"]) if feat["max_cape"] > 0 else 0
        lcl = max(200, 125 * (feat["t2m_max"] - feat["max_dp"]))

        overlap_txt = "Nessuna sovrapposizione critica tra CAPE estremo e forte shear nell'orizzonte selezionato."
        if feat["supercell_overlap_windows"]:
            w = feat["supercell_overlap_windows"][0]
            overlap_txt = f"⚠️ **Finestra di Rischio Supercellare Rilevata a {w['time']}:** CAPE di picco a **{w['cape']:.0f} J/kg** concomitante con Deep-Layer Shear a **{w['dls']:.0f} nodi** e Lapse Rate di **{w['lr']:.1f} °C**."

        foehn_txt = ""
        if feat["foehn_signals"]:
            foehn_txt = f"\n* 🏜️ **Segnale di Föhn / Aria Secca:** Individuato un crollo del Dew Point al suolo a **{feat['min_dp']:.1f} °C** ({feat['foehn_signals'][0]}). Indica correnti di caduta da Nord/Nord-Ovest che comprimono e asciugano l'aria, sterilizzando temporaneamente l'instabilità convettiva."

        return (
            f"### ⚡ Diagnosi Carburante Convettivo & Dinamica dell'Updraft ({horizon_mode})\n"
            f"* 💥 **Picco di Energia CAPE:** Raggiunge **{feat['max_cape']:.0f} J/kg** ({feat['max_cape_time']}).\n"
            f"* 🚀 **Velocità Ascensionale Massima (w_max = sqrt(2 * CAPE)):** L'aria all'interno dell'updraft può accelerare teoricamente fino a **{w_max:.1f} m/s ({w_max*3.6:.0f} km/h)**. Valori oltre i 40-50 m/s sostengono idrometeore congelate pesanti, innescando **grandinate con chicchi di diametro > 3-5 cm**.\n"
            f"* 💧 **Umidità nei Bassi Strati (Dew Point Td):** Picco di rugiada a **{feat['max_dp']:.1f} °C** ({feat['max_dp_time']}). Valori >= 19-21 °C indicano che la Valle Olona è carica di vapore d'acqua.\n"
            f"* ☁️ **Quota Base Nubi (LCL):** Stimata a circa **{lcl:.0f} metri** durante le ore più calde. Una base nube bassa (< 1000m) riduce l'evaporazione sotto la nube e sostiene forti rotazioni nei bassi strati.\n"
            f"* 🎯 **Sintesi Instabilità:** {overlap_txt}{foehn_txt}"
        )

    elif tab_name == "rain":
        return (
            f"### 🌧️ Diagnosi Precipitazioni, Downdraft & Cold Pool ({horizon_mode})\n"
            f"* 🌊 **Picco di Intensità Oraria:** Previsti fino a **{feat['max_rain']:.1f} mm/h** ({feat['max_rain_time']}).\n"
            f"* 📊 **Accumulo Cumulato Totale:** Raggiunge i **{feat['total_accum']:.1f} mm** sull'intero periodo.\n"
            f"* 🌪️ **Microfisica delle Correnti Discendenti (Downdraft):** L'evaporazione parziale della pioggia e della grandine negli strati intermedi (evaporative cooling) genera aria fredda e densa che precipita al suolo.\n"
            f"* ⚠️ **Rischio Downburst su Olgiate Olona:** L'impatto della sacca fredda (cold pool) contro la pianura riscaldata genera raffiche orizzontali lineari di **downburst oltre gli 80-100 km/h**. L'avanzamento del fronte freddo al suolo (outflow boundary) funge da cuneo che solleva nuova aria umida, innescando celle convettive a catena verso Busto Arsizio e Legnano."
        )

    elif tab_name == "shear":
        structure = "Supercella Mesociclonica" if feat["max_dls"] >= 35 else ("Multicella / Squall Line Organizzata" if feat["max_dls"] >= 20 else "Temporale a Cella Singola (Pulse Storm)")
        return (
            f"### 💨 Diagnosi Cinematica: Wind Shear & Struttura dei Temporali ({horizon_mode})\n"
            f"* 🌀 **Picco di Deep-Layer Shear (DLS 0-6km):** Differenziale di vento tra suolo (10m) e quota (5500m) pari a **{feat['max_dls']:.0f} nodi (kts)** ({feat['max_dls_time']}).\n"
            f"* 📐 **Morfologia Convettiva Attesa:** In base al valore di picco, la struttura dominante stimata è: **{structure}**.\n"
            f"* ⚙️ **Meccanismo di Auto-Sostentamento:** Uno shear >= 35 kts inclina l'asse dell'updraft. La pioggia e la grandine cadono a valle senza soffocare la corrente ascensionale, consentendo alla cella di rigenerarsi per diverse ore.\n"
            f"* 🏔️ **Forzante Orografica del Campo dei Fiori:** A Olgiate Olona, l'aria calda da Sud-Est converge verso nord contro i primi rilievi prealpini. L'elicità atmosferica (Storm-Relative Helicity) inietta rotazione nell'updraft, creando le condizioni ideali per **supercelle grandinigene con moti rotatori** in discesa verso l'alto Milanese."
        )

    elif tab_name == "delta":
        delta_val = feat.get("delta_t850_mean", 0.0)
        max_pos = feat.get("delta_t850_max_pos", 0.0)
        max_neg = feat.get("delta_t850_max_neg", 0.0)

        if delta_val >= 1.2 or max_pos >= 2.5:
            synop_trend = f"🔴 **Trend in Riscaldamento / Rallentamento della Saccatura (Picco variazione +{max_pos:.1f} °C):** Il run attuale ritarda l'affondo perturbato. Si profila una **falla iberica** (sprofondamento occidentale) che attiva un richiamo sciroccale molto caldo, prolungando l'accumulo di calore sensibile sulla pianura lombarda."
        elif delta_val <= -1.2 or max_neg <= -2.5:
            synop_trend = f"🔵 **Trend in Raffreddamento / Anticipo del Fronte (Picco variazione {max_neg:.1f} °C):** Il modello velocizza l'ingresso della goccia fredda sul Nord-Ovest, anticipando l'innesco dei primi temporali frontali."
        else:
            synop_trend = f"⚪ **Assetto Sinottico Confermato (Variazione media {delta_val:+.1f} °C):** Elevata stabilità probabilistica e allineamento tra i cluster ensemble."

        return (
            f"### 📊 Diagnosi Deriva Modellistica & Onde di Rossby ({horizon_mode})\n"
            f"* 🔄 **Comportamento Run-to-Run:** {synop_trend}\n"
            f"* 🛰️ **Analisi della Dispersione:** Nel breve termine (3 giorni) la traiettoria è ad alta precisione; oltre i 7-10 giorni le oscillazioni delle barre rosse e blu evidenziano la sensibilità caotica alle condizioni iniziali delle onde planetarie di Rossby."
        )

@st.cache_data(show_spinner=False)
def get_advanced_meteomind_briefing(tab_name, target_id, base_id, horizon_mode, feat_json_str):
    """Chiama l'AI Gemini per un'analisi personalizzata o esegue il generatore locale esperto."""
    feat = json.loads(feat_json_str)
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    except Exception:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return build_dynamic_expert_analysis(tab_name, feat, horizon_mode)

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""
Sei MeteoMind, meteorologo senior e stormchaser di altissimo livello specializzato nella dinamica dei temporali severi nel Nord Italia (Olgiate Olona, Valle Olona, fascia pedemontana del Campo dei Fiori).
Analizza i DATI REALI ed ESTRAI I DETTAGLI CRONOLOGICI SPECIFICI per la scheda '{tab_name}' sull'orizzonte '{horizon_mode}'.

NON FARE DEFINIZIONI GENERICHE DA DIZIONARIO. Devi commentare esattamente le date, le ore, i picchi e le firme atmosferiche presenti in questi dati:
{feat_json_str}

Spiega la fisica reale che accade nell'aria sopra Olgiate Olona, chiarendo termini complessi (Lapse Rate, CAPE, Dew Point, LCL, Downdraft, Downburst, Deep-Layer Shear, Supercella, Falla Iberica, Fohn) in relazione a queste specifiche cifre e date.
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        return build_dynamic_expert_analysis(tab_name, feat, horizon_mode)

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
feat_dict = extract_chronological_features(df_target, delta_df)
feat_json = json.dumps(feat_dict, indent=2)

st.markdown("---")

# 5 SCHEDE GRAFICHE INTERATTIVE CON TUTOR DINAMICO PUNTUALE
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

    st.markdown("#### 🧭 La Diagnosi del Meteorologo (Termodinamica & Quota)")
    st.info(get_advanced_meteomind_briefing("thermo", target_id, base_id, horizon_mode, feat_json))

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

    st.markdown("#### 🧭 La Diagnosi del Meteorologo (Carburante & Updraft)")
    st.info(get_advanced_meteomind_briefing("cape", target_id, base_id, horizon_mode, feat_json))

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

    st.markdown("#### 🧭 La Diagnosi del Meteorologo (Microfisica & Downdrafts)")
    st.info(get_advanced_meteomind_briefing("rain", target_id, base_id, horizon_mode, feat_json))

with tab4:
    fig_w = go.Figure()
    fig_w.add_trace(go.Scatter(x=df_target.index, y=df_target["dls_knots"], name="Deep-Layer Shear 0-6km (kts)", line=dict(color="#ff9933", width=2.5)))
    fig_w.add_hline(y=35, line_dash="dash", line_color="red", annotation_text="Soglia Supercellare (>35 kts)")
    fig_w.add_hline(y=20, line_dash="dot", line_color="yellow", annotation_text="Soglia Multicellare (>20 kts)")
    fig_w.update_layout(title=f"Cinematica: Deep-Layer
