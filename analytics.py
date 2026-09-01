import pandas as pd
import numpy as np

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

def generate_deep_meteorological_analysis(df_target, delta_df, horizon_hours):
    """
    Motore analitico avanzato: compila un bollettino quantitativo
    sul microclima di Olgiate Olona e della Pedemontana Varesina.
    """
    lapse_rate = df_target["t850"] - df_target["t500"]
    spread_depr = df_target["t2m"] - df_target["dewpoint"]
    lcl_height_m = (125 * spread_depr).clip(lower=200)
    w_max_series = np.sqrt(2 * df_target["cape_max"].clip(lower=0))
    
    check_idx = min(len(delta_df) - 1, int(horizon_hours / 2))
    delta_val = delta_df["delta_t850"].iloc[check_idx] if not delta_df.empty else 0
    
    if delta_val >= 1.5:
        synoptic_briefing = (
            f"🔴 **DERIVA IN RISCALDAMENTO / RITARDATA EVOLUZIONE (+{delta_val:.1f}°C a medio termine):** "
            "I cluster frenano l'avanzata della saccatura atlantica. Rischio di affondo troppo occidentale "
            "('falla iberica') con richiamo sciroccale caldo: aumenta l'accumulo di umidità nei bassi strati lungo la Valle Olona."
        )
    elif delta_val <= -1.5:
        synoptic_briefing = (
            f"🔵 **DERIVA IN RAFFREDDAMENTO / ANTICIPO DEL FRONTE ({delta_val:.1f}°C a medio termine):** "
            "I cluster accelerano l'ingresso dell'asse di saccatura. L'irruzione fredda a 500 hPa impatterà le Prealpi occidentali in anticipo."
        )
    else:
        synoptic_briefing = (
            f"⚪ **ASSETTO SINOTTICO CONSOLIDATO ({delta_val:+.1f}°C):** "
            "I membri ensemble mantengono elevata coerenza di traiettoria rispetto all'uscita precedente."
        )

    events = []
    for t in df_target.index:
        lr = lapse_rate.loc[t]
        cape = df_target.loc[t, "cape_max"]
        dp = df_target.loc[t, "dewpoint"]
        t2m = df_target.loc[t, "t2m"]
        rain = df_target.loc[t, "precip"]
        dls = df_target.loc[t, "dls_knots"]
        w_max = w_max_series.loc[t]
        lcl = lcl_height_m.loc[t]
        
        if lr >= 27.5 and (cape >= 1000 or rain >= 1.0):
            if cape >= 2200 and dls >= 35:
                mode = "🔴 **RISCHIO SEVERO: Possibile Supercella Isolata / Grandine Grossa (>3-5 cm)**"
                detail = (
                    f"Forte instabilità ($w_{{max}} \\approx {w_max:.0f}\\text{{ m/s}}$) con DLS sostenuto ({dls:.0f} kts). "
                    f"Base nube bassa ($LCL \\approx {lcl:.0f}\\text{{ m}}$). Forzante orografica ideale contro il Campo dei Fiori."
                )
            elif dls >= 25 or rain >= 3.0:
                mode = "🟠 **RISCHIO ELEVATO: Sistema Multicellare / Squall Line con Raffiche Lineari**"
                detail = f"Shear moderato ({dls:.0f} kts) con forte instabilità. Rischio downburst e grandine media."
            else:
                mode = "🟡 **RISCHIO MODERATO: Temporali Termoconvettivi o di Sbarramento**"
                detail = f"Convezione a prevalente forzante orografica o termica ($w_{{max}} \\approx {w_max:.0f}\\text{{ m/s}}$)."
                
            events.append({
                "time_str": t.strftime("%A %d/%m ore %H:%M UTC"),
                "mode": mode, "detail": detail,
                "lr": lr, "cape": cape, "dp": dp, "t2m": t2m,
                "dls": dls, "w_max": w_max, "lcl": lcl, "rain": rain
            })
            
    max_lr_val = lapse_rate.max()
    max_cape_val = df_target["cape_max"].max()
    
    thermo_guide = (
        f"**Diagnosi Termodinamica:** Massimo Lapse Rate $\Delta T_{{850-500}}$ a **{max_lr_val:.1f}°C** "
        f"($\\Gamma \\approx {max_lr_val/4:.2f}^\\circ\\text{{C/km}}$). "
        + ("Gradiente ripido: forte potenziale esplosivo." if max_lr_val >= 28.5 else "Gradiente nei ranghi.")
    )
    cape_guide = (
        f"**Diagnosi Carburante Convettivo:** Picco CAPE massimo a **{max_cape_val:.0f} J/kg** "
        f"($w_{{max}} \\approx {np.sqrt(2*max_cape_val):.0f}\\text{{ m/s}}$). "
        + ("Energia elevata con rischio severo." if max_cape_val >= 2000 else "Energia moderata o ordinaria.")
    )
    
    return synoptic_briefing, events, thermo_guide, cape_guide
