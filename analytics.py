import pandas as pd
import numpy as np

def compute_run_delta(df_base, df_target):
    common_index = df_base.index.intersection(df_target.index)
    df_b = df_base.loc[common_index]
    df_t = df_target.loc[common_index]
    
    delta_df = pd.DataFrame(index=common_index)
    delta_df["delta_t850"] = df_t["t850"] - df_b["t850"]
    delta_df["delta_t500"] = df_t["t500"] - df_b["t500"]
    delta_df["delta_t2m"] = df_t["t2m"] - df_b["t2m"]
    delta_df["delta_precip"] = df_t["precip"] - df_b["precip"]
    return delta_df

def generate_stormchaser_briefing(df_target, delta_df, target_id, base_id):
    """
    Genera un'analisi dettagliata da Stormchaser tarata sul microclima di Olgiate Olona.
    """
    horizon_idx = min(168, len(df_target) - 1)
    target_7d = df_target["t850"].iloc[horizon_idx]
    delta_7d = delta_df["delta_t850"].iloc[min(horizon_idx, len(delta_df)-1)]
    target_std = df_target["t850_std"].iloc[horizon_idx]
    
    # 1. Analisi Trend Sinottico
    if delta_7d >= 1.5:
        synoptic_summary = (
            f"**Trend in Riscaldamento / Rallentamento Fronte (+{delta_7d:.1f}°C a 7d):** "
            "I cluster stanno ritardando l'ingresso della saccatura atlantica. "
            "Possibile sprofondamento occidentale ('falla iberica') che attiva un richiamo caldo prefrontale "
            "africano, accumulando ulteriore umidità nei bassi strati lungo la Valle Olona."
        )
    elif delta_7d <= -1.5:
        synoptic_summary = (
            f"**Trend in Raffreddamento / Anticipo Fronte ({delta_7d:.1f}°C a 7d):** "
            "I cluster accelerano la progressione del cavo d'onda atlantico. "
            "L'irruzione fredda in quota viene vista impattare più rapidamente contro le Alpi occidentali."
        )
    else:
        synoptic_summary = (
            f"**Assetto Sinottico Confermato ({delta_7d:+.1f}°C a 7d):** "
            "Elevata coerenza tra le ultime emissioni. La traiettoria del flusso perturbato è stabile."
        )

    # 2. Scansione Eventi di Instabilità
    lapse_rate = df_target["t850"] - df_target["t500"]
    severe_windows = []
    
    for t, lr in lapse_rate.items():
        rain = df_target.loc[t, "precip"]
        t850 = df_target.loc[t, "t850"]
        t500 = df_target.loc[t, "t500"]
        t2m = df_target.loc[t, "t2m"]
        
        if lr >= 27.5 and rain >= 0.8:
            if lr >= 30.5 and rain >= 3.0:
                threat = "🔴 RISCHIO SEVERO: Possibile Innesco Supercellare / Grandine Grossa"
                desc = (
                    f"Gradiente termico verticale estremo ($\Delta T = {lr:.1f}^\circ\text{{C}}$) con $T_{{500}}$ a {t500:.1f}°C. "
                    "Se al suolo insiste il richiamo da Sud-Est lungo la pianura, l'impatto contro le Prealpi Varesine "
                    "(Campo dei Fiori) fornirà l'innesco forzato ideale per forti rotazioni negli updraft (alto SRH)."
                )
            elif lr >= 29.0:
                threat = "🟠 RISCHIO MODERATO: Multicelle Organizzate / Grandine Media"
                desc = (
                    f"Forte instabilità termodinamica ($\Delta T = {lr:.1f}^\circ\text{{C}}$). "
                    "Favorevole allo sviluppo di cluster temporaleschi e squall line a rapida propagazione."
                )
            else:
                threat = "🟡 RISCHIO CONVETTIVO: Rovesci Temporaleschi Sparsi"
                desc = f"Instabilità ordinaria ($\Delta T = {lr:.1f}^\circ\text{{C}}$). Convezione pomeridiana o passaggio instabile."
                
            severe_windows.append({
                "timestamp": t.strftime("%A %d/%m ore %H:%M UTC"),
                "threat": threat,
                "lapse_rate": lr,
                "t850": t850,
                "t500": t500,
                "rain": rain,
                "desc": desc
            })
            
    return synoptic_summary, severe_windows, target_std
  
