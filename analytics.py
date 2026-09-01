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

def generate_dynamic_explanations(df_target, delta_df):
    """
    Analizza i valori quantitativi del run attivo e genera spiegazioni
    didattiche dinamiche per ciascuna sezione e grafico.
    """
    # 1. Analisi Lapse Rate e Profilo Termico
    lapse_rate = df_target["t850"] - df_target["t500"]
    max_lr = lapse_rate.max()
    max_lr_time = lapse_rate.idxmax()
    t850_at_max = df_target.loc[max_lr_time, "t850"]
    t500_at_max = df_target.loc[max_lr_time, "t500"]
    
    if max_lr >= 30.0:
        thermo_exp = (
            f"🔥 **Gradiente Termico Verticale Estremo ($\Delta T = {max_lr:.1f}^\circ\\text{{C}}$ atteso il {max_lr_time.strftime('%d/%m ore %H:%M UTC')}):** "
            f"A circa 1500m abbiamo un'isoterma di **{t850_at_max:.1f}°C**, mentre a 5500m entra aria fredda a **{t500_at_max:.1f}°C**. "
            "Il lapse rate supera i $7.5^\circ\\text{C/km}$: la colonna d'aria è un detonatore. "
            "Ogni bolla d'aria che supera il LFC (Livello di Libera Convezione) accelererà verso l'alto con violenza, favorendo nubi a sviluppo verticale esplosivo."
        )
    elif max_lr >= 27.5:
        thermo_exp = (
            f"⚡ **Gradiente Instabile Moderato ($\Delta T = {max_lr:.1f}^\circ\\text{{C}}$ il {max_lr_time.strftime('%d/%m ore %H:%M')}):** "
            f"Isoterma 500 hPa a **{t500_at_max:.1f}°C** sopra una 850 hPa a **{t850_at_max:.1f}°C**. "
            "Condizioni ideali per convezione organizzata (multicelle o squall line) se supportata da forzante orografica o convergenza nei bassi strati."
        )
    else:
        thermo_exp = (
            f"🟢 **Profilo Termico Stabile o Poco Instabile (Max $\Delta T = {max_lr:.1f}^\circ\\text{{C}}$):** "
            "Il gradiente verticale non evidenzia anomalie fredde marcate in quota. Rischio grandinigeno basso, possibili solo rovesci ordinari o piogge da sbarramento."
        )

    # 2. Analisi CAPE e Dew Point (Carburante Convettivo)
    max_cape = df_target["cape_max"].max()
    max_cape_time = df_target["cape_max"].idxmax()
    dp_at_max = df_target.loc[max_cape_time, "dewpoint"]
    w_max = np.sqrt(2 * max_cape) if max_cape > 0 else 0
    
    if max_cape >= 2500:
        cape_exp = (
            f"💥 **Carico Energetico Severo (Picco CAPE: {max_cape:.0f} J/kg il {max_cape_time.strftime('%d/%m ore %H:%M UTC')}):** "
            f"Con un Dew Point a terra stimato a **{dp_at_max:.1f}°C**, l'umidità specifica nei primi 1000m della Valle Olona è altissima. "
            f"Velocità teorica massima dell'updraft: **$w_{{max}} \\approx {w_max:.1f}\\text{{ m/s}}$ ({w_max*3.6:.0f} km/h)**. "
            "Correnti ascensionali di questa violenza sostengono chicchi di grandine di grosse dimensioni (>4-5 cm) prima del collasso a terra."
        )
    elif max_cape >= 1200:
        cape_exp = (
            f"⚡ **Energia Convettiva Moderata (Picco CAPE: {max_cape:.0f} J/kg | Dew Point: {dp_at_max:.1f}°C):** "
            f"Updraft teorico massimo di circa **{w_max:.1f} m/s**. Sufficiente per innescare grandinate medie e intense raffiche di downdraft lineare."
        )
    else:
        cape_exp = (
            f"💧 **Energia Bassa o Assente (Max CAPE: {max_cape:.0f} J/kg):** "
            "Manca il carburante termodinamico nei bassi strati o l'aria è asciutta ($T_d$ basso). Possibili solo fenomeni stratiformi o convezione debole."
        )

    # 3. Analisi Delta Run-to-Run (Deriva Sinottica)
    horizon_idx = min(168, len(delta_df) - 1)
    delta_7d = delta_df["delta_t850"].iloc[horizon_idx]
    
    if delta_7d >= 1.5:
        delta_exp = (
            f"🔴 **Deriva in Riscaldamento (+{delta_7d:.1f}°C a +7 giorni):** "
            "Il run attivo ha ritardato l'avanzata della saccatura. Dinamica tipica da **falla iberica**: la perturbazione affonda a ovest del Portogallo "
            "e attiva sulla Lombardia un richiamo di Scirocco caldo e secco. Questo carica ulteriormente la molla termodinamica per i giorni successivi."
        )
    elif delta_7d <= -1.5:
        delta_exp = (
            f"🔵 **Deriva in Raffreddamento ({delta_7d:.1f}°C a +7 giorni):** "
            "Il modello anticipa l'irruzione fredda. L'asse della saccatura entra più franco verso il Golfo del Leone: "
            "il fronte temporalesco impatterà le Prealpi occidentali in anticipo."
        )
    else:
        delta_exp = (
            f"⚪ **Coerenza Modellistica Elevata ({delta_7d:+.1f}°C a +7 giorni):** "
            "Nessuna oscillazione significativa rispetto al run precedente. Il timing e la traiettoria del fronte sono stabili."
        )

    # 4. Scanner Locale Microclima Olgiate Olona / Varesotto
    local_alerts = []
    for t, lr in lapse_rate.items():
        rain = df_target.loc[t, "precip"]
        cape = df_target.loc[t, "cape_max"]
        t500 = df_target.loc[t, "t500"]
        t850 = df_target.loc[t, "t850"]
        dp = df_target.loc[t, "dewpoint"]
        
        # Filtro per evento severo
        if lr >= 28.0 and (rain >= 1.0 or cape >= 1500):
            if cape >= 2200 and lr >= 30.0:
                structure = "🔴 **Potenziale Supercellare con Rischio Grandine Grossa e Downburst**"
                micro_note = (
                    "Il richiamo caldo-umido padano impatta contro il massiccio del Campo dei Fiori fornendo la forzante meccanica. "
                    "Se i venti a 500 hPa mantengono un vettore sudoccidentale teso (>40 kts), l'updraft acquisisce rotazione ciclonica (mesociclone)."
                )
            elif cape >= 1200:
                structure = "🟠 **Multicella Intensa / Linea di Groppo (Squall Line)**"
                micro_note = (
                    "Possibile formazione di un sistema convettivo a mesoscala (MCS) in discesa dalle valli varesine e comasche verso Olgiate Olona e l'alto milanese."
                )
            else:
                structure = "🟡 **Rovesci Convettivi / Temporali di Sbarramento Orografico**"
                micro_note = "Temporali a cella singola o piogge da sollevamento orografico senza rotazione organizzata."
                
            local_alerts.append({
                "time_str": t.strftime("%A %d/%m ore %H:%M UTC"),
                "structure": structure,
                "micro_note": micro_note,
                "lr": lr,
                "t500": t500,
                "t850": t850,
                "cape": cape,
                "dp": dp,
                "rain": rain
            })

    return thermo_exp, cape_exp, delta_exp, local_alerts
    
