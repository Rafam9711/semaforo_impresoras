# -*- coding: utf-8 -*-
import io
import re
import unicodedata
from datetime import timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# =============== Utilidades ===============
def normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def contains_any(txt: str, keywords: List[str]) -> bool:
    return any(k in txt for k in keywords)

def drop_columns_robust(df: pd.DataFrame, to_remove: List[str]) -> pd.DataFrame:
    norm_cols = {c: normalize(c) for c in df.columns}
    remove_norms = {normalize(x) for x in to_remove}
    cols_to_drop = [orig for orig, n in norm_cols.items() if n in remove_norms]
    return df.drop(columns=cols_to_drop, errors="ignore")


# =============== Diccionarios NLP ===============
TAXONOMY = {
    "fusor":      ["fusor", "fuser", "fusion", "calentador"],
    "electronica":["placa", "board", "fuente", "motor", "sensor", "corto"],
    "atascos":    ["atasco", "paper jam", "atascada", "atoramiento", "obstruccion", "obstrucción"],
    "rodillos":   ["rodillo", "pickup roller", "arrastre", "empuje"],
    "consumibles":["toner", "tóner", "cartucho", "drum", "cilindro"],
    "adf_bandeja":["adf", "alimentador", "bandeja", "duplex", "dúplex"],
    "red_driver": ["wifi", "ethernet", "controlador", "driver", "firmware", "ip"],
    "calidad":    ["lineas", "líneas", "manchas", "borroso", "sombra", "ghosting", "rayas"],
}
CRITICAL_COMPS = {"fusor", "electronica"}

SEVERITY_HIGH = ["no enciende", "humo", "quemad", "corto", "fusor roto", "placa", "motor trabado", "error critico", "error crítico", "error fatal", "no imprime nada"]
SEVERITY_MED  = ["atasco recurrente", "rodillo", "sensor", "ruidos", "lineas severas", "líneas severas", "drum", "cilindro"]
SEVERITY_LOW  = ["limpieza", "configuracion", "configuración", "driver", "reinicio", "actualizacion", "actualización"]
REPEAT_HINTS  = ["otra vez", "nuevamente", "sigue igual", "persistente", "reincide", "mismo problema"]
SERVICE_ONLY_TYPES = {"entrega de mercancia", "entrega de mercancía"}
NEG_PAT = r"\bno\b[^.]{0,30}\b(fusor|placa|rodillo|sensor|drum|cilindro|motor)\b"

COMP_COLS = ["fusor","electronica","atascos","rodillos","consumibles","adf_bandeja","red_driver","calidad"]


# =============== Zero-shot opcional ===============
class ZeroShotWrapper:
    """Pequeño wrapper para no acoplar 'transformers' aquí dentro."""
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.labels = ["fusor","electrónica","atascos","rodillos","consumibles","ADF/bandeja","red/driver","calidad de impresión"]
        self.map = {
            "fusor":"fusor",
            "electrónica":"electronica",
            "atascos":"atascos",
            "rodillos":"rodillos",
            "consumibles":"consumibles",
            "adf/bandeja":"adf_bandeja",
            "red/driver":"red_driver",
            "calidad de impresión":"calidad",
        }

    def __call__(self, text: str, threshold: float = 0.6) -> List[str]:
        out = self.pipeline(text, self.labels, hypothesis_template="El problema está relacionado con {}.")
        labs = []
        for lbl, scr in zip(out["labels"], out["scores"]):
            if float(scr) >= threshold and lbl.lower() in self.map:
                labs.append(self.map[lbl.lower()])
        return list(set(labs))


# =============== Clasificación por ticket ===============
def classify_row(row: pd.Series,
                 col_asunto: str, col_resol: str, col_tipo: str,
                 use_negations: bool = True,
                 zs: Optional[ZeroShotWrapper] = None,
                 zs_thr: float = 0.6) -> dict:
    txt = normalize(f"{row.get(col_asunto, '')} {row.get(col_resol, '')} {row.get(col_tipo, '')}")
    cats = {k: 0 for k in TAXONOMY.keys()}
    # Reglas
    for cat, kws in TAXONOMY.items():
        if contains_any(txt, kws):
            cats[cat] = 1
    # Negaciones
    if use_negations and re.search(NEG_PAT, txt):
        for k in ["fusor","electronica","rodillos","consumibles"]:
            if k in cats:
                cats[k] = 0
    # Zero-shot si no detectó nada
    if zs is not None and sum(cats.values()) == 0:
        for k in zs(txt, threshold=zs_thr):
            cats[k] = 1

    # Severidad
    sev = 0.0
    if contains_any(txt, SEVERITY_HIGH): sev = 2.0
    elif contains_any(txt, SEVERITY_MED): sev = 1.0
    elif contains_any(txt, SEVERITY_LOW): sev = 0.5
    if contains_any(txt, REPEAT_HINTS): sev += 0.5

    # Bajada de severidad para logísticos
    tipo_norm = normalize(row.get(col_tipo, ""))
    if tipo_norm in SERVICE_ONLY_TYPES:
        sev = max(0.0, sev - 0.5)

    critical = 1 if (cats["fusor"] == 1 or cats["electronica"] == 1) else 0
    return {**cats, "sev": sev, "critical": critical, "texto": txt}


# =============== Historial por componente ===============
def build_history_features(tickets: pd.DataFrame, col_fecha: str, half_life_days: int = 45) -> pd.DataFrame:
    events = []
    for _, r in tickets.iterrows():
        for c in COMP_COLS:
            if c in tickets.columns and r.get(c, 0) == 1:
                events.append({
                    "device_id": r["device_id"],
                    "date": r[col_fecha],
                    "comp": c,
                    "sev": float(r["sev"]),
                    "critical": int(c in CRITICAL_COMPS),
                    "texto": r["texto"]
                })
    events = pd.DataFrame(events).sort_values(["device_id","comp","date"]).reset_index(drop=True)
    if events.empty:
        return pd.DataFrame(columns=["device_id","date","same_comp_120d","same_comp_decay","return_after_replace_60d","critical"])

    REPLACE_PAT = r"(se\s+(cambio|cambió|reemplaz[oó]|sustituy[oó])\s+)"
    events["mentioned_replace"] = events["texto"].str.contains(REPLACE_PAT, regex=True).astype(int)

    rows_hist = []
    for (dev, comp), g in events.groupby(["device_id","comp"]):
        g = g.reset_index(drop=True)
        for _, row in g.iterrows():
            d = row["date"]
            prev120 = g[(g["date"] <= d) & (g["date"] > d - pd.Timedelta(days=120))].iloc[:-1]
            same_comp_120d = len(prev120)

            prev180 = g[(g["date"] <= d) & (g["date"] > d - pd.Timedelta(days=180))].iloc[:-1]
            decay_sum = 0.0
            for _, p in prev180.iterrows():
                delta = (d - p["date"]).days
                w = np.exp(-np.log(2) * delta / half_life_days)
                decay_sum += w * float(p["sev"])

            prev60 = g[(g["date"] <= d) & (g["date"] > d - pd.Timedelta(days=60))].iloc[:-1]
            return_after_replace_60d = 1 if (prev60["mentioned_replace"].sum() > 0) else 0

            rows_hist.append({
                "device_id": dev, "comp": comp, "date": d,
                "same_comp_120d": same_comp_120d,
                "same_comp_decay": decay_sum,
                "return_after_replace_60d": return_after_replace_60d,
                "critical": int(comp in CRITICAL_COMPS),
            })
    return pd.DataFrame(rows_hist)


# =============== Reglas de decisión ===============
def decision_history_first(row,
                           freq_thr=3,
                           sev_thr=1.0,
                           same_comp_thr=2,
                           decay_thr_crit=2.5,
                           recent_days_thr=14):
    if row.get("return_after_replace_60d", 0)==1 and row.get("critical_flag", 0)==1:
        return "ROJO", "Reemplazar: reincidencia ≤60d después de un 'cambio/reemplazo' en componente crítico."
    if row.get("same_comp_120d", 0) >= same_comp_thr and row.get("critical_flag", 0)==1:
        return "ROJO", "Reemplazar: misma parte crítica repetida (≥2) en 120d."
    if row.get("same_comp_decay", 0.0) >= decay_thr_crit and row.get("critical_flag", 0)==1:
        return "ROJO", "Reemplazar: historial reciente fuerte del mismo componente crítico."
    if row["freq_90d"] >= freq_thr and row["sev_avg_90d"] >= sev_thr:
        return "ROJO", "Reemplazar: alta frecuencia y severidad en 90d."
    if row["days_since_prev"] <= recent_days_thr and row["sev_avg_90d"] > 1.0:
        return "ROJO", "Reemplazar: falla severa muy reciente."
    if (row["freq_90d"] >= max(2, freq_thr - 1)) or (row["sev_avg_90d"] >= 0.75) or \
       (row.get("same_comp_120d", 0)==1 and row.get("critical_flag", 0)==1):
        return "AMBAR", "Revisar: señales elevadas (frecuencia/severidad o 1 incidencia de parte crítica)."
    return "VERDE", "Mantener: sin señales fuertes en el historial."


# =============== Explicación en lenguaje humano ===============
def explicacion_humana(r: pd.Series,
                       freq_thr:int=3, sev_thr:float=1.0,
                       same_comp_thr:int=2, recent_days_thr:int=14) -> str:
    def sev_txt(v):
        if v >= 1.6: return "alta"
        if v >= 1.0: return "media"
        if v > 0:    return "baja"
        return "sin severidad"

    razones = []
    if int(r.get("return_after_replace_60d",0))==1 and int(r.get("critical_flag",0))==1:
        razones.append("volvió a fallar en menos de 60 días después de un cambio/reemplazo en una pieza crítica")
    if int(r.get("same_comp_120d",0)) >= same_comp_thr and int(r.get("critical_flag",0))==1:
        razones.append(f"el mismo componente crítico se repitió {int(r['same_comp_120d'])} veces en 120 días")
    if int(r.get("freq_90d",0)) >= freq_thr and float(r.get("sev_avg_90d",0)) >= sev_thr:
        razones.append(f"tuvo {int(r['freq_90d'])} visitas en 90 días con severidad {sev_txt(float(r['sev_avg_90d']))}")
    if int(r.get("days_since_prev",999)) <= recent_days_thr and float(r.get("sev_avg_90d",0)) > 1.0:
        razones.append("presentó una falla severa muy reciente")

    if not razones:
        if r["estado"]=="AMBAR":
            razones.append("se observan señales elevadas pero sin patrón crítico repetido")
        else:
            razones.append("el comportamiento es estable, sin riesgos recientes")

    if r["estado"]=="ROJO":
        recomendacion = "Prioriza reemplazo o intervención mayor. Si reparas, considera kit completo de la parte crítica y compara vs. equipo nuevo."
    elif r["estado"]=="AMBAR":
        recomendacion = f"Observa de cerca. Haz mantenimiento preventivo y revisa la parte señalada. Si repite la pieza crítica o hay ≥{freq_thr} visitas en 90 días, pasa a ROJO."
    else:
        recomendacion = "Sigue con mantenimiento normal."

    cliente = (r.get("Cliente","") or "—").strip()
    articulo = (r.get("Artículo","") or "—").strip()
    return (
        f"{r.device_id}: {r.estado}\n"
        f"Por qué: " + "; ".join(razones) + ".\n"
        f"Siguiente paso: {recomendacion}\n"
        f"Cliente: {cliente} • Equipo: {articulo}"
    )


# =============== Pipeline principal ===============
def process_filelike(file_bytes: bytes, filename: str, sheet_name: str,
                     to_remove: List[str],
                     colmap: Dict[str,str],
                     params: Dict[str, float],
                     use_negations: bool = True,
                     zs: Optional[ZeroShotWrapper] = None,
                     zs_thr: float = 0.6) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # leer
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        # seleccionar hoja
        names = xls.sheet_names
        norm_map = {normalize(n): n for n in names}
        sheet_used = norm_map.get(normalize(sheet_name), names[1] if len(names)>1 else names[0])
        df = pd.read_excel(xls, sheet_name=sheet_used)

    df = drop_columns_robust(df, to_remove)

    c_cliente  = colmap["cliente"]
    c_asunto   = colmap["asunto"]
    c_fecha    = colmap["fecha"]
    c_serie    = colmap["serie"]
    c_articulo = colmap["articulo"]
    c_tipo     = colmap["tipo"]
    c_resol    = colmap["resol"]

    data = df.copy()
    data[c_fecha] = pd.to_datetime(data[c_fecha], errors="coerce")
    data["device_id"] = data[c_serie].astype(str).where(~data[c_serie].isna(),
                         data[c_cliente].astype(str) + " | " + data[c_articulo].astype(str))
    data = data.dropna(subset=[c_fecha, "device_id"]).copy()

    # NLP por fila
    nlp_feats = data.apply(lambda r: classify_row(r, c_asunto, c_resol, c_tipo,
                                                  use_negations=use_negations,
                                                  zs=zs, zs_thr=zs_thr), axis=1, result_type="expand")
    tickets = pd.concat([data.reset_index(drop=True), nlp_feats], axis=1)

    # Ventanas 90/180
    tickets = tickets.sort_values(["device_id", c_fecha])
    rows = []
    for dev, g in tickets.groupby("device_id"):
        g = g.reset_index(drop=True)
        for i, r in g.iterrows():
            end = r[c_fecha]
            win90  = g[(g[c_fecha] > end - timedelta(days=90))  & (g[c_fecha] <= end)]
            win180 = g[(g[c_fecha] > end - timedelta(days=180)) & (g[c_fecha] <= end)]
            prev_date = g.loc[i-1, c_fecha] if i>0 else pd.NaT
            rows.append({
                "device_id": dev,
                "ref_date": end,
                "freq_90d": len(win90),
                "sev_avg_90d": float(win90["sev"].mean()) if len(win90) else 0.0,
                "crit_180d": int(win180["critical"].sum()),
                "rep_fusor_180d": int(win180["fusor"].sum()) if "fusor" in win180.columns else 0,
                "rep_placa_180d": int(win180["electronica"].sum()) if "electronica" in win180.columns else 0,
                "days_since_prev": int((end - prev_date).days) if pd.notna(prev_date) else 999,
            })
    timeline = pd.DataFrame(rows)

    # Historial por componente
    hist = build_history_features(tickets, c_fecha, half_life_days=int(params["half_life_days"]))

    # Peor componente del día
    if hist.empty:
        timeline["same_comp_120d"] = 0
        timeline["same_comp_decay"] = 0.0
        timeline["return_after_replace_60d"] = 0
        timeline["critical_flag"] = 0
    else:
        def worst_comp_stats(dev, ref_date):
            h = hist[(hist["device_id"]==dev) & (hist["date"]==ref_date)]
            if h.empty:
                return pd.Series({"same_comp_120d":0, "same_comp_decay":0.0, "return_after_replace_60d":0, "critical_flag":0})
            h = h.sort_values(["critical","same_comp_120d","same_comp_decay"], ascending=[False, False, False])
            top = h.iloc[0]
            return pd.Series({
                "same_comp_120d": int(top["same_comp_120d"]),
                "same_comp_decay": float(top["same_comp_decay"]),
                "return_after_replace_60d": int(top["return_after_replace_60d"]),
                "critical_flag": int(top["critical"]),
            })
        timeline[["same_comp_120d","same_comp_decay","return_after_replace_60d","critical_flag"]] = \
            timeline.apply(lambda r: worst_comp_stats(r["device_id"], r["ref_date"]), axis=1)

    # Decisión
    def dec(r):
        return decision_history_first(
            r,
            freq_thr=int(params["freq_thr"]),
            sev_thr=float(params["sev_thr"]),
            same_comp_thr=int(params["same_comp_thr"]),
            decay_thr_crit=float(params["decay_thr_crit"]),
            recent_days_thr=int(params["recent_days_thr"]),
        )
    timeline[["estado","motivo"]] = timeline.apply(lambda r: pd.Series(dec(r)), axis=1)

    # Último estado por equipo + metadatos
    if len(timeline):
        idx = timeline.groupby("device_id")["ref_date"].idxmax()
        semaforo = timeline.loc[idx].sort_values(["estado","device_id"]).reset_index(drop=True)
    else:
        semaforo = pd.DataFrame(columns=["device_id","estado","motivo"])

    last_meta = (data.sort_values(c_fecha)
                 .groupby("device_id")
                 .agg({colmap["cliente"]:"last", colmap["articulo"]:"last"})
                 .rename(columns={colmap["cliente"]:"Cliente", colmap["articulo"]:"Artículo"})
                 .reset_index())
    semaforo = semaforo.merge(last_meta, on="device_id", how="left")

    # === ML opcional: cargar modelo XGBoost si existe y puntuar ===
    try:
        import os
        import joblib
        from pathlib import Path

        mdl_path = Path(__file__).resolve().parent.parent / "models" / "xgb_change.pkl"
        if mdl_path.exists():
            bundle = joblib.load(str(mdl_path))
            FEATS = bundle.get("features", [
                "freq_90d","sev_avg_90d","days_since_prev",
                "same_comp_120d","same_comp_decay","critical_flag",
                "rep_fusor_180d","rep_placa_180d",
            ])
            model = bundle["model"]

            for c in FEATS:
                if c not in timeline.columns:
                    timeline[c] = 0.0

            X = timeline[FEATS].fillna(0.0).values
            p = model.predict_proba(X)[:, 1]
            timeline["p_change"] = p

            thr = float(bundle.get("best_thr", 0.65))
            timeline["estado_ml"] = np.where(
                p >= thr, "ROJO",
                np.where(p >= max(0.45, thr - 0.2), "AMBAR", "VERDE")
            )
        else:
            print(f"[ML] modelo no encontrado: {mdl_path}")
    except Exception as e:
        print(f"[ML] no se pudo cargar/puntuar modelo: {e}")

    return tickets, timeline, semaforo
