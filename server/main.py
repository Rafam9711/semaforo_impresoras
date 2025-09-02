# -*- coding: utf-8 -*-
import os
import uuid
import json
from contextlib import asynccontextmanager
from typing import Dict, Optional
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel

load_dotenv()

import unicodedata

def _no_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


# ML opcional
try:
    import joblib
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
except Exception as e:
    joblib = None
    XGBClassifier = None
    roc_auc_score = average_precision_score = f1_score = None
    print(f"[ML] Dependencias no cargadas: {e}")

# Zero-shot opcional (no obligatorio)
def _detect_device() -> int:
    try:
        if "HF_DEVICE" in os.environ:
            return int(os.environ["HF_DEVICE"])
        import torch  # noqa
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1

ZS_WRAPPER = None
def get_zero_shot(use_zeroshot: bool):
    global ZS_WRAPPER
    if not use_zeroshot:
        return None
    if ZS_WRAPPER is not None:
        return ZS_WRAPPER
    try:
        from transformers import pipeline
        from core.engine import ZeroShotWrapper
        model_id = os.getenv("HF_ZS_MODEL", "joeddav/xlm-roberta-large-xnli")
        device = _detect_device()
        clf = pipeline("zero-shot-classification", model=model_id, device=device)
        ZS_WRAPPER = ZeroShotWrapper(clf)
        print(f"[zero-shot] loaded model={model_id} on device={clf.model.device}")
        return ZS_WRAPPER
    except Exception as e:
        print(f"[zero-shot] no disponible: {e}")
        return None

# Core del motor
from core.engine import (
    process_filelike,
    explicacion_humana,
    build_history_features,
    decision_history_first,
)

# ---------- Lifespan (sin @on_event warnings) ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("PRELOAD_ZS", "0") == "1":
        _ = get_zero_shot(True)
    yield

app = FastAPI(title="SAME Impresoras API", lifespan=lifespan)

# Sesiones en RAM
SESSIONS: Dict[str, Dict] = {}

# ----------------- Modelos -----------------
class ProcessResponse(BaseModel):
    session_id: str
    counts: Dict[str, int]

class ChatIn(BaseModel):
    session_id: str
    message: str
    use_gpt: bool = False
    mode: Optional[str] = "rules"   # "rules" | "ml"

# ----------------- Rutas -----------------
@app.get("/", response_class=PlainTextResponse)
def home(): return "SAME Impresoras API OK. Visita /docs"

@app.get("/health", response_class=PlainTextResponse)
def health(): return "ok"

@app.post("/process", response_model=ProcessResponse)
async def process(
    file: UploadFile = File(...),
    sheet_name: str = Form("LLAMADAS DE SERVICIOS"),
    to_remove: str = Form("N\nTratado por\nModelo\nFecha asignada\nTipo de llamada\nCart en el folio\nEstado\nServicio Gratuito\nSubtipo de problema"),
    use_negations: bool = Form(True),
    use_zeroshot: bool = Form(False),
    zs_thr: float = Form(0.6),

    col_cliente: str = Form("Cliente"),
    col_asunto: str = Form("Asunto"),
    col_fecha: str = Form("Fecha de creación"),
    col_serie: str = Form("No. Serie"),
    col_articulo: str = Form("Descripción del artículo"),
    col_tipo: str = Form("Tipo de problema"),
    col_resol: str = Form("Resolución"),

    freq_thr: int = Form(3),
    sev_thr: float = Form(1.0),
    same_comp_thr: int = Form(2),
    decay_thr_crit: float = Form(2.5),
    recent_days_thr: int = Form(14),
    half_life_days: int = Form(45),
):
    byts = await file.read()
    colmap = dict(cliente=col_cliente, asunto=col_asunto, fecha=col_fecha,
                  serie=col_serie, articulo=col_articulo, tipo=col_tipo, resol=col_resol)
    params = dict(freq_thr=freq_thr, sev_thr=sev_thr, same_comp_thr=same_comp_thr,
                  decay_thr_crit=decay_thr_crit, recent_days_thr=recent_days_thr,
                  half_life_days=half_life_days)
    zs = get_zero_shot(use_zeroshot)
    tickets, timeline, semaforo = process_filelike(
        file_bytes=byts, filename=file.filename, sheet_name=sheet_name,
        to_remove=[x.strip() for x in to_remove.splitlines() if x.strip()],
        colmap=colmap, params=params, use_negations=use_negations, zs=zs, zs_thr=zs_thr
    )

    sid = uuid.uuid4().hex
    SESSIONS[sid] = dict(tickets=tickets, timeline=timeline, semaforo=semaforo,
                         params=params, colmap=colmap)
    counts = semaforo["estado"].value_counts().to_dict()
    for k in ["VERDE","AMBAR","ROJO"]: counts.setdefault(k, 0)
    return ProcessResponse(session_id=sid, counts=counts)

@app.get("/summary")
def summary(session_id: str):
    s = SESSIONS.get(session_id) or {}
    sem = s.get("semaforo", pd.DataFrame())
    return dict(total=int(sem["device_id"].nunique()) if not sem.empty else 0,
                estados=sem["estado"].value_counts().to_dict() if not sem.empty else {})

@app.get("/why")
def why(session_id: str, device_id: str):
    s = SESSIONS.get(session_id)
    if not s: raise HTTPException(404, "session not found")
    sem = s["semaforo"]
    row = sem[sem["device_id"].str.contains(device_id, case=False, na=False)]
    if row.empty: raise HTTPException(404, "device not found")
    r = row.iloc[0]
    txt = explicacion_humana(
        r,
        freq_thr=s["params"]["freq_thr"],
        sev_thr=s["params"]["sev_thr"],
        same_comp_thr=s["params"]["same_comp_thr"],
        recent_days_thr=s["params"]["recent_days_thr"],
    )
    return {"device_id": r.device_id, "estado": r.estado, "explicacion": txt}

@app.get("/export")
def export(session_id: str, which: str):
    s = SESSIONS.get(session_id)
    if not s: raise HTTPException(404, "session not found")
    if   which == "semaforo": df = s["semaforo"]
    elif which == "timeline": df = s["timeline"]
    elif which == "tickets":  df = s["tickets"]
    else: raise HTTPException(400, "which ∈ {semaforo|timeline|tickets}")
    csv = df.to_csv(index=False).encode("utf-8-sig")
    return StreamingResponse(iter([csv]), media_type="text/csv")

# --------- ML: entrenar/predict ----------
@app.post("/train_ml")
def train_ml(session_id: str = Query(...), valid_split: float = Query(0.2)):
    if joblib is None or XGBClassifier is None:
        raise HTTPException(500, "Instala dependencias ML: pip install joblib scikit-learn xgboost")
    s = SESSIONS.get(session_id)
    if not s: raise HTTPException(404, "session not found")

    df = s["timeline"].copy()
    if df.empty: raise HTTPException(400, "timeline vacío")

    df["ref_date"] = pd.to_datetime(df["ref_date"])
    df = df.sort_values(["device_id", "ref_date"]).reset_index(drop=True)

    nxt = df.groupby("device_id")["ref_date"].shift(-1)
    sev_next = (df.groupby("device_id")["sev_avg_90d"].shift(-1) > 1.0) & ((nxt - df["ref_date"]).dt.days <= 60)
    crit_next = ((df.groupby("device_id")["same_comp_120d"].shift(-1) >= 1) &
                 (df.groupby("device_id")["critical_flag"].shift(-1) == 1) &
                 ((nxt - df["ref_date"]).dt.days <= 120))
    df["y"] = (sev_next.fillna(False) | crit_next.fillna(False)).astype(int)

    FEATS = ["freq_90d","sev_avg_90d","days_since_prev","same_comp_120d",
             "same_comp_decay","critical_flag","rep_fusor_180d","rep_placa_180d"]
    for c in FEATS:
        if c not in df.columns: df[c] = 0.0
    X = df[FEATS].fillna(0.0).values
    y = df["y"].values

    cut = df["ref_date"].quantile(1 - valid_split)
    tr = df["ref_date"] <= cut
    va = df["ref_date"] > cut
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    if Xtr.shape[0] < 100 or Xva.shape[0] < 50:
        raise HTTPException(400, "Datos insuficientes para entrenar/validar")

    model = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        eval_metric="logloss", n_jobs=4
    )
    model.fit(Xtr, ytr)
    p_va = model.predict_proba(Xva)[:,1]
    auc = roc_auc_score(yva, p_va)
    ap  = average_precision_score(yva, p_va)
    thr_grid = np.linspace(0.2, 0.8, 13)
    f1s = [f1_score(yva, (p_va >= t).astype(int)) for t in thr_grid]
    best_t = float(thr_grid[int(np.argmax(f1s))])

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "features": FEATS, "best_thr": best_t}, "models/xgb_change.pkl")
    return {"ok": True, "valid_auc": float(auc), "valid_ap": float(ap), "best_thr_f1": best_t}

@app.post("/predict_ml")
def predict_ml(session_id: str = Query(...), threshold: float = Query(0.65)):
    if joblib is None:
        raise HTTPException(500, "Instala joblib/xgboost/sklearn para usar ML")
    s = SESSIONS.get(session_id)
    if not s: raise HTTPException(404, "session not found")

    mdl_path = "models/xgb_change.pkl"
    if not os.path.exists(mdl_path):
        raise HTTPException(400, "Modelo no encontrado; entrena con /train_ml")

    bundle = joblib.load(mdl_path)
    FEATS = bundle["features"]; model = bundle["model"]
    if "best_thr" in bundle and threshold < 0: threshold = float(bundle["best_thr"])

    tl = s["timeline"].copy()
    for c in FEATS:
        if c not in tl.columns: tl[c] = 0.0
    p = model.predict_proba(tl[FEATS].fillna(0.0).values)[:,1]
    tl["p_change"] = p
    tl["estado_ml"] = np.where(p >= threshold, "ROJO",
                        np.where(p >= max(0.45, threshold-0.2), "AMBAR", "VERDE"))
    idx = tl.groupby("device_id")["ref_date"].idxmax()
    sem_ml = tl.loc[idx].sort_values(["estado_ml","device_id"]).reset_index(drop=True)

    sem_prev = s["semaforo"]
    keep = [c for c in ["Cliente","Artículo"] if c in sem_prev.columns]
    if keep:
        meta = sem_prev[["device_id"] + keep].drop_duplicates("device_id")
        sem_ml = sem_ml.merge(meta, on="device_id", how="left")

    s["timeline"] = tl
    s["semaforo_ml"] = sem_ml
    s["ml_threshold"] = threshold

    counts = sem_ml["estado_ml"].value_counts().to_dict()
    for k in ["VERDE","AMBAR","ROJO"]: counts.setdefault(k, 0)
    return {"ok": True, "threshold": threshold, "counts": counts}

@app.post("/chat")
def chat(inp: ChatIn):
    import unicodedata, re, json
    from datetime import datetime

    s = SESSIONS.get(inp.session_id)
    if not s:
        raise HTTPException(404, "session not found")

    # ---------------- helpers ----------------
    def _no_accents(t: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFD", t or "") if unicodedata.category(c) != "Mn")

    def _norm_state(x: str) -> str:
        xx = _no_accents(str(x or "")).lower()
        if "rojo" in xx or xx == "r": return "ROJO"
        if "ambar" in xx or "amarill" in xx or "naranja" in xx or xx == "a": return "AMBAR"
        if "verde" in xx or xx == "v": return "VERDE"
        return ""

    def _top_list(df: pd.DataFrame, estado: str, n: int = 10) -> str:
        sort_cols = [c for c in ["sev_avg_90d","same_comp_120d","ref_date"] if c in df.columns]
        if not sort_cols: sort_cols = ["device_id"]
        sub = df[df["estado"] == estado].sort_values(sort_cols, ascending=[False]*len(sort_cols)).head(n)
        if sub.empty: return "(sin equipos)"
        lines = []
        for _, r in sub.iterrows():
            lines.append(f"- {r.device_id} · {r.get('Cliente','—')} · {r.get('Artículo','—')} · {r.get('motivo','—')}")
        return "\n".join(lines)

    # ---------------- fuente (reglas vs ML) ----------------
    mode = (inp.mode or "rules").lower()
    if mode.startswith("ml"):
        sem = s.get("semaforo_ml")
        if sem is None or sem.empty:
            return {"answer": "Aún no has aplicado el modelo ML. Usa 'Aplicar ML' primero."}
        sem = sem.copy()
        if "estado_ml" in sem.columns:
            sem["estado"] = sem["estado_ml"]
        if "motivo" not in sem.columns:
            if "p_change" in sem.columns:
                sem["motivo"] = sem["p_change"].apply(lambda x: f"Modelo ML · prob. reemplazo = {x:.2f}")
            else:
                sem["motivo"] = "Modelo ML"
    else:
        sem = s["semaforo"]

    q_raw = inp.message.strip()
    q = q_raw.lower()
    q0 = _no_accents(q)

    # ---------------- caso específico: POR QUÉ <serie> ----------------
    if q0.startswith("por que") or "porque " in q0 or q0.startswith("por qué"):
        key = q_raw.split()[-1]
        row = sem[sem["device_id"].str.contains(key, case=False, na=False)]
        if row.empty: return {"answer": f"No encuentro el equipo que contenga: {key}"}
        r = row.iloc[0]

        # Explicación base
        if mode.startswith("ml"):
            base = (f"{r.device_id}: {r.estado}\n"
                    f"Por qué: {r.motivo}.\n"
                    f"Cliente: {r.get('Cliente','—')} • Equipo: {r.get('Artículo','—')}")
        else:
            base = explicacion_humana(
                r,
                freq_thr=s["params"]["freq_thr"],
                sev_thr=s["params"]["sev_thr"],
                same_comp_thr=s["params"]["same_comp_thr"],
                recent_days_thr=s["params"]["recent_days_thr"],
            )

        # Enriquecer con últimos comentarios y piezas (si existen)
        T = s.get("tickets", pd.DataFrame()).copy()
        if not T.empty:
            def _pick(df, *names):
                for n in names:
                    if n in df.columns: return n
                return None
            col_dev   = _pick(T, "device_id","No. Serie","No. Serie (impresora)")
            col_fecha = _pick(T, "ref_date","Fecha de creación","Fecha")
            col_tipo  = _pick(T, "Tipo de problema","tipo","Problema")
            col_resol = _pick(T, "Resolución","resol","Notas")
            col_art   = _pick(T, "Descripción del artículo","Artículo","articulo")
            if col_dev:
                tt = T[T[col_dev].astype(str).str.contains(r.device_id, case=False, na=False)].copy()
                if not tt.empty:
                    if col_fecha: tt[col_fecha] = pd.to_datetime(tt[col_fecha], errors="coerce"); tt = tt.sort_values(col_fecha, ascending=False)
                    # últimos 3 comentarios
                    if col_tipo or col_resol:
                        cs = []
                        for _, rr in tt.head(3).iterrows():
                            a = rr.get(col_tipo, ""); b = rr.get(col_resol, "")
                            if a or b: cs.append(f"{str(a).strip()} — {str(b).strip()}")
                        r["recent_tech_comments"] = "; ".join([c for c in cs if c]).strip() or "—"
                    # piezas reemplazadas (heurística)
                    rep_mask = tt[col_resol].astype(str).str.contains("reemplaz", case=False, na=False) if col_resol else pd.Series(False, index=tt.index)
                    piezas = []
                    if col_art and col_art in tt.columns:
                        piezas = tt.loc[rep_mask, col_art].dropna().astype(str).str.strip().unique().tolist()
                    r["past_replacements"] = "; ".join(piezas[:3]) if piezas else "—"

        # GPT estructurado (si está activo)
        if inp.use_gpt and os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                client = OpenAI()
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.3,
                    max_tokens=500,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Eres un ingeniero de soporte de impresoras. "
                                "Devuelve SOLO un JSON válido con: resumen, pieza_sospechosa, justificacion, explicacion_cliente. "
                                "No inventes: usa únicamente los datos provistos."
                            )
                        },
                        {
                            "role": "user",
                            "content": f"""
Serie: {r.device_id}
Estado: {r.estado}
Explicación automática: {base}
Comentarios técnicos recientes: {r.get('recent_tech_comments','—')}
Historial de piezas reemplazadas: {r.get('past_replacements','—')}
"""
                        }
                    ]
                )
                data = json.loads(resp.choices[0].message.content)
                answer = (
                    f"### 🧠 Diagnóstico para **{r.device_id}** · **{r.estado}**\n\n"
                    f"**🔧 Pieza sospechosa:** {data.get('pieza_sospechosa','—')}\n\n"
                    f"**📋 Resumen**\n{data.get('resumen','')}\n\n"
                    f"**📑 Justificación**\n{data.get('justificacion','')}\n\n"
                    f"**🙂 Explicación para cliente**\n{data.get('explicacion_cliente','')}\n\n"
                    f"> Cliente: **{r.get('Cliente','—')}** · Equipo: **{r.get('Artículo','—')}**"
                )
                return {"answer": answer}
            except Exception as e:
                base = f"{base}\n\n(Nota: no se pudo enriquecer con GPT: {e})"

        # Sin GPT o si falla: devuelve base formateada
        answer = (
            f"### 🧠 Diagnóstico para **{r.device_id}** · **{r.estado}**\n\n"
            f"{base}\n\n"
            f"> Cliente: **{r.get('Cliente','—')}** · Equipo: **{r.get('Artículo','—')}**"
        )
        return {"answer": answer}

    # ---------------- CHAT NATURAL CON GPT SOBRE CONTEXTO ----------------
    # Si no es un "por qué", construimos un CONTEXTO desde el semáforo/tickets
    # para que GPT responda libremente (sin router de intents).
    if inp.use_gpt and os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()

            total = int(sem["device_id"].nunique()) if not sem.empty else 0
            counts = sem["estado"].value_counts().to_dict() if not sem.empty else {}

            # ¿menciona algún estado?
            states_asked = []
            for sname in ["ROJO","AMBAR","VERDE"]:
                if sname.lower() in q or _no_accents(sname.lower()) in q0:
                    states_asked.append(sname)

            # ¿menciona posible cliente?
            tokens = [t for t in re.findall(r"[A-Za-z0-9]+", q_raw) if len(t) >= 3]
            client_hits = []
            if "Cliente" in sem.columns and not sem.empty:
                for t in tokens:
                    m = sem["Cliente"].astype(str).str.contains(t, case=False, na=False)
                    if m.any():
                        client_hits.append(t)
            client_hits = list(dict.fromkeys(client_hits))[:2]  # máximo 2 pistas de cliente

            # ¿menciona posible serie?
            device_hits = []
            if not sem.empty:
                for t in tokens:
                    m = sem["device_id"].astype(str).str.contains(t, case=False, na=False)
                    if m.any():
                        device_hits.append(t)
            device_hits = list(dict.fromkeys(device_hits))[:2]

            # armar TOPs (globales o por estado pedido)
            tops = []
            want_states = states_asked if states_asked else ["ROJO","AMBAR","VERDE"]
            for st_ in want_states:
                tops.append(f"TOP {st_}:\n{_top_list(sem, st_, 10)}")

            # si hay clientes/series mencionados, crea vistas filtradas para el contexto
            extra_views = []
            if client_hits:
                for ckw in client_hits:
                    sub = sem[sem["Cliente"].astype(str).str.contains(ckw, case=False, na=False)]
                    if not sub.empty:
                        extra_views.append(f"CLIENTE {ckw} TOP:\n{_top_list(sub, 'ROJO', 6)}\n{_top_list(sub, 'AMBAR', 6)}\n{_top_list(sub, 'VERDE', 6)}")
            if device_hits:
                for dkw in device_hits:
                    sub = sem[sem["device_id"].astype(str).str.contains(dkw, case=False, na=False)]
                    if not sub.empty:
                        extra_views.append(f"SERIE {dkw} COINCIDENCIAS:\n" + "\n".join([f"- {r.device_id} · {r.get('Cliente','—')} · {r.get('Artículo','—')} · {r.get('motivo','—')}" for _, r in sub.head(8).iterrows()]))

            context = (
                f"FECHA: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"TOTAL_EQUIPOS: {total}\n"
                f"CONTEOS: {counts}\n\n"
                + "\n\n".join(tops + extra_views)
            )

            prompt_system = (
                "Eres un asistente técnico de impresoras para operaciones. "
                "Responde en español, claro y conciso, usando EXCLUSIVAMENTE el CONTEXTO que te paso. "
                "Si faltan datos para responder exactamente, di qué falta o responde la mejor aproximación. "
                "Si el usuario pide 'solución' o recomendaciones, sugiere acciones basadas en estado y motivo, "
                "priorizando seguridad y tiempos de operación. Usa Markdown con títulos y bullets."
            )

            prompt_user = f"CONSULTA: {q_raw}\n\nCONTEXTO:\n{context}"

            comp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=700,
                messages=[
                    {"role":"system","content": prompt_system},
                    {"role":"user","content": prompt_user},
                ]
            )
            answer = comp.choices[0].message.content.strip()
            return {"answer": answer}

        except Exception as e:
            # caemos a reglas básicas si falla GPT
            pass

    # ---------------- fallback sin GPT (reglas suaves) ----------------
    # estados sueltos tipo "ambar/verde/rojo(s)"
    estado = _norm_state(q0)
    if estado:
        return {"answer": f"Top {estado}:\n{_top_list(sem, estado, 10)}"}

    if q0 in ("ayuda","help","?","menu","opciones"):
        return {"answer": "Puedes escribir en lenguaje natural: 'ámbar', 'verdes del IMSS', 'rojos top 10', 'por qué 3XB...', 'solución para Xerox', etc."}

    # sin match
    return {"answer": "No entendí del todo. Intenta algo como: 'ámbar', 'verdes del IMSS', 'rojos top 10', 'por qué <serie>' o describe qué necesitas y activando el switch de GPT te respondo con el contexto."}
