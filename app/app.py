# streamlit run app/app.py
# UI en Streamlit (habla con el backend FastAPI). Chat ahora está al final del tab "Explorador de tablas".

import os
import io
import requests
import pandas as pd
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="SAME · Salud de Impresoras", layout="wide")
st.title("SAME · Salud de Impresoras – Chat")

# ================= Sidebar =================
st.sidebar.header("Archivo")
file = st.sidebar.file_uploader("Sube tu .xlsx o .csv", type=["xlsx","xls","csv"])
sheet_name = st.sidebar.text_input("Hoja (si es Excel)", "LLAMADAS DE SERVICIOS")
to_remove_txt = st.sidebar.text_area(
    "Columnas a eliminar (una por línea)",
    "N\nTratado por\nModelo\nFecha asignada\nTipo de llamada\nCart. en el folio\nEstado\nServicio Gratuito\nSubtipo de problema",
    height=140,
)

st.sidebar.header("Umbrales")
freq_thr = st.sidebar.slider("Frecuencia 90d (ROJO si ≥)", 1, 6, 3)
sev_thr = st.sidebar.slider("Severidad 90d (ROJO si ≥)", 0.0, 2.0, 1.0, 0.1)
same_comp_thr = st.sidebar.slider("Mismo componente crítico 120d (ROJO si ≥)", 1, 5, 2)
decay_thr_crit = st.sidebar.slider("Historial crítico reciente (decay)", 0.0, 6.0, 2.5, 0.1)
recent_days_thr = st.sidebar.slider("Falla severa muy reciente (días ≤)", 7, 30, 14)
half_life_days = st.sidebar.slider("Half-life historial (días)", 15, 120, 45)

st.sidebar.header("NLP avanzado")
use_negations = st.sidebar.checkbox("Negaciones ('no es el fusor')", value=False)
use_zeroshot = st.sidebar.checkbox("Zero-Shot (HF) si diccionario no detecta", value=False)
zs_thr = st.sidebar.slider("Umbral Zero-Shot", 0.30, 0.90, 0.60, 0.01)

# =============== Estado de sesión ===============
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat" not in st.session_state:
    st.session_state.chat = []  # [{"role":"user"|"assistant","content":str}]

# =============== Helpers =================
def backend_export(which: str) -> pd.DataFrame:
    r = requests.get(
        f"{BACKEND}/export",
        params={"session_id": st.session_state.session_id, "which": which},
        timeout=120,
    )
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

def kpi_semaforo() -> pd.DataFrame:
    try:
        sem = backend_export("semaforo")
        k1, k2, k3 = st.columns(3)
        k1.metric("Equipos", sem["device_id"].nunique())
        k2.metric("ROJO (Reglas)", int((sem["estado"] == "ROJO").sum()))
        k3.metric("ÁMBAR (Reglas)", int((sem["estado"] == "AMBAR").sum()))
        return sem
    except Exception as e:
        st.warning(f"No se pudo cargar semáforo: {e}")
        return pd.DataFrame()

def filtro_dataframe(df: pd.DataFrame, key: str, estado_col: str = "estado") -> pd.DataFrame:
    """Caja de búsqueda + multiselección de estado con keys únicos (evita DuplicateWidgetID)."""
    with st.expander("🔎 Filtros", expanded=False):
        q = st.text_input("Buscar por cliente/serie/artículo", "", key=f"{key}_q")
        estados = []
        if estado_col in df.columns:
            opciones = [x for x in ["ROJO", "AMBAR", "VERDE"] if x in df[estado_col].unique().tolist()]
            estados = st.multiselect("Estado", opciones, default=[], key=f"{key}_est")

    if df.empty:
        return df

    ff = df.copy()
    if q:
        mask = (
            ff["device_id"].astype(str).str.contains(q, case=False, na=False)
            | ff.get("Cliente", pd.Series("", index=ff.index)).fillna("").str.contains(q, case=False, na=False)
            | ff.get("Artículo", pd.Series("", index=ff.index)).fillna("").str.contains(q, case=False, na=False)
        )
        ff = ff[mask]
    if estados:
        ff = ff[ff[estado_col].isin(estados)]
    return ff

# =============== Acciones (botonera superior) ===============
c1, c2, c3 = st.columns([1,1,1])
with c1:
    if st.button("Procesar archivo en backend", type="primary", use_container_width=True):
        if not file:
            st.error("Sube un archivo primero.")
        else:
            files = {"file": (file.name, file.getvalue())}
            data = dict(
                sheet_name=sheet_name, to_remove=to_remove_txt,
                use_negations=use_negations, use_zeroshot=use_zeroshot, zs_thr=zs_thr,
                col_cliente="Cliente", col_asunto="Asunto", col_fecha="Fecha de creación",
                col_serie="No. Serie", col_articulo="Descripción del artículo",
                col_tipo="Tipo de problema", col_resol="Resolución",
                freq_thr=int(freq_thr), sev_thr=float(sev_thr),
                same_comp_thr=int(same_comp_thr), decay_thr_crit=float(decay_thr_crit),
                recent_days_thr=int(recent_days_thr), half_life_days=int(half_life_days),
            )
            r = requests.post(f"{BACKEND}/process", data=data, files=files, timeout=300)
            r.raise_for_status()
            res = r.json()
            st.session_state.session_id = res["session_id"]
            st.session_state.chat = []  # reset chat con nuevo dataset
            st.success(f"Listo. session_id: {res['session_id']} · Conteos: {res['counts']}")

with c2:
    if st.button("Recalcular (sin re-subir)", use_container_width=True):
        if not st.session_state.session_id:
            st.error("Primero procesa un archivo.")
        else:
            rr = requests.post(
                f"{BACKEND}/recompute",
                params=dict(
                    session_id=st.session_state.session_id,
                    freq_thr=int(freq_thr), sev_thr=float(sev_thr),
                    same_comp_thr=int(same_comp_thr), decay_thr_crit=float(decay_thr_crit),
                    recent_days_thr=int(recent_days_thr), half_life_days=int(half_life_days),
                ),
                timeout=180,
            )
            st.info(rr.json())

with c3:
    col31, col32 = st.columns(2)
    with col31:
        if st.button("⚙️ Entrenar modelo", use_container_width=True):
            if not st.session_state.session_id:
                st.error("Primero procesa un archivo.")
            else:
                rr = requests.post(
                    f"{BACKEND}/train_ml",
                    params={"session_id": st.session_state.session_id},
                    timeout=600,
                )
                st.info(rr.json())
    with col32:
        thr_ml = st.number_input("Umbral ML", min_value=0.0, max_value=1.0, value=0.65, step=0.01)
        if st.button("🤖 Aplicar ML", use_container_width=True):
            if not st.session_state.session_id:
                st.error("Primero procesa un archivo.")
            else:
                rr = requests.post(
                    f"{BACKEND}/predict_ml",
                    params={"session_id": st.session_state.session_id, "threshold": float(thr_ml)},
                    timeout=180,
                )
                st.success(rr.json())

st.markdown("---")

# =============== KPIs + Tabs ===============
if st.session_state.session_id:
    semaforo = kpi_semaforo()
    # Eliminamos el tab de Chat; queda dentro del tab 0
    tabs = st.tabs(["🔎 Explorador de tablas", "🚦 Semáforo (Reglas)", "🤖 Semáforo (ML)"])

    # --------- 1) Explorador ----------
    with tabs[0]:
        t1, t2 = st.columns(2)
        with t1:
            st.subheader("Tickets")
            df_t = backend_export("tickets")
            df_t_f = filtro_dataframe(
                df_t,
                key="tickets",
                estado_col="estado" if "estado" in df_t.columns else "estado",
            )
            st.dataframe(df_t_f, use_container_width=True, height=360)
        with t2:
            st.subheader("Timeline")
            df_l = backend_export("timeline")
            df_l_f = filtro_dataframe(
                df_l,
                key="timeline",
                estado_col="estado" if "estado" in df_l.columns else "estado",
            )
            st.dataframe(df_l_f, use_container_width=True, height=360)

        st.markdown("---")
        # ========== Chat al final del tab "Explorador de tablas" ==========
        st.subheader("Chat")

        # Cabecera: fuente + “sugerencias”
        cA, cB = st.columns([3,1])
        with cA:
            st.caption("Pregunta (e.g., 'rojos', 'ver <serie>', 'por qué <serie>', 'cliente <nombre>')")
        with cB:
            chat_mode = st.radio("Fuente", ["Reglas", "ML"], index=0, key="chat_mode")

        # Sugerencias rápidas
        colq1, colq2, colq3, colq4, colq5 = st.columns(5)
        if colq1.button("rojos", key="q_rojos"): st.session_state.chat.append({"role":"user","content":"rojos"})
        if colq2.button("ámbar", key="q_ambar"): st.session_state.chat.append({"role":"user","content":"ámbar"})
        if colq3.button("ayuda", key="q_ayuda"): st.session_state.chat.append({"role":"user","content":"ayuda"})
        if colq4.button("cliente IMSS", key="q_imss"): st.session_state.chat.append({"role":"user","content":"cliente IMSS"})
        if colq5.button("ver <serie>", key="q_ver"): st.session_state.chat.append({"role":"user","content":"ver PHBTB..."})

        # Render historial
        for m in st.session_state.chat:
            with st.chat_message("user" if m["role"]=="user" else "assistant"):
                st.markdown(m["content"])

        use_gpt = st.checkbox("Pulir respuestas con GPT mini (opcional)", value=False, key="chat_gpt")

        # Entrada
        prompt = st.chat_input("Escribe aquí…", key="chat_input")
        # Si usaste botones de sugerencia y no escribiste nada, procesa el último
        if st.session_state.chat and st.session_state.chat[-1]["role"] == "user" and \
           st.session_state.chat[-1]["content"] in ["rojos","ámbar","ayuda","ver PHBTB..."] and not prompt:
            prompt = st.session_state.chat[-1]["content"]

        if prompt:
            # guarda y muestra mensaje del usuario
            st.session_state.chat.append({"role":"user","content":prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # llamada al backend
            try:
                r = requests.post(
                    f"{BACKEND}/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "message": prompt,
                        "use_gpt": bool(use_gpt),
                        "mode": "ml" if chat_mode == "ML" else "rules",
                    },
                    timeout=60,
                )
                answer = r.json().get("answer","(sin respuesta)")
            except Exception as e:
                answer = f"Error consultando backend: {e}"

            # muestra respuesta
            st.session_state.chat.append({"role":"assistant","content":answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

            st.experimental_rerun()  # refrescar

    # --------- 2) Semáforo Reglas ----------
    with tabs[1]:
        st.subheader("Semáforo · Reglas")
        if not semaforo.empty:
            sem_f = filtro_dataframe(semaforo, key="sem_reglas", estado_col="estado")
            st.dataframe(sem_f, use_container_width=True, height=520)
        else:
            st.info("Procesa un archivo para ver el semáforo.")

    # --------- 3) Semáforo ML ----------
    with tabs[2]:
        st.subheader("Semáforo · Modelo ML")
        try:
            df_l = backend_export("timeline")
            if "estado_ml" in df_l.columns:
                last_idx = df_l.groupby("device_id")["ref_date"].idxmax()
                sem_ml = df_l.loc[last_idx].sort_values(["estado_ml","device_id"])
                sem_ml_f = filtro_dataframe(
                    sem_ml.rename(columns={"estado_ml": "estado"}),
                    key="sem_ml",
                    estado_col="estado",
                )
                st.dataframe(sem_ml_f, use_container_width=True, height=520)
            else:
                st.info("Aplica el modelo con '🤖 Aplicar ML'.")
        except Exception as e:
            st.warning(f"No se pudo cargar ML: {e}")

else:
    st.info("Sube y procesa un archivo para comenzar.")

