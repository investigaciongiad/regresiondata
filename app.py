"""
Aplicación Streamlit: Análisis de Regresión Lineal (simple y múltiple).

Carga datos desde CSV o Excel, permite elegir la variable objetivo (Y) y las
variables predictoras (X), ajusta un modelo de regresión lineal con
statsmodels, muestra métricas de desempeño, gráficos representativos y
permite realizar predicciones (manuales y por lotes).

Para ejecutar localmente:
    streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración general de la página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Regresión Lineal — Análisis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
ARCHIVO_EJEMPLO = BASE_DIR / "data" / "datos_ejemplo.csv"

COLOR_LINEA = "#E4572E"
COLOR_AZUL = "#2E86AB"


# ---------------------------------------------------------------------------
# Utilidades de carga de datos
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Leyendo archivo…")
def leer_archivo(nombre_archivo: str, contenido: bytes) -> pd.DataFrame:
    """Lee un CSV o Excel subido. Para CSV prueba varias codificaciones y
    detecta el separador (coma o punto y coma)."""
    sufijo = Path(nombre_archivo).suffix.lower()
    if sufijo == ".xlsx":
        return pd.read_excel(io.BytesIO(contenido), engine="openpyxl")
    if sufijo == ".xls":
        raise ValueError(
            "El formato .xls (Excel antiguo) no está soportado. "
            "Guarda el archivo como .xlsx y vuelve a subirlo."
        )
    # CSV: probar codificaciones y separadores
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        for sep in (None, ";", ","):
            try:
                buf = io.BytesIO(contenido)
                return pd.read_csv(
                    buf,
                    sep=sep,
                    engine="python",
                    encoding=encoding,
                )
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
    # Último recurso: lectura permisiva
    return pd.read_csv(
        io.BytesIO(contenido),
        sep=None,
        engine="python",
        encoding="utf-8",
        encoding_errors="replace",
    )


@st.cache_data(show_spinner=False)
def leer_datos_ejemplo() -> pd.DataFrame:
    return pd.read_csv(ARCHIVO_EJEMPLO)


def convertir_texto_numerico(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Intenta convertir columnas de texto con números (p. ej. '5,5') a
    numéricas. Devuelve el DataFrame transformado y la lista de columnas
    convertidas."""
    convertidas = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype == bool:
            continue
        serie = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
            .str.replace(" ", "", regex=False)
        )
        convertida = pd.to_numeric(serie, errors="coerce")
        if convertida.notna().mean() >= 0.8:
            df[col] = convertida
            convertidas.append(col)
    return df, convertidas


# ---------------------------------------------------------------------------
# Ajuste del modelo
# ---------------------------------------------------------------------------

def metricas_regresion(y_real: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula métricas de desempeño a partir de valores reales y predichos."""
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuos = y_real - y_pred
    n = len(y_real)
    mse = float(np.mean(residuos**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuos)))
    ss_res = float(np.sum(residuos**2))
    ss_tot = float(np.sum((y_real - np.mean(y_real)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "Observaciones (n)": n,
        "R²": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse,
    }


def entrenar_modelo(
    df: pd.DataFrame,
    variable_y: str,
    variables_x: list[str],
    usar_split: bool,
    proporcion_test: float,
    semilla: int,
) -> dict:
    """Ajusta el modelo OLS y devuelve un diccionario con todo lo necesario
    para las métricas, los gráficos y las predicciones."""
    y_todos = df[variable_y].to_numpy(dtype=float)
    exog_todos = sm.add_constant(df[variables_x])

    # División train/test (opcional)
    n_total = len(df)
    usar_split = usar_split and n_total >= 10
    if usar_split:
        n_test = max(1, int(round(n_total * proporcion_test)))
        rng = np.random.default_rng(semilla)
        idx = rng.permutation(n_total)
        idx_test, idx_train = idx[:n_test], idx[n_test:]
        X_train = exog_todos.iloc[idx_train]
        y_train = y_todos[idx_train]
        X_test = exog_todos.iloc[idx_test]
        y_test = y_todos[idx_test]
    else:
        X_train, y_train = exog_todos, y_todos
        X_test = y_test = None

    modelo = sm.OLS(y_train, X_train).fit()

    # Predicciones sobre todos los datos (para gráficos de diagnóstico)
    y_pred_todos = modelo.predict(exog_todos)

    # Tabla de coeficientes
    conf_int = modelo.conf_int()
    tabla_coefs = pd.DataFrame(
        {
            "Variable": modelo.params.index,
            "Coeficiente": modelo.params.values,
            "Error estándar": modelo.bse.values,
            "t": modelo.tvalues.values,
            "p-valor": modelo.pvalues.values,
            "IC 95% inferior": conf_int[0].values,
            "IC 95% superior": conf_int[1].values,
        }
    )
    tabla_coefs.loc[
        tabla_coefs["Variable"] == "const", "Variable"
    ] = "Intercepto (constante)"

    # Métricas de entrenamiento (modelo completo o sobre train)
    metricas_train = metricas_regresion(y_train, modelo.fittedvalues)

    # Métricas sobre el conjunto de test (si aplica)
    metricas_test = None
    if usar_split:
        y_pred_test = modelo.predict(X_test)
        metricas_test = metricas_regresion(y_test, y_pred_test)

    n_entrenamiento = int(len(y_train))

    return {
        "modelo": modelo,
        "tipo": "simple" if len(variables_x) == 1 else "multiple",
        "variable_y": variable_y,
        "variables_x": variables_x,
        "df": df,
        "n_total": n_total,
        "n_entrenamiento": n_entrenamiento,
        "n_test": int(len(y_test)) if usar_split else 0,
        "tabla_coefs": tabla_coefs,
        "metricas_train": metricas_train,
        "metricas_test": metricas_test,
        "y_todos": y_todos,
        "y_pred_todos": y_pred_todos,
        "y_train": y_train,
        "y_test": y_test,
    }


def formula_modelo(resultado: dict) -> str:
    """Devuelve la ecuación del modelo como texto legible."""
    coefs = resultado["modelo"].params
    intercepto = coefs["const"]
    y = resultado["variable_y"]
    terminos = [f"{intercepto:.4f}"]
    for x in resultado["variables_x"]:
        signo = "+" if coefs[x] >= 0 else "-"
        terminos.append(f"{signo} {abs(coefs[x]):.4f}·{x}")
    return f"ŷ ({y}) = " + " ".join(terminos)


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def grafico_simple(resultado: dict) -> go.Figure:
    """Gráfico de dispersión con la recta de regresión y banda de confianza
    (regresión lineal simple: un único predictor)."""
    x_var = resultado["variables_x"][0]
    y_var = resultado["variable_y"]
    df = resultado["df"]
    modelo = resultado["modelo"]

    x_min, x_max = float(df[x_var].min()), float(df[x_var].max())
    if x_max == x_min:
        x_max = x_min + 1.0
    rejilla = np.linspace(x_min, x_max, 100)
    exog_rejilla = sm.add_constant(pd.DataFrame({x_var: rejilla}))
    prediccion = modelo.get_prediction(exog_rejilla).summary_frame(alpha=0.05)

    fig = px.scatter(
        df,
        x=x_var,
        y=y_var,
        opacity=0.75,
        labels={x_var: x_var, y_var: y_var},
        template="plotly_white",
    )
    fig.add_trace(
        go.Scatter(
            x=rejilla,
            y=prediccion["mean"].to_numpy(),
            mode="lines",
            name="Modelo ajustado",
            line=dict(color=COLOR_LINEA, width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([rejilla, rejilla[::-1]]),
            y=np.concatenate(
                [
                    prediccion["mean_ci_lower"].to_numpy(),
                    prediccion["mean_ci_upper"].to_numpy()[::-1],
                ]
            ),
            fill="toself",
            fillcolor="rgba(228, 87, 46, 0.12)",
            line=dict(width=0),
            name="Intervalo de confianza 95%",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        height=480,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def grafico_real_vs_predicho(resultado: dict) -> go.Figure:
    """Valores reales frente a predichos con línea de identidad."""
    y_real = resultado["y_todos"]
    y_pred = resultado["y_pred_todos"]
    minimo = float(min(y_real.min(), y_pred.min()))
    maximo = float(max(y_real.max(), y_pred.max()))

    fig = px.scatter(
        x=y_pred,
        y=y_real,
        opacity=0.7,
        labels={"x": "Valores predichos (ŷ)", "y": "Valores reales (y)"},
        template="plotly_white",
    )
    fig.add_trace(
        go.Scatter(
            x=[minimo, maximo],
            y=[minimo, maximo],
            mode="lines",
            name="Línea de identidad (perfecto)",
            line=dict(color=COLOR_LINEA, width=2, dash="dash"),
        )
    )
    fig.update_layout(height=430)
    return fig


def grafico_residuos(resultado: dict) -> go.Figure:
    """Residuos frente a valores predichos (debe verse sin patrones)."""
    y_pred = resultado["y_pred_todos"]
    residuos = resultado["y_todos"] - resultado["y_pred_todos"]
    fig = px.scatter(
        x=y_pred,
        y=residuos,
        opacity=0.7,
        labels={"x": "Valores predichos (ŷ)", "y": "Residuos"},
        template="plotly_white",
    )
    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_LINEA)
    fig.update_layout(height=400)
    return fig


def grafico_coeficientes(resultado: dict) -> go.Figure:
    """Barras con los coeficientes del modelo y sus intervalos de confianza
    (regresión múltiple)."""
    coefs = resultado["tabla_coefs"]
    coefs = coefs[coefs["Variable"] != "Intercepto (constante)"].copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=coefs["Variable"],
            y=coefs["Coeficiente"],
            marker_color=COLOR_AZUL,
            error_y=dict(
                type="data",
                symmetric=False,
                array=coefs["IC 95% superior"] - coefs["Coeficiente"],
                arrayminus=coefs["Coeficiente"] - coefs["IC 95% inferior"],
                visible=True,
            ),
            name="Coeficiente",
        )
    )
    fig.add_hline(y=0, line_width=1, line_color="#666666")
    fig.update_layout(
        height=430,
        template="plotly_white",
        yaxis_title="Coeficiente estimado",
        xaxis_title="",
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Predicciones
# ---------------------------------------------------------------------------

def predecir_fila(modelo, variables_x: list[str], valores: dict) -> dict:
    """Predice para una fila nueva (diccionario variable -> valor)."""
    fila = pd.DataFrame([{x: valores[x] for x in variables_x}])
    exog = sm.add_constant(fila, has_constant="add")
    marco = modelo.get_prediction(exog).summary_frame(alpha=0.05)
    return {
        "prediccion": float(marco["mean"].iloc[0]),
        "ic_inf": float(marco["obs_ci_lower"].iloc[0]),
        "ic_sup": float(marco["obs_ci_upper"].iloc[0]),
    }


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------

st.title("📈 Análisis de Regresión Lineal")
st.caption(
    "Carga un archivo CSV o Excel, selecciona la variable objetivo (Y) y las "
    "variables predictoras (X). La aplicación construye automáticamente un "
    "modelo de **regresión lineal simple** (1 predictor) o **múltiple** "
    "(2 o más), muestra sus métricas y gráficos, y permite predecir."
)

# ------------------------------- Barra lateral -----------------------------

with st.sidebar:
    st.header("1️⃣ Datos de entrada")

    fuente = st.radio(
        "Origen de los datos",
        options=["Subir archivo (CSV / Excel)", "Usar datos de ejemplo"],
        help=(
            "Los archivos deben tener una fila de encabezado con los nombres "
            "de las variables. Se aceptan .csv y .xlsx."
        ),
    )

    df_original: pd.DataFrame | None = None
    nombre_fuente = ""
    error_datos = None
    if fuente.startswith("Subir"):
        archivo = st.file_uploader(
            "Selecciona un archivo",
            type=["csv", "xlsx"],
            help="Formato: CSV (coma o punto y coma) u hoja de cálculo .xlsx.",
        )
        if archivo is None:
            st.info("⬅️ Sube un archivo CSV o Excel para comenzar.")
            st.stop()
        try:
            df_original = leer_archivo(archivo.name, archivo.getvalue())
            nombre_fuente = archivo.name
        except Exception as exc:  # noqa: BLE001 — mostrar cualquier error de carga
            error_datos = str(exc)
    else:
        if not ARCHIVO_EJEMPLO.exists():
            st.error("No se encontró el archivo de datos de ejemplo.")
            st.stop()
        df_original = leer_datos_ejemplo()
        nombre_fuente = "datos_ejemplo.csv"

    if error_datos is not None:
        st.error(f"No se pudo leer el archivo: {error_datos}")
        st.stop()

    if df_original is None or df_original.empty:
        st.error("El archivo no contiene filas.")
        st.stop()

    # Conversión opcional de texto numérico (p. ej. decimales con coma)
    convertir = st.checkbox(
        "Convertir columnas de texto con números a numéricas",
        value=False,
        help="Útil para CSV exportados desde Excel con decimales tipo '5,5'.",
    )
    if convertir:
        df_original, convertidas = convertir_texto_numerico(df_original.copy())
        if convertidas:
            st.caption(f"Columnas convertidas: {', '.join(convertidas)}")
        else:
            st.caption("No se encontraron columnas convertibles.")

    st.divider()
    st.header("2️⃣ Variables del modelo")

    columnas_numericas = list(
        df_original.select_dtypes(include=[np.number]).columns
    )
    if not columnas_numericas:
        st.error(
            "No se encontraron variables numéricas en el archivo. Revisa que "
            "los datos sean numéricos (o activa la conversión anterior)."
        )
        st.stop()

    variable_y = st.selectbox(
        "Variable objetivo (Y)",
        options=columnas_numericas,
        help="Variable que el modelo intenta explicar o predecir.",
    )
    variables_x = st.multiselect(
        "Variables predictoras (X)",
        options=[c for c in columnas_numericas if c != variable_y],
        default=[c for c in columnas_numericas if c != variable_y][:2],
        help=(
            "Elige 1 variable para regresión simple o varias para regresión "
            "múltiple."
        ),
    )

    if not variables_x:
        st.warning("Selecciona al menos una variable predictora (X).")
        st.stop()

    tipo_modelo = "simple" if len(variables_x) == 1 else "múltiple"
    st.success(
        f"Modelo a construir: **regresión lineal {tipo_modelo}** con "
        f"{len(variables_x)} predictor(es)."
    )

    st.divider()
    st.header("3️⃣ Opciones de ajuste")

    usar_split = st.toggle(
        "Separar datos en entrenamiento y prueba",
        value=False,
        help="Entrena con una parte de los datos y evalúa con la parte restante.",
    )
    proporcion_test = 0.25
    if usar_split:
        proporcion_test = st.slider(
            "Proporción de prueba", 0.1, 0.5, 0.25, 0.05, format="%.2f"
        )
    semilla = st.number_input(
        "Semilla aleatoria", min_value=0, value=42, step=1,
        help="Para que la división train/test sea reproducible.",
    )

# ------------------------------ Cuerpo principal ----------------------------

# Limpieza: solo filas sin valores faltantes en las columnas seleccionadas
columnas_modelo = [variable_y] + variables_x
df_modelo = df_original[columnas_modelo].dropna().copy()
filas_eliminadas = len(df_original) - len(df_modelo)
if df_modelo.empty:
    st.error("Después de eliminar filas con valores faltantes no quedan datos.")
    st.stop()

# Clave para reutilizar el modelo ya ajustado entre interacciones
clave_firma = (
    pd.util.hash_pandas_object(df_original[columnas_modelo]).sum(),
    tuple(variables_x),
    variable_y,
    usar_split,
    proporcion_test,
    int(semilla),
)
clave_modelo = f"modelo_{hash(clave_firma)}"
if clave_modelo not in st.session_state:
    with st.spinner("Ajustando el modelo…"):
        st.session_state[clave_modelo] = entrenar_modelo(
            df_modelo, variable_y, variables_x, usar_split, proporcion_test,
            int(semilla),
        )
resultado = st.session_state[clave_modelo]

# Encabezado del modelo
st.subheader(
    f"🧮 Modelo ajustado: regresión lineal "
    f"{'simple' if resultado['tipo'] == 'simple' else 'múltiple'}"
)
st.markdown(f"**Ecuación del modelo**  \n`{formula_modelo(resultado)}`")

if filas_eliminadas > 0:
    st.caption(
        f"ℹ️ Se usaron {len(df_modelo)} filas de {len(df_original)} "
        f"({filas_eliminadas} eliminadas por valores faltantes)."
    )

# ------------------------------- Métricas ----------------------------------

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
met_train = resultado["metricas_train"]
col_m1.metric("R² (entrenamiento)", f"{met_train['R²']:.4f}")
col_m2.metric("RMSE", f"{met_train['RMSE']:.4f}")
col_m3.metric("MAE", f"{met_train['MAE']:.4f}")
col_m4.metric("Observaciones", f"{resultado['n_entrenamiento']}")
if resultado["metricas_test"] is not None:
    met_test = resultado["metricas_test"]
    col_m5.metric("R² (prueba)", f"{met_test['R²']:.4f}")
else:
    col_m5.metric("Split train/test", "No aplicado")

st.divider()

# --------------------------- Tabla de coeficientes --------------------------

with st.expander("📋 Tabla de coeficientes del modelo", expanded=True):
    coefs_tabla = resultado["tabla_coefs"].copy()
    st.dataframe(
        coefs_tabla,
        width="stretch",
        hide_index=True,
        column_config={
            col: st.column_config.NumberColumn(col, format="%.4g")
            for col in coefs_tabla.columns
            if col != "Variable"
        },
    )
    csv_coefs = coefs_tabla.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Descargar coeficientes (CSV)",
        data=csv_coefs,
        file_name="coeficientes_modelo.csv",
        mime="text/csv",
    )

# -------------------------------- Gráficos ----------------------------------

st.subheader("📊 Gráficos del modelo")

if resultado["tipo"] == "simple":
    st.markdown(
        "**Dispersión y recta de regresión.** Los puntos son los datos "
        "observados; la línea naranja es el modelo y la banda, el intervalo "
        "de confianza del 95% para la media predicha."
    )
    st.plotly_chart(grafico_simple(resultado), width="stretch")
else:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Coeficientes estimados** (con intervalo de confianza 95%).")
        st.plotly_chart(grafico_coeficientes(resultado), width="stretch")
    with c2:
        st.markdown("**Valores reales vs. predichos.**")
        st.plotly_chart(grafico_real_vs_predicho(resultado), width="stretch")

    if resultado["metricas_test"] is not None:
        met = resultado["metricas_test"]
        st.info(
            f"Rendimiento en datos de **prueba** (no vistos durante el "
            f"entrenamiento): R² = {met['R²']:.4f}, RMSE = {met['RMSE']:.4f}."
        )

st.markdown("**Residuos** (diferencias entre lo real y lo predicho).")
st.plotly_chart(grafico_residuos(resultado), width="stretch")

# --------------------- Análisis exploratorio (opcional) ---------------------

with st.expander("🔍 Análisis exploratorio de los datos"):
    pestaña_previa, pestaña_correlacion = st.tabs(
        ["Vista previa y estadísticos", "Matriz de correlación"]
    )
    with pestaña_previa:
        st.dataframe(df_original, width="stretch")
        st.dataframe(
            df_original.describe().T, width="stretch"
        )
    with pestaña_correlacion:
        corr = df_original.select_dtypes(include=[np.number]).corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            template="plotly_white",
            labels=dict(color="Correlación"),
        )
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, width="stretch")

st.divider()

# ------------------------------- Predicciones -------------------------------

st.subheader("🔮 Predicciones con el modelo")

if resultado["tipo"] == "simple":
    st.markdown(
        "Usa el control para mover el valor de la variable predictora y mira "
        "cómo cambia la predicción en tiempo real."
    )
else:
    st.markdown(
        "Introduce valores para cada variable predictora; la predicción se "
        "actualiza automáticamente."
    )

columnas_pred = st.columns(min(len(variables_x), 4))
valores_nuevos: dict[str, float] = {}
for i, x in enumerate(variables_x):
    col_serie = df_modelo[x]
    paso = 1.0 if pd.api.types.is_integer_dtype(df_original[x]) else None
    minimo = float(col_serie.min())
    maximo = float(col_serie.max())
    if maximo == minimo:
        maximo = minimo + 1.0
    with columnas_pred[i % len(columnas_pred)]:
        valores_nuevos[x] = st.number_input(
            f"Valor de **{x}**",
            min_value=minimo,
            max_value=maximo,
            value=float(col_serie.median()),
            step=paso,
            format="%.4f" if paso is None else "%.0f",
        )

prediccion = predecir_fila(resultado["modelo"], variables_x, valores_nuevos)
p1, p2, p3 = st.columns(3)
p1.metric("Predicción de " + variable_y, f"{prediccion['prediccion']:,.4f}")
p2.metric("Intervalo de predicción 95% (inf.)", f"{prediccion['ic_inf']:,.4f}")
p3.metric("Intervalo de predicción 95% (sup.)", f"{prediccion['ic_sup']:,.4f}")

st.markdown(
    "💡 El intervalo de predicción indica el rango en el que se espera que "
    "caiga una **nueva observación** individual el 95% de las veces."
)

# ----------------------- Predicción por lotes (archivo) ---------------------

with st.expander("📁 Predecir con un archivo (lote)"):
    st.markdown(
        f"Sube un CSV o Excel con las mismas columnas predictoras "
        f"(`{', '.join(variables_x)}`). Se añadirá una columna `prediccion` "
        f"con el valor estimado de **{variable_y}** para cada fila."
    )
    archivo_lote = st.file_uploader(
        "Archivo para predecir", type=["csv", "xlsx"], key="lote"
    )
    if archivo_lote is not None:
        try:
            df_lote = leer_archivo(archivo_lote.name, archivo_lote.getvalue())
            faltantes = [x for x in variables_x if x not in df_lote.columns]
            if faltantes:
                st.error(
                    "El archivo no contiene las columnas: "
                    + ", ".join(faltantes)
                )
            else:
                df_lote_limpio = df_lote[variables_x].dropna()
                if len(df_lote_limpio) != len(df_lote):
                    st.warning(
                        f"{len(df_lote) - len(df_lote_limpio)} fila(s) con "
                        "valores faltantes fueron omitidas."
                    )
                exog_lote = sm.add_constant(df_lote_limpio, has_constant="add")
                exog_lote = exog_lote.reindex(
                    columns=resultado["modelo"].model.exog_names, fill_value=0
                )
                df_lote_limpio = df_lote_limpio.copy()
                df_lote_limpio[f"prediccion_{variable_y}"] = resultado[
                    "modelo"
                ].predict(exog_lote)
                st.dataframe(df_lote_limpio, width="stretch")
                st.download_button(
                    "⬇️ Descargar predicciones (CSV)",
                    data=df_lote_limpio.to_csv(index=False).encode("utf-8-sig"),
                    file_name="predicciones.csv",
                    mime="text/csv",
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo procesar el archivo: {exc}")

st.divider()
st.caption(
    "Modelo ajustado por mínimos cuadrados ordinarios (statsmodels OLS). "
    "Fuente de datos: "
    + (nombre_fuente if nombre_fuente else "desconocida")
    + "."
)
