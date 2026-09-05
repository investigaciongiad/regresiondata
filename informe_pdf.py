"""
Generación del informe PDF del modelo de regresión.

Módulo independiente de la interfaz (no usa Streamlit) para poder probarlo
fácilmente. El informe incluye:

    1. Resumen del modelo (tipo, variables, ecuación, observaciones)
    2. Métricas de desempeño (entrenamiento y prueba si aplica)
    3. Tabla de coeficientes
    4. Interpretación automática de los resultados
    5. Gráficos representativos (exportados desde las figuras de Plotly)
    6. Predicción actual (si se proporciona)
"""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
from plotly import io as pio
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

COLOR_PRIMARIO = colors.HexColor("#2E86AB")
COLOR_ACENTO = colors.HexColor("#E4572E")
COLOR_FILA_ALT = colors.HexColor("#EAF2F8")
COLOR_TEXTO = colors.HexColor("#1F2933")
COLOR_SUAVE = colors.HexColor("#6B7A8F")
ANCHO_UTIL = A4[0] - 2 * 1.9 * cm

# Símbolos no incluidos en la codificación WinAnsi/Latin-1 del PDF
_REEMPLAZOS = {
    "ŷ": "Y_pred",
    "−": "-",
    "–": "-",
    "—": "-",
    "•": "-",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "…": "...",
    "→": "->",
    "≥": ">=",
    "≤": "<=",
    "√": "raiz",
}


def _seguro(texto: object) -> str:
    """Convierte cualquier texto a algo seguro para las fuentes estándar del
    PDF (acentos españoles incluidos)."""
    texto = str(texto)
    for origen, destino in _REEMPLAZOS.items():
        texto = texto.replace(origen, destino)
    # Escapar caracteres especiales de XML y limitar a WinAnsi/Latin-1
    texto = (
        texto.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return texto.encode("latin-1", "replace").decode("latin-1")


def _fmt_numero(valor: float) -> str:
    """Formatea números para las tablas de forma legible."""
    import math

    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "—"
    magnitud = abs(valor)
    if magnitud == 0:
        return "0"
    if magnitud >= 1_000_000 or magnitud < 1e-4:
        return f"{valor:.3e}"
    return f"{valor:,.4f}"


def _estilos() -> dict[str, ParagraphStyle]:
    titulo = ParagraphStyle(
        "titulo",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=COLOR_PRIMARIO,
        spaceAfter=4,
    )
    subtitulo = ParagraphStyle(
        "subtitulo",
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        textColor=COLOR_SUAVE,
        spaceAfter=10,
    )
    h1 = ParagraphStyle(
        "h1",
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=COLOR_PRIMARIO,
        spaceBefore=14,
        spaceAfter=6,
    )
    cuerpo = ParagraphStyle(
        "cuerpo",
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=COLOR_TEXTO,
    )
    negrita = ParagraphStyle(
        "negrita", parent=cuerpo, fontName="Helvetica-Bold"
    )
    ecuacion = ParagraphStyle(
        "ecuacion",
        fontName="Courier",
        fontSize=9,
        leading=12.5,
        textColor=COLOR_TEXTO,
        backColor=colors.HexColor("#F0F4F8"),
        borderPadding=5,
        spaceBefore=4,
        spaceAfter=4,
    )
    nota = ParagraphStyle(
        "nota",
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=10.5,
        textColor=COLOR_SUAVE,
    )
    celda = ParagraphStyle(
        "celda",
        fontName="Helvetica",
        fontSize=8.5,
        leading=10.5,
        textColor=COLOR_TEXTO,
    )
    celda_cabecera = ParagraphStyle(
        "celda_cabecera",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=10.5,
        textColor=colors.white,
    )
    return {
        "titulo": titulo,
        "subtitulo": subtitulo,
        "h1": h1,
        "cuerpo": cuerpo,
        "negrita": negrita,
        "ecuacion": ecuacion,
        "nota": nota,
        "celda": celda,
        "celda_cabecera": celda_cabecera,
    }


def _tabla(datos: list[list[str]], anchos: list[float]) -> Table:
    """Construye una tabla con el estilo visual de la aplicación."""
    estilos = _estilos()
    filas = []
    for i, fila in enumerate(datos):
        if i == 0:
            filas.append(
                [Paragraph(_seguro(c), estilos["celda_cabecera"]) for c in fila]
            )
        else:
            filas.append([Paragraph(_seguro(c), estilos["celda"]) for c in fila])

    tabla = Table(filas, colWidths=[a * cm for a in anchos], repeatRows=1)
    estilo = [
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARIO),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C5D3DE")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if len(datos) > 1:
        estilo.append(
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_FILA_ALT])
        )
    tabla.setStyle(TableStyle(estilo))
    return tabla


def _pie_pagina(canvas, doc) -> None:  # noqa: ANN001
    """Dibuja el pie de página con la fecha y el número de página."""
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(COLOR_SUAVE)
    canvas.drawCentredString(
        A4[0] / 2,
        1.15 * cm,
        _seguro(
            "Generado el "
            + datetime.now().strftime("%d/%m/%Y a las %H:%M")
            + f"   |   Página {doc.page}"
        ),
    )
    canvas.restoreState()


def _imagen_desde_figura(figura) -> Image | None:
    """Convierte una figura de Plotly a PNG y la devuelve como flowable."""
    try:
        png = pio.to_image(figura, format="png", width=1100, height=620)
        lector = ImageReader(io.BytesIO(png))
        ancho_natural, alto_natural = lector.getSize()
        ancho = ANCHO_UTIL
        alto = ancho * alto_natural / ancho_natural
        return Image(io.BytesIO(png), width=ancho, height=alto)
    except Exception:  # noqa: BLE001 — si falla la exportación, se omite el gráfico
        return None


def _resumen_de_interpretacion(
    r2_train: float,
    rmse_train: float,
    tabla_coefs: pd.DataFrame,
) -> list[str]:
    """Construye frases automáticas de interpretación de los resultados."""
    frases = []
    frases.append(
        f"El modelo explica aproximadamente un {r2_train * 100:.1f}% de la "
        f"variabilidad de la variable objetivo (R² = {r2_train:.4f})."
    )
    significativas = [
        str(x)
        for x in tabla_coefs.loc[
            (tabla_coefs["Variable"] != "Intercepto (constante)")
            & (tabla_coefs["p-valor"] < 0.05),
            "Variable",
        ]
    ]
    if significativas:
        frases.append(
            "Con un nivel de significancia del 5%, las variables "
            + ", ".join(f"'{x}'" for x in significativas)
            + " tienen una relación estadísticamente significativa con la "
            "variable objetivo."
        )
    else:
        frases.append(
            "Con un nivel de significancia del 5%, ninguna de las variables "
            "predictoras resultó estadísticamente significativa."
        )
    frases.append(
        f"El error promedio de las predicciones (RMSE) es de {rmse_train:,.4f} "
        "unidades de la variable objetivo; cuanto menor, más precisas son las "
        "predicciones."
    )
    frases.append(
        "Recuerda que la regresión lineal asume relación lineal, "
        "independencia y homocedasticidad de los residuos, entre otros "
        "supuestos que conviene verificar antes de usar el modelo."
    )
    return frases


def generar_informe_pdf(
    *,
    nombre_archivo: str,
    tipo: str,
    variable_y: str,
    variables_x: list[str],
    ecuacion: str,
    n_total: int,
    n_entrenamiento: int,
    n_test: int,
    usar_split: bool,
    metricas_train: dict[str, float],
    metricas_test: dict[str, float] | None,
    tabla_coefs: pd.DataFrame,
    figuras: list[tuple[str, object]],
    valores_prediccion: dict[str, float] | None = None,
    prediccion: dict | None = None,
) -> bytes:
    """Genera el informe PDF completo y devuelve su contenido en bytes."""
    s = _estilos()
    tipo_etiqueta = "simple" if tipo == "simple" else "múltiple"
    fecha = datetime.now().strftime("%d/%m/%Y %H:%M")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.9 * cm,
        rightMargin=1.9 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.9 * cm,
        title="Informe de Análisis de Regresión Lineal",
        author="Aplicación de Análisis de Regresión Lineal (Streamlit)",
    )

    historia = []

    # ------------------------------ Encabezado -----------------------------
    historia.append(Paragraph("Informe de Regresión Lineal", s["titulo"]))
    historia.append(
        Paragraph(
            f"Modelo: regresión lineal {tipo_etiqueta}  |  Fecha: {fecha}",
            s["subtitulo"],
        )
    )

    # ----------------------------- 1. Resumen ------------------------------
    historia.append(Paragraph("1. Resumen del modelo", s["h1"]))
    observaciones = f"{n_total:,} filas usadas"
    if usar_split:
        observaciones += (
            f"  (entrenamiento: {n_entrenamiento:,}, prueba: {n_test:,})"
        )
    lineas_resumen = [
        ("Fuente de datos", nombre_archivo),
        ("Tipo de modelo", f"Regresión lineal {tipo_etiqueta}"),
        ("Variable objetivo (Y)", variable_y),
        ("Variables predictoras (X)", ", ".join(variables_x)),
        ("Observaciones", observaciones),
    ]
    for etiqueta, valor in lineas_resumen:
        historia.append(
            Paragraph(
                f"<b>{_seguro(etiqueta)}:</b> {_seguro(valor)}",
                s["cuerpo"],
            )
        )
    historia.append(Spacer(1, 4))
    historia.append(Paragraph("Ecuación del modelo ajustado", s["negrita"]))
    historia.append(Paragraph(_seguro(ecuacion), s["ecuacion"]))

    # ---------------------------- 2. Métricas ------------------------------
    historia.append(Paragraph("2. Métricas de desempeño", s["h1"]))
    def _r2_ajustado(r2: float, n_obs: int) -> float:
        if n_obs > len(variables_x) + 1:
            return 1 - (1 - r2) * (n_obs - 1) / (
                n_obs - len(variables_x) - 1
            )
        return float("nan")

    r2_train = float(metricas_train["R²"])
    rmse_train = float(metricas_train["RMSE"])

    filas_metricas = [
        ["Métrica", "Entrenamiento", "Prueba" if usar_split else ""]
    ]
    etiquetas = ["R²", "R² ajustado", "RMSE", "MAE", "MSE"]
    for etiqueta in etiquetas:
        if etiqueta == "R² ajustado":
            valor_train = _r2_ajustado(r2_train, n_entrenamiento)
            if metricas_test is not None:
                r2_test = float(metricas_test["R²"])
                valor_test = _r2_ajustado(r2_test, n_test)
            else:
                valor_test = None
        else:
            valor_train = float(metricas_train[etiqueta])
            valor_test = (
                float(metricas_test[etiqueta])
                if metricas_test is not None
                else None
            )
        filas_metricas.append(
            [
                etiqueta,
                _fmt_numero(valor_train),
                _fmt_numero(valor_test) if valor_test is not None else "—",
            ]
        )
    historia.append(_tabla(filas_metricas, [5.5, 4.5, 4.5]))
    if usar_split:
        historia.append(
            Paragraph(
                "Las métricas de 'Prueba' se calcularon con datos que el "
                "modelo no vio durante el entrenamiento.",
                s["nota"],
            )
        )

    # ------------------------- 3. Coeficientes -----------------------------
    historia.append(Paragraph("3. Coeficientes del modelo", s["h1"]))
    filas_coef = [
        ["Variable", "Coeficiente", "Error estándar", "t", "p-valor", "IC 95%"]
    ]
    for _, fila in tabla_coefs.iterrows():
        ic = (
            f"{_fmt_numero(float(fila['IC 95% inferior']))} a "
            f"{_fmt_numero(float(fila['IC 95% superior']))}"
        )
        filas_coef.append(
            [
                str(fila["Variable"]),
                _fmt_numero(float(fila["Coeficiente"])),
                _fmt_numero(float(fila["Error estándar"])),
                _fmt_numero(float(fila["t"])),
                _fmt_numero(float(fila["p-valor"])),
                ic,
            ]
        )
    historia.append(
        _tabla(filas_coef, [4.3, 2.5, 2.3, 1.7, 1.9, 3.1])
    )
    historia.append(
        Paragraph(
            "IC 95%: intervalo de confianza del coeficiente. Un p-valor menor "
            "que 0.05 indica significancia estadística.",
            s["nota"],
        )
    )

    # ------------------------- 4. Interpretación ---------------------------
    historia.append(Paragraph("4. Interpretación", s["h1"]))
    for frase in _resumen_de_interpretacion(
        r2_train, rmse_train, tabla_coefs
    ):
        historia.append(Paragraph(f"- {_seguro(frase)}", s["cuerpo"]))

    # ---------------------------- 5. Gráficos ------------------------------
    if figuras:
        historia.append(Paragraph("5. Gráficos del modelo", s["h1"]))
        for titulo, figura in figuras:
            imagen = _imagen_desde_figura(figura)
            if imagen is not None:
                historia.append(Paragraph(_seguro(titulo), s["negrita"]))
                historia.append(imagen)
                historia.append(Spacer(1, 8))
            else:
                historia.append(
                    Paragraph(
                        f"- {_seguro(titulo)}: no se pudo generar la imagen.",
                        s["nota"],
                    )
                )

    # ------------------------- 6. Predicción -------------------------------
    if prediccion is not None and valores_prediccion is not None:
        historia.append(Paragraph("6. Predicción realizada", s["h1"]))
        filas_pred = [["Variable", "Valor utilizado"]]
        for x in variables_x:
            filas_pred.append([x, _fmt_numero(float(valores_prediccion[x]))])
        historia.append(_tabla(filas_pred, [8.0, 7.5]))
        historia.append(Spacer(1, 6))
        historia.append(
            Paragraph(
                f"<b>Valor predicho de '{_seguro(variable_y)}':</b> "
                f"{float(prediccion['prediccion']):,.4f}",
                s["cuerpo"],
            )
        )
        historia.append(
            Paragraph(
                f"Intervalo de predicción al 95%: "
                f"{float(prediccion['ic_inf']):,.4f} a "
                f"{float(prediccion['ic_sup']):,.4f}.",
                s["cuerpo"],
            )
        )

    # ------------------------------ Pie ------------------------------------
    historia.append(Spacer(1, 10))
    historia.append(
        Paragraph(
            "Informe generado automáticamente por la aplicación 'Análisis de "
            "Regresión Lineal' (Streamlit + statsmodels OLS).",
            s["nota"],
        )
    )

    doc.build(historia, onFirstPage=_pie_pagina, onLaterPages=_pie_pagina)
    return buf.getvalue()
