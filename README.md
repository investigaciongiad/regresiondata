# 📈 Análisis de Regresión Lineal con Streamlit

Aplicación web construida con [Streamlit](https://streamlit.io/) y Python para
realizar **análisis de regresión lineal simple o múltiple** a partir de un
archivo de datos **CSV o Excel** que tú cargues.

## ✨ Funcionalidades

- **Carga de datos**: sube un archivo `.csv` (coma o punto y coma) o `.xlsx`,
  o prueba con el conjunto de datos de ejemplo incluido (`data/datos_ejemplo.csv`).
- **Reconocimiento automático de variables**: la aplicación detecta las
  columnas numéricas del archivo y las muestra para que selecciones la
  variable objetivo (Y) y las predictoras (X). También puedes convertir
  columnas de texto con números (p. ej. decimales con coma `5,5`) a numéricas.
- **Modelo simple o múltiple automático**: al elegir **1 predictor** se
  construye una regresión lineal simple; con **2 o más**, una regresión
  múltiple (mínimos cuadrados ordinarios con `statsmodels`).
- **Métricas de desempeño**: R², R² del conjunto de prueba (opcional), RMSE,
  MAE, MSE y número de observaciones. Opcionalmente puedes separar los datos
  en **entrenamiento/prueba**.
- **Tabla de coeficientes**: intercepto, coeficientes, error estándar,
  estadístico t, p-valor e intervalos de confianza del 95%, descargable en CSV.
- **Gráficos representativos**:
  - Regresión simple: dispersión con la recta del modelo y banda de confianza.
  - Regresión múltiple: coeficientes con sus intervalos de confianza y gráfico
    de valores reales vs. predichos.
  - Residuos frente a valores predichos (diagnóstico del modelo).
  - Análisis exploratorio: matriz de correlación y estadísticos descriptivos.
- **Predicciones**:
  - **Manual**: controles para cada variable predictora con predicción e
    intervalo de predicción del 95% en tiempo real.
  - **Por lotes**: sube un CSV/Excel con las mismas columnas predictoras y
    descarga las predicciones resultantes.
- **Informe en PDF**: genera y descarga un informe profesional con el resumen
  del modelo, su ecuación, métricas de desempeño, tabla de coeficientes,
  interpretación automática, gráficos y la predicción actual.

## 🗂️ Estructura del proyecto

```
├── app.py                        # Aplicación Streamlit (interfaz)
├── informe_pdf.py                # Generación del informe PDF (reportlab)
├── requirements.txt              # Dependencias de Python
├── .streamlit/
│   └── config.toml               # Configuración de tema y servidor
├── data/
│   └── datos_ejemplo.csv         # Dataset de ejemplo (simulado)
└── README.md
```

## 🚀 Ejecutar localmente

Requisitos: Python 3.10 o superior.

```bash
# 1. Clonar o descargar el proyecto y entrar en la carpeta
cd analisis-regresion-streamlit

# 2. Crear y activar un entorno virtual
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

Se abrirá el navegador en `http://localhost:8501`.

## ☁️ Desplegar en Streamlit Community Cloud

El despliegue es gratuito y se hace conectando un repositorio de GitHub.

### 1. Sube el proyecto a GitHub

1. Crea una cuenta en [github.com](https://github.com) si no la tienes.
2. Crea un **nuevo repositorio** (botón verde *New*). Puedes dejarlo vacío
   (sin README) y usar el nombre `analisis-regresion-streamlit`.
3. Sube el código. Desde la terminal, dentro de la carpeta del proyecto:

```bash
git init
git add .
git commit -m "Aplicación de análisis de regresión lineal con Streamlit"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/analisis-regresion-streamlit.git
git push -u origin main
```

> Sustituye `TU_USUARIO` por tu nombre de usuario de GitHub. También puedes
> subir los archivos desde la propia web de GitHub (botón *Add file* →
> *Upload files*), sin usar la terminal.

### 2. Conecta el repositorio en Streamlit Cloud

1. Entra en [share.streamlit.io](https://share.streamlit.io) (o
   [streamlit.io/cloud](https://streamlit.io/cloud)) e inicia sesión **con tu
   cuenta de GitHub**.
2. Pulsa **Create app** → **Deploy a public app from GitHub**.
3. Acepta la autorización que GitHub te pida para que Streamlit pueda leer tus
   repositorios.
4. Selecciona:
   - **Repository**: `TU_USUARIO/analisis-regresion-streamlit`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Pulsa **Deploy**.

En unos minutos la aplicación estará publicada en una URL del tipo
`https://TU_USUARIO-analisis-regresion-streamlit.streamlit.app`.

### 3. Actualizar la aplicación

Cada vez que hagas `git push` a la rama `main`, Streamlit Cloud vuelve a
desplegar la aplicación automáticamente.

## 📄 Formato de los datos de entrada

- Un archivo **CSV** (separado por coma o punto y coma) o **Excel** (`.xlsx`).
- La primera fila debe contener los **nombres de las variables**.
- Cada columna es una variable; al menos dos deben ser **numéricas** (una
  objetivo y una o más predictoras).
- Las filas con valores faltantes en las columnas seleccionadas se eliminan
  automáticamente antes de ajustar el modelo.
- Si tu CSV viene de Excel en español (decimales con coma), activa la casilla
  *"Convertir columnas de texto con números a numéricas"* en la barra lateral.

> **Nota**: los archivos `.xls` (Excel antiguo) no están soportados; guarda la
> hoja como `.xlsx`.

## 🛠️ Tecnologías

- [Streamlit](https://streamlit.io/) — interfaz web
- [statsmodels](https://www.statsmodels.org/) — regresión por mínimos
  cuadrados ordinarios (OLS)
- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) — manejo
  de datos
- [Plotly](https://plotly.com/python/) — gráficos interactivos
- [openpyxl](https://openpyxl.readthedocs.io/) — lectura de archivos Excel
- [reportlab](https://www.reportlab.com/) — generación del informe PDF
- [kaleido](https://github.com/plotly/Kaleido) — exportación de los gráficos
  de Plotly a imágenes para el PDF

## 📝 Notas sobre la interpretación

- **R²** indica la proporción de la variabilidad de Y explicada por el modelo
  (0 a 1; más alto, mejor ajuste a los datos observados).
- **p-valor** de cada coeficiente: si es menor que 0.05, la variable tiene una
  relación estadísticamente significativa con Y.
- **RMSE / MAE** miden el error promedio de las predicciones en las mismas
  unidades que Y (menor es mejor).
- Un buen modelo de regresión debe tener residuos distribuidos al azar
  alrededor de cero, sin patrones claros en el gráfico de residuos.
