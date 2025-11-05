import streamlit as st
from PIL import Image
import os

# =========================
# CONFIGURACIÓN DE LA PÁGINA
# =========================
st.set_page_config(
    page_title="Inicio | Procesamiento de Datasets en ML",
    page_icon="🤖",
    layout="wide"
)

# =========================
# SECCIÓN 1: PORTADA Y TÍTULO
# =========================
PATH_PORTADA = "img/portada_ml.jpg"

st.markdown(
    '''
    <style>
        .title-text {
            font-size: 2.4em; 
            font-weight: 800; 
            margin-bottom: 0px;
            color: #1E3A8A;
        }
        .subtitle-text {
            font-size: 1.3em; 
            font-weight: 500;
            margin-top: 5px;
            color: #4B5563;
        }
        .author-text {
            font-size: 1.1em; 
            font-weight: 400;
            margin-top: 0px;
            color: #4B5563;
        }
        .body-text {
            font-size: 1.1em;
            line-height: 1.6;
            color: #374151;
        }
        .highlight-box {
            background-color: #EFF6FF;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #1E3A8A;
            margin: 20px 0;
        }
        .step-card {
            background-color: #F9FAFB;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 3px solid #3B82F6;
        }
    </style>
    ''',
    unsafe_allow_html=True
)

try:
    portada = Image.open(PATH_PORTADA)
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(portada, width=220)
    with col2:
        st.markdown("<h1 class='title-text'>Procesamiento de Datasets en Machine Learning</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle-text'>Actividad Individual Práctica</p>", unsafe_allow_html=True)
        st.markdown("<p class='author-text'>Universidad Nacional de Trujillo</p>", unsafe_allow_html=True)
except FileNotFoundError:
    st.markdown("<h1 class='title-text'>Procesamiento de Datasets en Machine Learning</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Actividad Individual Práctica</p>", unsafe_allow_html=True)
    st.warning(f"⚠️ No se encontró la imagen de portada en la ruta: {PATH_PORTADA}")

# =========================
# SECCIÓN 2: DESCRIPCIÓN DE LA APP
# =========================
st.markdown(
    """
    <div class='body-text' style='margin-top: 20px;'>
        Esta aplicación implementa un flujo completo de <b>procesamiento de datos</b> aplicado a 
        tres datasets clásicos de Machine Learning. Cada ejercicio demuestra las etapas fundamentales 
        del preprocesamiento de datos, desde la carga hasta la preparación final para modelos predictivos.
        <br><br>
        Utilice el menú lateral <b>(☰)</b> para navegar entre los diferentes ejercicios.
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# SECCIÓN 3: ETAPAS DEL PROCESAMIENTO
# =========================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🪜 Etapas del Procesamiento de Datos")

etapas = [
    ("1️⃣", "Carga del Dataset", "Importación de datos desde archivos CSV o bibliotecas"),
    ("2️⃣", "Exploración Inicial", "Análisis de estructura, tipos de datos y valores nulos"),
    ("3️⃣", "Limpieza de Datos", "Tratamiento de valores faltantes, duplicados y outliers"),
    ("4️⃣", "Codificación", "Transformación de variables categóricas a numéricas"),
    ("5️⃣", "Normalización", "Estandarización de características numéricas"),
    ("6️⃣", "División de Datos", "Separación en conjuntos de entrenamiento y prueba")
]

cols = st.columns(3)
for i, (emoji, titulo, desc) in enumerate(etapas):
    with cols[i % 3]:
        st.markdown(
            f"""
            <div class='step-card'>
                <h4 style='margin:0; color:#1E3A8A;'>{emoji} {titulo}</h4>
                <p style='margin:5px 0 0 0; font-size:0.9em; color:#6B7280;'>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================
# SECCIÓN 4: EJERCICIOS DISPONIBLES
# =========================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📚 Ejercicios Disponibles")

ejercicios = [
    {
        "icono": "🚢",
        "titulo": "Ejercicio 1: Dataset Titanic",
        "descripcion": "Preparación de datos para predecir la supervivencia de pasajeros",
        "caracteristicas": [
            "Análisis de 891 pasajeros",
            "Tratamiento de valores nulos",
            "Codificación de variables categóricas",
            "División 70/30"
        ]
    },
    {
        "icono": "🎓",
        "titulo": "Ejercicio 2: Student Performance",
        "descripcion": "Procesamiento para predecir el rendimiento académico de estudiantes",
        "caracteristicas": [
            "Análisis de factores académicos y sociales",
            "One-Hot Encoding",
            "Normalización de variables",
            "División 80/20"
        ]
    },
    {
        "icono": "🌸",
        "titulo": "Ejercicio 3: Dataset Iris",
        "descripcion": "Flujo completo con visualización de características",
        "caracteristicas": [
            "Dataset clásico de clasificación",
            "Estandarización con StandardScaler",
            "Visualización interactiva",
            "División 70/30"
        ]
    }
]

for ejercicio in ejercicios:
    with st.expander(f"{ejercicio['icono']} **{ejercicio['titulo']}**", expanded=False):
        st.markdown(f"**Objetivo:** {ejercicio['descripcion']}")
        st.markdown("**Características principales:**")
        for carac in ejercicio['caracteristicas']:
            st.markdown(f"- {carac}")

# =========================
# SECCIÓN 5: INSTRUCCIONES
# =========================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='highlight-box'>
        <h3 style='margin-top:0; color:#1E3A8A;'>💡 Instrucciones de Uso</h3>
        <ol style='line-height: 1.8;'>
            <li>Navegue por el <b>menú lateral</b> para acceder a cada ejercicio</li>
            <li>Cada página incluye el <b>código completo</b> con explicaciones</li>
            <li>Los resultados se muestran en <b>tiempo real</b> con tablas y gráficos</li>
            <li>Puede descargar los datos procesados desde cada ejercicio</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# SECCIÓN 6: INFORMACIÓN DEL DESARROLLADOR
# =========================
PATH_LOGO_UNT = "img/UNT_logo.png"

try:
    logo_unt = Image.open(PATH_LOGO_UNT)
    col_dev, col_logo = st.columns([5, 1])
    with col_dev:
        st.markdown(
            """
            <p style='font-size: 0.95em; font-weight: 600; margin-bottom: 3px;'>
                Desarrollado por: <span style='color:#1E3A8A;'>Tu Nombre Completo</span>
            </p>
            <p style='font-size: 0.9em; margin-top: 0px; margin-bottom: 2px; color:#4B5563;'>
                Escuela Profesional de Ingeniería de Sistemas
            </p>
            <p style='font-size: 0.9em; margin-top: 0px; color:#4B5563;'>
                Universidad Nacional de Trujillo
            </p>
            """,
            unsafe_allow_html=True
        )
    with col_logo:
        st.image(logo_unt, width=80)
except FileNotFoundError:
    st.caption("**Desarrollado por:** Tu Nombre Completo")
    st.caption("Escuela Profesional de Ingeniería de Sistemas")
    st.caption("Universidad Nacional de Trujillo")
    st.warning(f"⚠️ No se encontró el logo en la ruta: {PATH_LOGO_UNT}")

# Footer
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: #9CA3AF; font-size: 0.85em; padding: 20px 0;'>
        📊 Aplicación de Procesamiento de Datos en Machine Learning | 2024
    </div>
    """,
    unsafe_allow_html=True
)