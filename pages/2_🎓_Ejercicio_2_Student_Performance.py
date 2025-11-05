import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import *

# =========================
# CONFIGURACIÓN
# =========================
st.set_page_config(
    page_title="Ejercicio 2 - Student Performance",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Ejercicio 2: Procesamiento del Dataset Student Performance")
st.markdown("**Objetivo:** Procesar los datos para predecir la nota final (G3) de los estudiantes")

# =========================
# CARGA DE DATOS
# =========================
st.header("1️⃣ Carga del Dataset")

try:
    df = pd.read_csv('data/student-mat.csv')
    st.success(f"✅ Dataset cargado exitosamente: {df.shape[0]} estudiantes y {df.shape[1]} variables")
    
    with st.expander("👀 Ver primeras filas del dataset"):
        st.dataframe(df.head(10))
except FileNotFoundError:
    st.error("❌ No se encontró el archivo 'data/student-mat.csv'. Verifique la ruta.")
    st.stop()

# =========================
# EXPLORACIÓN INICIAL
# =========================
st.header("2️⃣ Exploración Inicial")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total de Estudiantes", df.shape[0])
with col2:
    st.metric("Variables", df.shape[1])
with col3:
    st.metric("Duplicados", df.duplicated().sum())
with col4:
    st.metric("Valores Nulos", df.isnull().sum().sum())

st.subheader("📊 Información del Dataset")

col1, col2 = st.columns(2)

with col1:
    st.write("**Variables Categóricas:**")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    st.write(", ".join(categorical_cols))
    
with col2:
    st.write("**Variables Numéricas:**")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    st.write(", ".join(numerical_cols))

st.subheader("📈 Estadísticas Descriptivas - Variables Numéricas")
st.dataframe(df[numerical_cols].describe())

st.subheader("📋 Información de Variables Categóricas")
with st.expander("Ver valores únicos de cada variable categórica"):
    for col in categorical_cols:
        st.write(f"**{col}:** {df[col].unique().tolist()}")

# =========================
# LIMPIEZA DE DATOS
# =========================
st.header("3️⃣ Limpieza de Datos")

st.subheader("🔍 Verificación de Duplicados e Inconsistencias")

# Eliminar duplicados si existen
duplicados_antes = df.duplicated().sum()
df_clean = df.drop_duplicates()
duplicados_despues = df_clean.duplicated().sum()

col1, col2 = st.columns(2)
with col1:
    st.metric("Duplicados encontrados", duplicados_antes)
with col2:
    st.metric("Duplicados eliminados", duplicados_antes - duplicados_despues)

# Verificar valores inconsistentes
st.write("**Verificación de rangos válidos:**")
inconsistencias = []

# Verificar que las notas estén en el rango correcto (0-20)
if df_clean[['G1', 'G2', 'G3']].min().min() < 0 or df_clean[['G1', 'G2', 'G3']].max().max() > 20:
    inconsistencias.append("Notas fuera del rango [0-20]")

# Verificar edad (debe ser entre 15-22)
if df_clean['age'].min() < 15 or df_clean['age'].max() > 25:
    inconsistencias.append("Edades inusuales detectadas")

if inconsistencias:
    st.warning(f"⚠️ Inconsistencias encontradas: {', '.join(inconsistencias)}")
else:
    st.success("✅ No se encontraron inconsistencias en los datos")

st.write(f"**Dimensiones después de limpieza:** {df_clean.shape}")

# =========================
# CODIFICACIÓN
# =========================
st.header("4️⃣ Codificación de Variables Categóricas (One-Hot Encoding)")

# Identificar columnas categóricas para One-Hot Encoding
categorical_to_encode = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 
                          'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 
                          'activities', 'nursery', 'higher', 'internet', 'romantic']

st.write(f"**Variables a codificar:** {len(categorical_to_encode)}")
st.write(", ".join(categorical_to_encode))

# Aplicar One-Hot Encoding
df_encoded = pd.get_dummies(df_clean, columns=categorical_to_encode, drop_first=False)

col1, col2 = st.columns(2)
with col1:
    st.metric("Columnas antes", df_clean.shape[1])
with col2:
    st.metric("Columnas después", df_encoded.shape[1])

with st.expander("Ver primeras filas después de la codificación"):
    st.dataframe(df_encoded.head())

# =========================
# NORMALIZACIÓN
# =========================
st.header("5️⃣ Normalización de Variables Numéricas")

# Variables numéricas a normalizar
numeric_to_normalize = ['age', 'absences', 'G1', 'G2']

st.write(f"**Variables a normalizar:** {', '.join(numeric_to_normalize)}")

# Aplicar normalización Min-Max
scaler = MinMaxScaler()
df_normalized = df_encoded.copy()
df_normalized[numeric_to_normalize] = scaler.fit_transform(df_encoded[numeric_to_normalize])

col1, col2 = st.columns(2)

with col1:
    st.write("**Antes de la normalización:**")
    st.dataframe(df_encoded[numeric_to_normalize].describe())

with col2:
    st.write("**Después de la normalización:**")
    st.dataframe(df_normalized[numeric_to_normalize].describe())

# =========================
# SEPARACIÓN X e y
# =========================
st.header("6️⃣ Separación de Características (X) y Variable Objetivo (y)")

X = df_normalized.drop('G3', axis=1)
y = df_normalized['G3']

col1, col2 = st.columns(2)
with col1:
    st.write("**X (Características):**")
    st.write(f"- Dimensiones: {X.shape}")
    st.write(f"- Número de características: {X.shape[1]}")
with col2:
    st.write("**y (Variable objetivo - G3):**")
    st.write(f"- Dimensiones: {y.shape}")
    st.write(f"- Rango: [{y.min():.2f}, {y.max():.2f}]")

# =========================
# DIVISIÓN DE DATOS
# =========================
st.header("7️⃣ División en Conjuntos de Entrenamiento y Prueba")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total de datos", len(df_normalized))
with col2:
    st.metric("Entrenamiento (80%)", len(X_train))
with col3:
    st.metric("Prueba (20%)", len(X_test))

st.write("**Dimensiones finales:**")
dimensiones_df = pd.DataFrame({
    'Conjunto': ['X_train', 'X_test', 'y_train', 'y_test'],
    'Dimensiones': [str(X_train.shape), str(X_test.shape), str(y_train.shape), str(y_test.shape)]
})
st.dataframe(dimensiones_df, use_container_width=True)

# =========================
# RETO ADICIONAL: CORRELACIÓN
# =========================
st.header("🎯 Reto Adicional: Análisis de Correlación entre G1, G2 y G3")

# Calcular correlaciones
correlation_matrix = df_clean[['G1', 'G2', 'G3']].corr()

col1, col2 = st.columns([1, 1])

with col1:
    st.write("**Matriz de Correlación:**")
    st.dataframe(correlation_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1))
    
    st.write("**Interpretación:**")
    st.write(f"- Correlación G1-G2: **{correlation_matrix.loc['G1', 'G2']:.3f}**")
    st.write(f"- Correlación G1-G3: **{correlation_matrix.loc['G1', 'G3']:.3f}**")
    st.write(f"- Correlación G2-G3: **{correlation_matrix.loc['G2', 'G3']:.3f}**")
    
    if correlation_matrix.loc['G2', 'G3'] > 0.8:
        st.success("✅ Existe una fuerte correlación positiva entre G2 y G3")

with col2:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Mapa de Calor - Correlación entre Notas', fontsize=14, fontweight='bold')
    st.pyplot(fig)

# Gráficos de dispersión
st.subheader("📊 Gráficos de Dispersión")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# G1 vs G3
axes[0].scatter(df_clean['G1'], df_clean['G3'], alpha=0.6, color='#3B82F6')
axes[0].set_xlabel('G1 (Primera Nota)')
axes[0].set_ylabel('G3 (Nota Final)')
axes[0].set_title('G1 vs G3')
axes[0].grid(True, alpha=0.3)

# G2 vs G3
axes[1].scatter(df_clean['G2'], df_clean['G3'], alpha=0.6, color='#10B981')
axes[1].set_xlabel('G2 (Segunda Nota)')
axes[1].set_ylabel('G3 (Nota Final)')
axes[1].set_title('G2 vs G3')
axes[1].grid(True, alpha=0.3)

# G1 vs G2
axes[2].scatter(df_clean['G1'], df_clean['G2'], alpha=0.6, color='#F59E0B')
axes[2].set_xlabel('G1 (Primera Nota)')
axes[2].set_ylabel('G2 (Segunda Nota)')
axes[2].set_title('G1 vs G2')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# =========================
# TABLA FINAL
# =========================
st.header("📋 Primeros 5 Registros Procesados")
st.dataframe(df_normalized.head(), use_container_width=True)

# =========================
# ESTADÍSTICAS FINALES
# =========================
st.header("📊 Resumen de Distribución de Notas")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**G1 (Primera Nota)**")
    st.write(f"Media: {df_clean['G1'].mean():.2f}")
    st.write(f"Mediana: {df_clean['G1'].median():.2f}")
    st.write(f"Desv. Std: {df_clean['G1'].std():.2f}")

with col2:
    st.write("**G2 (Segunda Nota)**")
    st.write(f"Media: {df_clean['G2'].mean():.2f}")
    st.write(f"Mediana: {df_clean['G2'].median():.2f}")
    st.write(f"Desv. Std: {df_clean['G2'].std():.2f}")

with col3:
    st.write("**G3 (Nota Final)**")
    st.write(f"Media: {df_clean['G3'].mean():.2f}")
    st.write(f"Mediana: {df_clean['G3'].median():.2f}")
    st.write(f"Desv. Std: {df_clean['G3'].std():.2f}")

# Histogramas
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df_clean['G1'], bins=20, color='#3B82F6', alpha=0.7, edgecolor='black')
axes[0].set_title('Distribución de G1')
axes[0].set_xlabel('Nota')
axes[0].set_ylabel('Frecuencia')

axes[1].hist(df_clean['G2'], bins=20, color='#10B981', alpha=0.7, edgecolor='black')
axes[1].set_title('Distribución de G2')
axes[1].set_xlabel('Nota')
axes[1].set_ylabel('Frecuencia')

axes[2].hist(df_clean['G3'], bins=20, color='#F59E0B', alpha=0.7, edgecolor='black')
axes[2].set_title('Distribución de G3')
axes[2].set_xlabel('Nota')
axes[2].set_ylabel('Frecuencia')

plt.tight_layout()
st.pyplot(fig)

# =========================
# DESCARGA
# =========================
st.header("💾 Descargar Datos Procesados")

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(df_normalized)
st.download_button(
    label="📥 Descargar CSV procesado",
    data=csv,
    file_name='student_performance_procesado.csv',
    mime='text/csv',
)

st.success("✅ Ejercicio 2 completado exitosamente")