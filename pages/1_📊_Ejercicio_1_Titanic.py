import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    page_title="Ejercicio 1 - Titanic",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Ejercicio 1: Análisis del Dataset Titanic")
st.markdown("**Objetivo:** Preparar los datos para predecir la supervivencia de los pasajeros")

# =========================
# CARGA DE DATOS
# =========================
st.header("1️⃣ Carga del Dataset")

try:
    df = pd.read_csv('data/Titanic-Dataset.csv')
    st.success(f"✅ Dataset cargado exitosamente: {df.shape[0]} filas y {df.shape[1]} columnas")
    
    with st.expander("👀 Ver primeras filas del dataset"):
        st.dataframe(df.head(10))
except FileNotFoundError:
    st.error("❌ No se encontró el archivo 'data/titanic.csv'. Verifique la ruta.")
    st.stop()

# =========================
# EXPLORACIÓN INICIAL
# =========================
st.header("2️⃣ Exploración Inicial")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total de Pasajeros", df.shape[0])
with col2:
    st.metric("Variables", df.shape[1])
with col3:
    st.metric("Duplicados", df.duplicated().sum())

st.subheader("📊 Información del Dataset")
col1, col2 = st.columns(2)

with col1:
    st.write("**Tipos de datos:**")
    st.dataframe(pd.DataFrame({
        'Columna': df.dtypes.index,
        'Tipo': df.dtypes.values
    }))

with col2:
    st.write("**Valores nulos:**")
    missing = df.isnull().sum()
    missing_df = pd.DataFrame({
        'Columna': missing.index,
        'Valores Nulos': missing.values,
        'Porcentaje': (missing.values / len(df) * 100).round(2)
    })
    st.dataframe(missing_df[missing_df['Valores Nulos'] > 0])

st.subheader("📈 Estadísticas Descriptivas")
st.dataframe(df.describe())

# =========================
# LIMPIEZA DE DATOS
# =========================
st.header("3️⃣ Limpieza de Datos")

st.subheader("🗑️ Eliminación de Columnas Irrelevantes")
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df_clean = df.drop(columns=columns_to_drop)
st.write(f"Columnas eliminadas: {', '.join(columns_to_drop)}")
st.write(f"Nuevas dimensiones: {df_clean.shape}")

st.subheader("🔧 Tratamiento de Valores Nulos")

# Age: rellenar con la media
df_clean['Age'].fillna(df_clean['Age'].mean(), inplace=True)

# Embarked: rellenar con la moda
df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)

# Fare: rellenar con la mediana
df_clean['Fare'].fillna(df_clean['Fare'].median(), inplace=True)

st.write("✅ Valores nulos restantes:", df_clean.isnull().sum().sum())

with st.expander("Ver datos después de la limpieza"):
    st.dataframe(df_clean.head())

# =========================
# CODIFICACIÓN
# =========================
st.header("4️⃣ Codificación de Variables Categóricas")

df_encoded = df_clean.copy()

# Codificar Sex
le_sex = LabelEncoder()
df_encoded['Sex'] = le_sex.fit_transform(df_encoded['Sex'])
st.write(f"**Sex:** {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")

# Codificar Embarked
le_embarked = LabelEncoder()
df_encoded['Embarked'] = le_embarked.fit_transform(df_encoded['Embarked'])
st.write(f"**Embarked:** {dict(zip(le_embarked.classes_, le_embarked.transform(le_embarked.classes_)))}")

with st.expander("Ver datos codificados"):
    st.dataframe(df_encoded.head(10))

# =========================
# NORMALIZACIÓN
# =========================
st.header("5️⃣ Normalización/Estandarización")

numeric_features = ['Age', 'Fare']
scaler = StandardScaler()

df_normalized = df_encoded.copy()
df_normalized[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

st.write("Variables estandarizadas:", ', '.join(numeric_features))

col1, col2 = st.columns(2)
with col1:
    st.write("**Antes de la estandarización:**")
    st.dataframe(df_encoded[numeric_features].describe())
with col2:
    st.write("**Después de la estandarización:**")
    st.dataframe(df_normalized[numeric_features].describe())

# =========================
# DIVISIÓN DE DATOS
# =========================
st.header("6️⃣ División en Conjuntos de Entrenamiento y Prueba")

X = df_normalized.drop('Survived', axis=1)
y = df_normalized['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total de datos", len(df_normalized))
with col2:
    st.metric("Entrenamiento (70%)", len(X_train))
with col3:
    st.metric("Prueba (30%)", len(X_test))

st.write("**Dimensiones finales:**")
st.write(f"- X_train: {X_train.shape}")
st.write(f"- X_test: {X_test.shape}")
st.write(f"- y_train: {y_train.shape}")
st.write(f"- y_test: {y_test.shape}")

# =========================
# TABLA FINAL
# =========================
st.header("📋 Tabla con Primeros 5 Registros Procesados")

resultado_final = df_normalized.head()
st.dataframe(resultado_final, use_container_width=True)

# =========================
# VISUALIZACIÓN
# =========================
st.header("📊 Visualizaciones")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8, 6))
    df_clean['Survived'].value_counts().plot(kind='bar', ax=ax, color=['#EF4444', '#10B981'])
    ax.set_title('Distribución de Supervivencia')
    ax.set_xlabel('Sobrevivió (0=No, 1=Sí)')
    ax.set_ylabel('Cantidad')
    plt.xticks(rotation=0)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8, 6))
    df_clean.groupby(['Pclass', 'Survived']).size().unstack().plot(kind='bar', ax=ax, stacked=True, color=['#EF4444', '#10B981'])
    ax.set_title('Supervivencia por Clase')
    ax.set_xlabel('Clase')
    ax.set_ylabel('Cantidad')
    ax.legend(['No Sobrevivió', 'Sobrevivió'])
    plt.xticks(rotation=0)
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
    file_name='titanic_procesado.csv',
    mime='text/csv',
)

st.success("✅ Ejercicio 1 completado exitosamente")