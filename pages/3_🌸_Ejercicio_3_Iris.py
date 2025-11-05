import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
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
    page_title="Ejercicio 3 - Iris",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Ejercicio 3: Dataset Iris")
st.markdown("**Objetivo:** Implementar un flujo completo de preprocesamiento y visualizar resultados")

# =========================
# CARGA DE DATOS
# =========================
st.header("1️⃣ Carga del Dataset desde sklearn.datasets")

# Cargar dataset
iris = load_iris()

st.success("✅ Dataset Iris cargado exitosamente desde sklearn.datasets")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Muestras", iris.data.shape[0])
with col2:
    st.metric("Características", iris.data.shape[1])
with col3:
    st.metric("Clases", len(iris.target_names))

st.write("**Nombres de las clases:**", ', '.join(iris.target_names))
st.write("**Nombres de las características:**", ', '.join(iris.feature_names))

# =========================
# CONVERSIÓN A DATAFRAME
# =========================
st.header("2️⃣ Conversión a DataFrame con Nombres de Columnas")

# Crear DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: iris.target_names[0], 
                                   1: iris.target_names[1], 
                                   2: iris.target_names[2]})

st.write("**Estructura del DataFrame:**")
st.dataframe(df.head(10))

st.write("**Información del DataFrame:**")
col1, col2 = st.columns(2)
with col1:
    st.write("**Tipos de datos:**")
    tipos_df = pd.DataFrame({
        'Columna': df.dtypes.index,
        'Tipo': df.dtypes.values
    })
    st.dataframe(tipos_df)

with col2:
    st.write("**Distribución de clases:**")
    dist_clases = df['species'].value_counts()
    st.dataframe(dist_clases)

# =========================
# ESTADÍSTICAS DESCRIPTIVAS
# =========================
st.header("📊 Estadísticas Descriptivas")

st.dataframe(df.describe())

# Estadísticas por clase
st.subheader("📈 Estadísticas por Especie")

especie_seleccionada = st.selectbox(
    "Seleccione una especie:",
    options=iris.target_names
)

df_especie = df[df['species'] == especie_seleccionada]
st.dataframe(df_especie.describe())

# =========================
# ESTANDARIZACIÓN
# =========================
st.header("3️⃣ Estandarización con StandardScaler")

# Separar características y target
X = df.drop(['target', 'species'], axis=1)
y = df['target']

st.write("**Datos antes de la estandarización:**")
col1, col2 = st.columns(2)
with col1:
    st.write("Primeras 5 filas:")
    st.dataframe(X.head())
with col2:
    st.write("Estadísticas:")
    st.dataframe(X.describe())

# Aplicar estandarización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

st.write("**Datos después de la estandarización:**")
col1, col2 = st.columns(2)
with col1:
    st.write("Primeras 5 filas:")
    st.dataframe(X_scaled_df.head())
with col2:
    st.write("Estadísticas:")
    st.dataframe(X_scaled_df.describe())

# Comparación visual
st.subheader("📊 Comparación: Antes vs Después de Estandarización")

feature_to_compare = st.selectbox(
    "Seleccione una característica para comparar:",
    options=X.columns
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Antes
axes[0].hist(X[feature_to_compare], bins=20, color='#3B82F6', alpha=0.7, edgecolor='black')
axes[0].set_title(f'{feature_to_compare} - Antes de Estandarización')
axes[0].set_xlabel('Valor')
axes[0].set_ylabel('Frecuencia')
axes[0].axvline(X[feature_to_compare].mean(), color='red', linestyle='--', label=f'Media: {X[feature_to_compare].mean():.2f}')
axes[0].legend()

# Después
axes[1].hist(X_scaled_df[feature_to_compare], bins=20, color='#10B981', alpha=0.7, edgecolor='black')
axes[1].set_title(f'{feature_to_compare} - Después de Estandarización')
axes[1].set_xlabel('Valor')
axes[1].set_ylabel('Frecuencia')
axes[1].axvline(X_scaled_df[feature_to_compare].mean(), color='red', linestyle='--', label=f'Media: {X_scaled_df[feature_to_compare].mean():.2f}')
axes[1].legend()

plt.tight_layout()
st.pyplot(fig)

# =========================
# DIVISIÓN DE DATOS
# =========================
st.header("4️⃣ División del Dataset (70% Entrenamiento, 30% Prueba)")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.3, random_state=42, stratify=y
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total de datos", len(X_scaled_df))
with col2:
    st.metric("Entrenamiento (70%)", len(X_train))
with col3:
    st.metric("Prueba (30%)", len(X_test))

st.write("**Dimensiones finales:**")
dimensiones_df = pd.DataFrame({
    'Conjunto': ['X_train', 'X_test', 'y_train', 'y_test'],
    'Dimensiones': [str(X_train.shape), str(X_test.shape), str(y_train.shape), str(y_test.shape)]
})
st.dataframe(dimensiones_df, use_container_width=True)

# Verificar distribución de clases
st.subheader("📊 Distribución de Clases en los Conjuntos")

col1, col2 = st.columns(2)

with col1:
    st.write("**Conjunto de Entrenamiento:**")
    train_dist = pd.Series(y_train).value_counts().sort_index()
    train_dist.index = [iris.target_names[i] for i in train_dist.index]
    st.dataframe(train_dist)

with col2:
    st.write("**Conjunto de Prueba:**")
    test_dist = pd.Series(y_test).value_counts().sort_index()
    test_dist.index = [iris.target_names[i] for i in test_dist.index]
    st.dataframe(test_dist)

# =========================
# VISUALIZACIÓN
# =========================
st.header("5️⃣ Visualización: Gráfico de Dispersión")

st.subheader("📊 Sepal Length vs Petal Length (Diferenciado por Clase)")

# Preparar datos para visualización
df_viz = X_scaled_df.copy()
df_viz['species'] = df['species']
df_viz['target'] = y

# Gráfico interactivo
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#EF4444', '#10B981', '#3B82F6']
for i, species in enumerate(iris.target_names):
    mask = df_viz['species'] == species
    ax.scatter(
        df_viz[mask]['sepal length (cm)'],
        df_viz[mask]['petal length (cm)'],
        c=colors[i],
        label=species,
        alpha=0.7,
        s=100,
        edgecolors='black',
        linewidth=0.5
    )

ax.set_xlabel('Sepal Length (estandarizado)', fontsize=12, fontweight='bold')
ax.set_ylabel('Petal Length (estandarizado)', fontsize=12, fontweight='bold')
ax.set_title('Distribución de Iris: Sepal Length vs Petal Length', fontsize=14, fontweight='bold')
ax.legend(title='Especie', fontsize=10)
ax.grid(True, alpha=0.3)

st.pyplot(fig)

# Opciones adicionales de visualización
st.subheader("🎨 Visualización Personalizada")

col1, col2 = st.columns(2)
with col1:
    feature_x = st.selectbox(
        "Seleccione característica para eje X:",
        options=X.columns,
        index=0
    )
with col2:
    feature_y = st.selectbox(
        "Seleccione característica para eje Y:",
        options=X.columns,
        index=2
    )

fig, ax = plt.subplots(figsize=(10, 6))

for i, species in enumerate(iris.target_names):
    mask = df_viz['species'] == species
    ax.scatter(
        df_viz[mask][feature_x],
        df_viz[mask][feature_y],
        c=colors[i],
        label=species,
        alpha=0.7,
        s=100,
        edgecolors='black',
        linewidth=0.5
    )

ax.set_xlabel(f'{feature_x} (estandarizado)', fontsize=12, fontweight='bold')
ax.set_ylabel(f'{feature_y} (estandarizado)', fontsize=12, fontweight='bold')
ax.set_title(f'{feature_x} vs {feature_y}', fontsize=14, fontweight='bold')
ax.legend(title='Especie', fontsize=10)
ax.grid(True, alpha=0.3)

st.pyplot(fig)

# =========================
# MATRIZ DE CORRELACIÓN
# =========================
st.header("📊 Análisis de Correlación entre Características")

correlation_matrix = X_scaled_df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Matriz de Correlación - Dataset Iris', fontsize=14, fontweight='bold')
st.pyplot(fig)

# =========================
# PAIRPLOT
# =========================
st.header("📈 Pairplot - Relaciones entre todas las características")

with st.spinner("Generando pairplot..."):
    fig = plt.figure(figsize=(12, 10))
    
    # Crear un pairplot manual
    features = X_scaled_df.columns
    n_features = len(features)
    
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            ax = plt.subplot(n_features, n_features, i * n_features + j + 1)
            
            if i == j:
                # Diagonal: histogramas
                for k, species in enumerate(iris.target_names):
                    mask = df_viz['species'] == species
                    ax.hist(df_viz[mask][feat1], alpha=0.5, color=colors[k], bins=15)
                ax.set_ylabel('')
            else:
                # Fuera de diagonal: scatter plots
                for k, species in enumerate(iris.target_names):
                    mask = df_viz['species'] == species
                    ax.scatter(df_viz[mask][feat2], df_viz[mask][feat1], 
                             c=colors[k], alpha=0.5, s=20)
            
            if i == n_features - 1:
                ax.set_xlabel(feat2.split()[0], fontsize=8)
            else:
                ax.set_xlabel('')
                
            if j == 0:
                ax.set_ylabel(feat1.split()[0], fontsize=8)
            else:
                ax.set_ylabel('')
                
            ax.tick_params(labelsize=6)
    
    plt.tight_layout()
    st.pyplot(fig)

# =========================
# ESTADÍSTICAS FINALES
# =========================
st.header("📋 Estadísticas Descriptivas del Dataset Estandarizado")

st.dataframe(X_scaled_df.describe(), use_container_width=True)

# =========================
# DESCARGA
# =========================
st.header("💾 Descargar Datos Procesados")

# Crear DataFrame completo para descarga
df_download = X_scaled_df.copy()
df_download['target'] = y
df_download['species'] = df['species']

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(df_download)
st.download_button(
    label="📥 Descargar CSV procesado",
    data=csv,
    file_name='iris_procesado.csv',
    mime='text/csv',
)

st.success("✅ Ejercicio 3 completado exitosamente")

# =========================
# RESUMEN FINAL
# =========================
st.header("📝 Resumen del Procesamiento")

resumen = pd.DataFrame({
    'Etapa': [
        'Dataset Original',
        'Después de Estandarización',
        'Conjunto de Entrenamiento',
        'Conjunto de Prueba'
    ],
    'Filas': [
        len(df),
        len(X_scaled_df),
        len(X_train),
        len(X_test)
    ],
    'Columnas': [
        df.shape[1],
        X_scaled_df.shape[1] + 1,  # +1 por target
        X_train.shape[1],
        X_test.shape[1]
    ]
})

st.dataframe(resumen, use_container_width=True)