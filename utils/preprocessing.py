import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def load_titanic_data(filepath):
    """Carga el dataset de Titanic"""
    return pd.read_csv(filepath)

def load_student_data(filepath):
    """Carga el dataset de Student Performance"""
    return pd.read_csv(filepath)

def get_basic_info(df):
    """Retorna información básica del dataset"""
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes,
        'missing': df.isnull().sum(),
        'duplicates': df.duplicated().sum()
    }
    return info

def handle_missing_values(df, numeric_strategy='mean', categorical_strategy='mode'):
    """Maneja valores nulos"""
    df_clean = df.copy()
    
    for column in df_clean.columns:
        if df_clean[column].isnull().sum() > 0:
            if df_clean[column].dtype in ['int64', 'float64']:
                if numeric_strategy == 'mean':
                    df_clean[column].fillna(df_clean[column].mean(), inplace=True)
                elif numeric_strategy == 'median':
                    df_clean[column].fillna(df_clean[column].median(), inplace=True)
            else:
                if categorical_strategy == 'mode':
                    df_clean[column].fillna(df_clean[column].mode()[0], inplace=True)
    
    return df_clean

def encode_categorical(df, columns, method='label'):
    """Codifica variables categóricas"""
    df_encoded = df.copy()
    
    if method == 'label':
        le = LabelEncoder()
        for col in columns:
            if col in df_encoded.columns:
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    elif method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=False)
    
    return df_encoded

def normalize_features(df, columns, method='standard'):
    """Normaliza características numéricas"""
    df_normalized = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    
    df_normalized[columns] = scaler.fit_transform(df_normalized[columns])
    
    return df_normalized, scaler

def split_data(X, y, test_size=0.3, random_state=42):
    """Divide los datos en entrenamiento y prueba"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)