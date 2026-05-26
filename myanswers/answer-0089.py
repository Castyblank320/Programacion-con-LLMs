import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def optimization(df, target_col):
    """
    Reduce la dimensionalidad con PCA manteniendo al menos el 95% de la varianza.
    
    Pasos:
    1. Separar características (X) y objetivo (y).
    2. Eliminar filas con valores NaN en las características.
    3. Escalar las características con StandardScaler.
    4. Calcular PCA para determinar el número óptimo de componentes (95% varianza).
    5. Entrenar el PCA final y transformar los datos.
    6. Retornar la tupla (pca, X_transformada, n_componentes).
    """
    # 1. Separar características (X) y objetivo (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Manejar valores faltantes: eliminar filas con NaN en las características
    #    Nota: Esto es necesario porque PCA no puede procesar datos con NaN.
    #    Es la misma estrategia que usa el generador de casos de uso.
    feature_cols = X.columns.tolist()
    df_clean = df.dropna(subset=feature_cols).copy()
    
    # Verificar que haya suficientes datos después de la limpieza
    if len(df_clean) == 0:
        raise ValueError("No quedan datos después de eliminar filas con NaN")
    
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values

    # 3. Escalar las características con StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Calcular PCA para determinar el número óptimo de componentes
    #    (mantener al menos 95% de varianza explicada)
    pca_full = PCA()
    pca_full.fit(X_scaled)
    varianza_acumulada = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Encontrar el mínimo número de componentes que alcanza o supera el 95%
    n_components = np.searchsorted(varianza_acumulada, 0.95) + 1
    # Asegurar que n_components no supere el número de features disponibles
    n_components = min(n_components, X_scaled.shape[1])

    # 5. Entrenar el PCA final y transformar los datos
    pca = PCA(n_components=n_components)
    X_transformada = pca.fit_transform(X_scaled)

    # 6. Retornar la tupla solicitada
    return (pca, X_transformada, n_components)