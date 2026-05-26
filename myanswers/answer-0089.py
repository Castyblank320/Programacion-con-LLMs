import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def optimization(df, target_col):
    """
    Reduce la dimensionalidad con PCA manteniendo al menos el 95% de la varianza.
    """
    # 1. Separar características (X) y objetivo (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Escalar las características con StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Calcular PCA para determinar el número óptimo de componentes
    pca_full = PCA()
    pca_full.fit(X_scaled)

    # Obtener la varianza explicada acumulada
    varianza_acumulada = np.cumsum(pca_full.explained_variance_ratio_)

    # Encontrar el número mínimo de componentes para alcanzar al menos el 95% de varianza
    n_components = np.searchsorted(varianza_acumulada, 0.95) + 1

    # 4. Entrenar el PCA final y transformar los datos
    pca = PCA(n_components=n_components)
    X_transformada = pca.fit_transform(X_scaled)

    # 5. Devolver la tupla solicitada
    return (pca, X_transformada, n_components)