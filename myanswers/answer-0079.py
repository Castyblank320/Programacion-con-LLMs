import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

def detectar_clics_fraudulentos(df, target_col, n_features):
    """
    Selecciona características y entrena un modelo Naive Bayes para detección de fraude.
    """
    # 1. Separar características (X) y objetivo (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Crear pipeline con los pasos requeridos
    pipeline = Pipeline([
        ('scaler', MaxAbsScaler()),
        ('selector', RFE(
            estimator=LogisticRegression(max_iter=1000, solver='liblinear'),
            n_features_to_select=n_features
        )),
        ('modelo', GaussianNB())
    ])

    # 3. Configurar validación cruzada repetida y estratificada
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    # 4. Calcular puntuaciones de balanced_accuracy
    scores = cross_val_score(
        pipeline, X, y,
        cv=rkf,
        scoring='balanced_accuracy'
    )

    # 5. Retornar el promedio de las puntuaciones
    return scores.mean()