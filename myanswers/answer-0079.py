import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def detectar_clics_fraudulentos(df, target_col, n_features):
    """
    Selecciona características y entrena un modelo Naive Bayes para detección de fraude.
    
    Parámetros:
    df : pd.DataFrame
        DataFrame con características y columna objetivo.
    target_col : str
        Nombre de la columna objetivo.
    n_features : int
        Número de características a seleccionar con RFE.
    
    Retorna:
    float
        Precisión balanceada promedio (cross-validation).
    """
    # 1. Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Imputar valores faltantes (por si acaso) con la media
    #    Esto no debería ser necesario si el generador no genera NaN,
    #    pero lo incluimos por robustez.
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # 3. Crear pipeline
    pipeline = Pipeline([
        ('scaler', MaxAbsScaler()),
        ('selector', RFE(
            estimator=LogisticRegression(max_iter=1000, solver='liblinear'),
            n_features_to_select=n_features
        )),
        ('modelo', GaussianNB())
    ])
    
    # 4. Validación cruzada repetida estratificada
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    
    # 5. Calcular balanced_accuracy
    scores = cross_val_score(
        pipeline, X_imputed, y,
        cv=rkf,
        scoring='balanced_accuracy'
    )
    
    # 6. Retornar el promedio
    return scores.mean()