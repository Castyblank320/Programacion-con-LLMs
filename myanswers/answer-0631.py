import numpy as np
from sklearn.linear_model import LinearRegression

def predecir_consumo_energia(X, y, X_new):
    """
    Entrena un modelo de regresión lineal y predice nuevos valores.
    """
    # 1. Crear y entrenar el modelo
    model = LinearRegression()
    model.fit(X, y)

    # 2. Realizar predicciones
    predictions = model.predict(X_new)

    # 3. Devolver el array de predicciones
    return predictions