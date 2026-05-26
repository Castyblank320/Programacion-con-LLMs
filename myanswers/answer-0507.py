import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

def evaluar_impacto_escalado_knn(X, y, n_neighbors, test_size, random_state):
    """
    Compara el rendimiento de KNN con y sin escalado de características.
    """
    # 1. División de datos estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # --- 2. Modelo SIN Escalado ---
    knn_raw = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_raw.fit(X_train, y_train)
    pred_raw = knn_raw.predict(X_test)

    acc_raw = accuracy_score(y_test, pred_raw)
    f1_raw = f1_score(y_test, pred_raw, average="weighted", zero_division=0)

    # --- 3. Modelo CON Escalado (StandardScaler) ---
    scaler = StandardScaler()
    # Ajustar el escalador SOLO con los datos de entrenamiento
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn_scaled = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_scaled.fit(X_train_scaled, y_train)
    pred_scaled = knn_scaled.predict(X_test_scaled)

    acc_scaled = accuracy_score(y_test, pred_scaled)
    f1_scaled = f1_score(y_test, pred_scaled, average="weighted", zero_division=0)

    # --- 4. Calcular las mejoras ---
    mejora_accuracy = acc_scaled - acc_raw
    mejora_f1 = f1_scaled - f1_raw

    # Crear diccionario de salida con valores redondeados a 6 decimales
    output_data = {
        "acc_sin_escalar": round(float(acc_raw), 6),
        "f1_sin_escalar": round(float(f1_raw), 6),
        "acc_con_escalar": round(float(acc_scaled), 6),
        "f1_con_escalar": round(float(f1_scaled), 6),
        "mejora_accuracy": round(float(mejora_accuracy), 6),
        "mejora_f1": round(float(mejora_f1), 6),
    }
    return output_data