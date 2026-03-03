import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generar_caso_de_uso_detectar_outliers_produccion():
    """
    Genera un caso de prueba aleatorio para la función detectar_outliers_produccion.
    Retorna (input_dict, output_expected)
    """
    # Generar máquinas
    maquinas = [f'M{str(i).zfill(2)}' for i in range(random.randint(2, 5))]
    n_rows = random.randint(20, 40)
    
    data = []
    start_date = datetime(2023, 1, 1)
    days_range = 60
    for _ in range(n_rows):
        maquina = random.choice(maquinas)
        fecha = start_date + timedelta(days=random.randint(0, days_range))
        # Producción con algunos outliers
        if random.random() < 0.1:  # 10% outliers
            prod = random.uniform(500, 1000)  # rango anómalo
        else:
            prod = random.uniform(50, 200)    # rango normal
        data.append([maquina, fecha, prod])
    
    df = pd.DataFrame(data, columns=['maquina_id', 'fecha', 'produccion'])
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Seleccionar método aleatorio
    metodo = random.choice(['iqr', 'zscore'])
    
    # Calcular outliers esperados
    df_expected = df.copy()
    
    def marcar_outliers(grupo):
        if metodo == 'iqr':
            Q1 = grupo['produccion'].quantile(0.25)
            Q3 = grupo['produccion'].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            return (grupo['produccion'] < lower) | (grupo['produccion'] > upper)
        else:  # zscore
            media = grupo['produccion'].mean()
            std = grupo['produccion'].std()
            if std == 0:
                return np.zeros(len(grupo), dtype=bool)
            z = (grupo['produccion'] - media) / std
            return np.abs(z) > 3
    
    df_expected['outlier'] = df_expected.groupby('maquina_id').apply(marcar_outliers).reset_index(level=0, drop=True)
    
    input_data = {
        'df': df,
        'columna': 'produccion',
        'metodo': metodo
    }
    
    output_expected = df_expected
    
    return input_data, output_expected