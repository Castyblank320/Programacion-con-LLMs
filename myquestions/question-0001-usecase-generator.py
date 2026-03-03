import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generar_caso_de_uso_analizar_cohorte_compras():
    """
    Genera un caso de prueba aleatorio para la función analizar_cohorte_compras.
    Retorna (input_dict, output_expected)
    """
    # Generar fechas aleatorias entre 2020-01-01 y 2022-12-31
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2022, 12, 31)
    days_range = (end_date - start_date).days

    n_rows = random.randint(20, 50)
    data = []
    clientes = [f'C{str(i).zfill(3)}' for i in range(random.randint(5, 15))]
    for _ in range(n_rows):
        cliente = random.choice(clientes)
        fecha = start_date + timedelta(days=random.randint(0, days_range))
        monto = round(random.uniform(10, 500), 2)
        data.append([cliente, fecha, monto])
    
    df = pd.DataFrame(data, columns=['cliente_id', 'fecha_compra', 'monto_compra'])
    
    # Calcular el output esperado
    df_copy = df.copy()
    df_copy['fecha_compra'] = pd.to_datetime(df_copy['fecha_compra'])
    df_copy['mes_compra'] = df_copy['fecha_compra'].dt.to_period('M').astype(str)
    
    # Primera compra por cliente
    primera_compra = df_copy.groupby('cliente_id')['mes_compra'].min().reset_index()
    primera_compra.columns = ['cliente_id', 'cohorte']
    
    # Unir para obtener cohorte de cada compra
    df_cohorte = df_copy.merge(primera_compra, on='cliente_id')
    # Calcular el número de meses desde la primera compra
    df_cohorte['mes_cohorte'] = (df_cohorte['fecha_compra'].dt.to_period('M') - 
                                  pd.PeriodIndex(df_cohorte['cohorte'], freq='M')).apply(lambda x: x.n)
    
    # Crear tabla de retención
    cohorte_counts = df_cohorte.groupby(['cohorte', 'mes_cohorte'])['cliente_id'].nunique().reset_index()
    # Pivotar
    pivot = cohorte_counts.pivot(index='cohorte', columns='mes_cohorte', values='cliente_id')
    # Normalizar por tamaño de cohorte (mes_cohorte 0)
    cohort_size = pivot[0]
    pivot_retention = pivot.divide(cohort_size, axis=0)
    # Renombrar columnas
    pivot_retention.columns = [f'mes_{int(col)}' for col in pivot_retention.columns]
    pivot_retention.index.name = 'cohorte'
    
    output_expected = pivot_retention
    
    input_data = {
        'df': df,
        'fecha_col': 'fecha_compra',
        'cliente_col': 'cliente_id',
        'venta_col': 'monto_compra'  # aunque no se usa, se incluye por consistencia
    }
    
    return input_data, output_expected

if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_analizar_cohorte_compras()
    print("=== INPUT (primeras filas del DataFrame) ===")
    print(entrada['df'].head())
    print("\n=== OUTPUT (tabla de retención) ===")
    print(salida)