import numpy as np
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import random

def generar_caso_de_uso_pipeline_pca_logistico():
    """
    Genera un caso de prueba aleatorio para la función pipeline_pca_logistico.
    Retorna (input_dict, output_expected)
    """
    # Generar datos aleatorios de clasificación
    n_samples = random.randint(100, 300)
    n_features = random.randint(10, 20)
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=n_features//2, random_state=42)
    
    # Decidir n_componentes: aleatorio entre 0.7 y 0.95 (float) o entero entre 3 y 8
    if random.random() < 0.5:
        n_componentes = random.uniform(0.7, 0.95)
    else:
        n_componentes = random.randint(3, min(8, n_features))
    
    # Construir pipeline y calcular output esperado
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_componentes)),
        ('lr', LogisticRegression(random_state=42))
    ])
    pipeline.fit(X, y)
    
    accuracy = pipeline.score(X, y)
    explained_variance_ratio = pipeline.named_steps['pca'].explained_variance_ratio_.tolist()
    
    output_expected = {
        'pipeline': pipeline,
        'accuracy': accuracy,
        'explained_variance_ratio': explained_variance_ratio
    }
    
    input_data = {
        'X': X,
        'y': y,
        'n_componentes': n_componentes
    }
    
    return input_data, output_expected

if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_pipeline_pca_logistico()
    print("=== INPUT ===")
    print(f"X shape: {entrada['X'].shape}")
    print(f"y shape: {entrada['y'].shape}")
    print(f"n_componentes (solicitado): {entrada['n_componentes']}")
    
    n_components_real = salida_esperada['pipeline'].named_steps['pca'].n_components_
    print(f"n_componentes real (retenidos): {n_components_real}")
    
    print("\n=== OUTPUT ESPERADO ===")
    print(f"accuracy: {salida_esperada['accuracy']}")
    evr = salida_esperada['explained_variance_ratio']
    print(f"explained_variance_ratio: {evr}")
    print(f"Suma de varianzas: {sum(evr):.6f}")
    print("pipeline entrenado: OK")