import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import random

def generar_caso_de_uso_optimizar_random_forest():
    """
    Genera un caso de prueba aleatorio para la función optimizar_random_forest.
    Retorna (input_dict, output_expected)
    """
    # Generar datos aleatorios de clasificación
    n_samples = random.randint(100, 300)
    n_features = random.randint(5, 10)
    n_informative = random.randint(2, n_features-1)
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=n_informative, n_redundant=0,
                               random_state=42)
    
    # Generar param_grid aleatorio
    param_grid = {
        'n_estimators': random.choice([[50, 100], [100, 200], [50, 150]]),
        'max_depth': random.choice([None, [3,5,7], [5,10]]),
        'min_samples_split': random.choice([[2,5], [2,3,4]])
    }
    cv = random.randint(3, 5)
    
    # Calcular el output esperado
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', return_train_score=False)
    grid_search.fit(X, y)
    
    output_expected = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_
    }
    
    input_data = {
        'X': X,
        'y': y,
        'param_grid': param_grid,
        'cv': cv
    }
    
    return input_data, output_expected

if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_optimizar_random_forest()
    print("=== INPUT ===")
    print(f"X shape: {entrada['X'].shape}")
    print(f"y shape: {entrada['y'].shape}")
    print(f"param_grid: {entrada['param_grid']}")
    print(f"cv: {entrada['cv']}")
    print("\n=== OUTPUT ESPERADO ===")
    print(f"best_params: {salida_esperada['best_params']}")
    print(f"best_score: {salida_esperada['best_score']}")
    print(f"best_estimator: {salida_esperada['best_estimator']}")