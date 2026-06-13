import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle
import os
import json

def run_grid_search():
    """
    Effectue une recherche par grille pour trouver les meilleurs
    hyperparamètres pour un RandomForestRegressor.
    """
    print("Chargement des données normalisées...")
    X_train_scaled = pd.read_csv("data/processed/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    
    print(f"X_train_scaled shape : {X_train_scaled.shape}")
    print(f"y_train shape        : {y_train.shape}")
    
    # Définir le modèle de base
    rf = RandomForestRegressor(random_state=42)
    
    # Définir la grille de paramètres
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"]
    }
    
    print("\nGrille de paramètres :")
    print(json.dumps(
        {k: str(v) for k, v in param_grid.items()},
        indent=2
    ))
    
    # GridSearchCV avec cross-validation 5 folds
    print("\nDémarrage du GridSearch (cv=5)...")
    print("Cela peut prendre quelques minutes...")
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    # Récupérer les meilleurs paramètres
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Inverser le signe (neg_mse)
    
    print(f"\nMeilleurs paramètres trouvés :")
    print(json.dumps(best_params, indent=2))
    print(f"Meilleur MSE (cross-val) : {best_score:.4f}")
    
    # Sauvegarder les meilleurs paramètres
    os.makedirs("models", exist_ok=True)
    
    with open("models/best_params.pkl", "wb") as f:
        pickle.dump(best_params, f)
    
    # Sauvegarder aussi en JSON pour lisibilité
    with open("models/best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    
    print("\nMeilleurs paramètres sauvegardés :")
    print("  - models/best_params.pkl")
    print("  - models/best_params.json")

if __name__ == "__main__":
    run_grid_search()
