import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import pickle
import json
import os

def evaluate_model():
    """
    Évalue le modèle entraîné sur les données de test,
    sauvegarde les métriques et les prédictions.
    """
    print("Chargement des données de test...")
    X_test_scaled = pd.read_csv("data/processed/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    
    print(f"X_test_scaled shape : {X_test_scaled.shape}")
    print(f"y_test shape        : {y_test.shape}")
    
    # Charger le modèle entraîné
    print("\nChargement du modèle entraîné...")
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Faire les prédictions
    print("Génération des prédictions...")
    y_pred = model.predict(X_test_scaled)
    
    # Calculer les métriques
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Afficher les métriques
    print("\n" + "="*40)
    print("       MÉTRIQUES D'ÉVALUATION")
    print("="*40)
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")
    print("="*40)
    
    # Créer le dictionnaire des scores
    scores = {
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4)
    }
    
    # Sauvegarder les métriques en JSON
    os.makedirs("metrics", exist_ok=True)
    
    with open("metrics/scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    
    print("\nMétriques sauvegardées dans metrics/scores.json")
    
    # Sauvegarder les prédictions
    predictions_df = pd.DataFrame({
        "y_test": y_test.values,
        "y_pred": y_pred,
        "residuals": y_test.values - y_pred
    })
    
    os.makedirs("data/processed", exist_ok=True)
    predictions_df.to_csv("data/processed/predictions.csv", index=False)
    
    print("Prédictions sauvegardées dans data/processed/predictions.csv")

if __name__ == "__main__":
    evaluate_model()
