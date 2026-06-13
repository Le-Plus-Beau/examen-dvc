import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

def train_model():
    """
    Entraîne un RandomForestRegressor avec les meilleurs paramètres
    trouvés lors du GridSearch.
    """
    print("Chargement des données normalisées...")
    X_train_scaled = pd.read_csv("data/processed/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    
    print(f"X_train_scaled shape : {X_train_scaled.shape}")
    print(f"y_train shape        : {y_train.shape}")
    
    # Charger les meilleurs paramètres
    print("\nChargement des meilleurs paramètres...")
    with open("models/best_params.pkl", "rb") as f:
        best_params = pickle.load(f)
    
    print(f"Paramètres utilisés : {best_params}")
    
    # Créer et entraîner le modèle
    print("\nEntraînement du modèle RandomForestRegressor...")
    model = RandomForestRegressor(
        **best_params,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    print("Modèle entraîné avec succès !")
    
    # Afficher l'importance des features
    feature_importances = pd.Series(
        model.feature_importances_,
        index=X_train_scaled.columns
    ).sort_values(ascending=False)
    
    print("\nImportance des features :")
    print(feature_importances)
    
    # Sauvegarder le modèle entraîné
    os.makedirs("models", exist_ok=True)
    
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Sauvegarder l'importance des features
    feature_importances.to_csv(
        "models/feature_importances.csv",
        header=["importance"]
    )
    
    print("\nModèle sauvegardé :")
    print("  - models/model.pkl")
    print("  - models/feature_importances.csv")

if __name__ == "__main__":
    train_model()
