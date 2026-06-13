import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

def normalize_data():
    """
    Normalise les features avec StandardScaler.
    Fit sur X_train, transform sur X_train et X_test.
    Sauvegarde le scaler pour une utilisation future.
    """
    print("Chargement des données splitées...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")

    print(f"X_train shape : {X_train.shape}")
    print(f"X_test shape  : {X_test.shape}")

    # Supprimer la colonne date si présente
    if "date" in X_train.columns:
        X_train = X_train.drop(columns=["date"])
        X_test = X_test.drop(columns=["date"])
        print("Colonne 'date' supprimée.")

    # Initialiser et fitter le scaler sur X_train uniquement
    scaler = StandardScaler()

    print("\nFit du StandardScaler sur X_train...")
    X_train_scaled = scaler.fit_transform(X_train)

    print("Transform de X_test...")
    X_test_scaled = scaler.transform(X_test)

    # Convertir en DataFrame en conservant les noms de colonnes
    X_train_scaled_df = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns
    )
    X_test_scaled_df = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns
    )

    # Sauvegarder les données normalisées
    X_train_scaled_df.to_csv("data/processed/X_train_scaled.csv", index=False)
    X_test_scaled_df.to_csv("data/processed/X_test_scaled.csv", index=False)

    # Sauvegarder le scaler
    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\nDonnées normalisées sauvegardées :")
    print("  - data/processed/X_train_scaled.csv")
    print("  - data/processed/X_test_scaled.csv")
    print("  - models/scaler.pkl")

    # Afficher quelques statistiques
    print(f"\nMoyenne X_train_scaled (doit être ~0) : {X_train_scaled_df.mean().mean():.6f}")
    print(f"Std X_train_scaled (doit être ~1)    : {X_train_scaled_df.std().mean():.6f}")

if __name__ == "__main__":
    normalize_data()
