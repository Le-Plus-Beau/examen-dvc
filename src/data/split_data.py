import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data():
    """
    Charge les données brutes, sépare features/target,
    puis split en train/test et sauvegarde les 4 datasets.
    """
    # Charger les données
    print("Chargement des données brutes...")
    df = pd.read_csv("data/raw/raw.csv")
    
    print(f"Shape du dataset : {df.shape}")
    print(f"Colonnes : {df.columns.tolist()}")
    
    # Séparation features / target
    # silica_concentrate est la dernière colonne
    X = df.drop(columns=["silica_concentrate"])
    y = df["silica_concentrate"]
    
    print(f"\nFeatures shape : {X.shape}")
    print(f"Target shape : {y.shape}")
    
    # Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    
    print(f"\nX_train shape : {X_train.shape}")
    print(f"X_test shape  : {X_test.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"y_test shape  : {y_test.shape}")
    
    # Créer le dossier processed si nécessaire
    os.makedirs("data/processed", exist_ok=True)
    
    # Sauvegarder les datasets
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    
    print("\nDatasets sauvegardés dans data/processed/")

if __name__ == "__main__":
    split_data()
