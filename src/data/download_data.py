import requests
import os

def download_data():
    """Télécharge les données brutes depuis S3."""
    url = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
    
    # Créer le dossier si nécessaire
    os.makedirs("data/raw", exist_ok=True)
    
    output_path = "data/raw/raw.csv"
    
    print(f"Téléchargement des données depuis {url}...")
    response = requests.get(url)
    response.raise_for_status()
    
    with open(output_path, "wb") as f:
        f.write(response.content)
    
    print(f"Données sauvegardées dans {output_path}")
    print(f"Taille du fichier : {os.path.getsize(output_path)} bytes")

if __name__ == "__main__":
    download_data()
