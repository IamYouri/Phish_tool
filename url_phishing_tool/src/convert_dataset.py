import pandas as pd
from urllib.parse import urlparse

# Charger la nouvelle dataset
data = pd.read_csv('../phishing_dataset/phishing_urls.csv')  # Remplacez par le chemin vers votre nouvelle dataset

# Fonction pour extraire les features d'une URL
def extract_url_features(url):
    features = {}
    parsed_url = urlparse(url)
    features['NumDots'] = url.count('.')
    features['NumDash'] = url.count('-')
    features['UrlLength'] = len(url)
    path_parts = parsed_url.path.strip('/').split('/')
    features['PathLevel'] = len(path_parts) if path_parts != [''] else 0
    features['NumQueryComponents'] = url.count('?')
    features['HostnameLength'] = len(parsed_url.hostname) if parsed_url.hostname else 0
    features['DoubleSlashInPath'] = url.count('//')
    features['AtSymbol'] = url.count('@')
    features['IpAddress'] = int(parsed_url.hostname.replace(".", "").isdigit()) if parsed_url.hostname else 0
    features['NoHttps'] = 1 if parsed_url.scheme != 'https' else 0
    return features

# Extraire les features de toutes les URLs
url_features = []
for idx, row in data.iterrows():
    url = row['URL']
    label = row['Label']  # Phishing ou Non-Phishing
    features = extract_url_features(url)
    features['CLASS_LABEL'] = 1 if label.lower() == 'bad' else 0  # Convertir les labels
    url_features.append(features)

# Créer un DataFrame à partir des features
transformed_data = pd.DataFrame(url_features)

# Sauvegarder les données transformées
transformed_data.to_csv('transformed_phishing_urls.csv', index=False)
print("Dataset transformé sauvegardé dans 'transformed_large_url_dataset.csv'")
