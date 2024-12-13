import re
from email import policy
from email.parser import BytesParser

# Fonction pour extraire les URLs d'un email
def extract_urls_from_email(email_text):
    # Regex pour capturer les URLs dans le texte
    url_pattern = r'(https?://[^\s]+)'
    urls = re.findall(url_pattern, email_text)
    return urls

# Exemple d'utilisation
email_text = "Click on this link to get a prize: https://fakephishinglink.com and don't forget to visit http://anotherlink.com"
urls = extract_urls_from_email(email_text)
print("Extracted URLs:", urls)

from urllib.parse import urlparse

def extract_url_features(urls):
    url_features = []

    for url in urls:
        features = {}

        # Extraire certaines features simples
        parsed_url = urlparse(url)
        features['UrlLength'] = len(url)
        features['NumDots'] = url.count('.')
        features['NumDash'] = url.count('-')
        features['NumQueryComponents'] = url.count('?')
        features['NumHash'] = url.count('#')

        # Ajouter d'autres features spécifiques à votre modèle si nécessaire
        # Comme par exemple vérifier si le domaine est un IP ou un nom de domaine
        features['HasIP'] = int(parsed_url.netloc.replace(".", "").isdigit())

        url_features.append(features)
    
    return url_features

# Exemple d'utilisation
urls = ["https://fakephishinglink.com?query=test", "http://anotherlink.com/path"]
features = extract_url_features(urls)
print("URL Features:", features)
