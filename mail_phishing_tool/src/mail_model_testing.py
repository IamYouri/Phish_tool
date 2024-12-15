import re
import pandas as pd
import joblib
from urllib.parse import urlparse
from matplotlib_venn import venn2
import matplotlib.pyplot as plt

# Function to clean the text
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'[^\w@.:\/\?=&\-_]+', ' ', text)  # Remove unwanted special characters
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    return text.strip()  # Remove leading/trailing spaces

# Function to extract URLs from an email
def extract_urls_from_email(email_text):
    url_pattern = r'(https?://[^\s]+)'  # Regex to extract URLs
    return re.findall(url_pattern, email_text)

# Function to extract features from URLs
def extract_url_features(urls):
    url_features = []
    for url in urls:
        features = {}
        parsed_url = urlparse(url)
        
        features['NumDots'] = url.count('.')
        features['NumDash'] = url.count('-')
        features['UrlLength'] = len(url)
        
        # Extract PathLevel: Count slashes in the path
        path_parts = parsed_url.path.strip('/').split('/')
        features['PathLevel'] = len(path_parts) if path_parts != [''] else 0
        
        features['NumQueryComponents'] = url.count('?')
        features['HostnameLength'] = len(parsed_url.hostname) if parsed_url.hostname else 0
        features['DoubleSlashInPath'] = url.count('//')
        features['AtSymbol'] = url.count('@')
        features['IpAddress'] = int(parsed_url.hostname.replace(".", "").isdigit()) if parsed_url.hostname else 0
        features['NoHttps'] = 1 if parsed_url.scheme != 'https' else 0
        
        url_features.append(features)
    
    return pd.DataFrame(url_features)

# Load CEAS-08 dataset
df = pd.read_csv('../kaggle_dataset/CEAS-08.csv')

# Combine subject and body to create a text column
df['text'] = df['subject'].fillna('') + " " + df['body'].fillna('')

# Clean the text data
df['cleaned_text'] = df['text'].apply(clean_text)

# Load the TF-IDF model and vectorizer
email_model = joblib.load("phishing_email_model_optimized.pkl")
vectorizer = joblib.load("tfidf_vectorizer_optimized.pkl")

# Transform the cleaned text using the TF-IDF vectorizer
email_features = vectorizer.transform(df['cleaned_text'])

# Predict phishing emails using the email model
predicted_labels = email_model.predict(email_features)

# plot the metrics 
from sklearn.metrics import classification_report
print(classification_report(df['label'], predicted_labels))

# plot the confusion matrix of the email model 
from sklearn.metrics import confusion_matrix
import seaborn as sns

conf_matrix = confusion_matrix(df['label'], predicted_labels)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix of Email Model')
plt.savefig('confusion_matrix_email_model_now.png')
plt.show()




# Load the URL model and scaler
url_model = joblib.load("./models/xgboost_model.pkl")
scaler = joblib.load("./models/scaler_xgboost.pkl")

# Analyze phishing emails
emails_flagged_as_phishing = df['cleaned_text'][predicted_labels == 1]
indices_flagged_as_phishing = df.index[predicted_labels == 1]

# Initialize counters
email_only_flagged_count = 0
url_flagged_email_count = 0
both_flagged_count = 0

# Process emails flagged by the email model
for email, idx in zip(emails_flagged_as_phishing, indices_flagged_as_phishing):
    urls = extract_urls_from_email(email)
    url_phishing_detected = False  # Track if URL model flags this email as phishing
    
    if urls:  # If URLs are present, check them
        url_features = extract_url_features(urls)
        url_features_scaled = scaler.transform(url_features)
        url_predictions = url_model.predict(url_features_scaled)
        
        # If at least one URL is flagged as phishing
        if sum(url_predictions == 1) > 0:
            url_flagged_email_count += 1
            url_phishing_detected = True
    
    # Update counters based on model agreement
    if url_phishing_detected:
        both_flagged_count += 1
    else:
        email_only_flagged_count += 1

# Recalculer correctement le nombre de "URL seulement"
url_only_flagged_count = url_flagged_email_count - both_flagged_count

# Tracer le diagramme de Venn corrigé
venn2(
    subsets=(url_only_flagged_count, email_only_flagged_count, both_flagged_count),
    set_labels=('Email Model', 'URL Model')
)
plt.title('Venn Diagram for Phishing Emails (Corrected)')
plt.savefig('venn_diagram_phishing_emails_corrected.png')
plt.show()

# Impression des valeurs pour vérification
print(f"Emails flagged as phishing by email model only: {email_only_flagged_count}")
print(f"Emails flagged as phishing by both models: {both_flagged_count}")
print(f"Emails flagged as phishing by URL model only: {url_only_flagged_count}")


