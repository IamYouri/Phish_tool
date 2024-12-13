import re
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from urllib.parse import urlparse
import seaborn as sns
import matplotlib.pyplot as plt

# Function to extract URLs from email text
def extract_urls_from_email(email_text):
    url_pattern = r'(https?://[^\s]+)' # the ? is for allowing both http and https
    urls = re.findall(url_pattern, email_text)
    return urls

# Function to extract features from URLs
def extract_url_features(urls):
    url_features = []
    for url in urls:
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
        url_features.append(features)
    return pd.DataFrame(url_features)

# Function to clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'[^\w@.:\/\?=&\-_]+', ' ', text)  # Keep allowed special characters
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# Load the CEAS dataset
df = pd.read_csv('../kaggle_dataset/CEAS-08.csv')

# Combine subject and body into a single field
df['text'] = df['subject'].fillna('') + " " + df['body'].fillna('')

# Clean the combined text
df['cleaned_text'] = df['text'].apply(clean_text)

# Load the URL model and scaler
url_model = joblib.load('./models/xgboost_model.pkl')
scaler = joblib.load('./models/scaler_xgboost.pkl')

# Initialize variables for evaluation
true_labels = []  # True labels based on the CEAS dataset
predicted_labels = []  # Predictions from the URL model

# Loop through each email
for idx, email in df.iterrows():
    # Extract URLs from the email
    urls = extract_urls_from_email(email['cleaned_text'])

    if urls:
        # Extract features from URLs
        url_features = extract_url_features(urls)

        # Scale the features
        url_features_scaled = scaler.transform(url_features)

        # Predict with the URL model
        url_predictions = url_model.predict(url_features_scaled)

        # If any URL is flagged as phishing, consider the email as phishing
        if sum(url_predictions == 1) > 0:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)
    else:
        # If no URLs are found, consider it non-phishing
        predicted_labels.append(0)

    # Append the true label for this email
    true_labels.append(email['label'])

# Evaluate the URL model
print("Accuracy on CEAS Dataset (URL Model):", accuracy_score(true_labels, predicted_labels))
print("\nClassification Report (URL Model on CEAS Dataset):")
print(classification_report(true_labels, predicted_labels))

# Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)
print("\nConfusion Matrix:")
print(cm)

# Plot the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Phishing', 'Phishing'], yticklabels=['Non-Phishing', 'Phishing'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - URL Model on CEAS Dataset - XGboost')
plt.savefig('confusion_matrix_url_model_ceas_xgboost.png')
plt.show()
