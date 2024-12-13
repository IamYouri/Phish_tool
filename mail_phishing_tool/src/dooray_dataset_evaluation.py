import base64
import os
from email import policy
from email.parser import BytesParser
import joblib
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import re

# This file is used to parse the emails and extract the subject and body of the email
# It then uses the trained model to predict if the email is phishing or not
# The results are saved in a log file and printed to the console
# The number of phishing and non-phishing emails are also counted and displayed

# parse eml files and extract subject and body of the email
def parse_eml_files(eml_folder):
    emails = []
    for file_name in os.listdir(eml_folder):
        if file_name.endswith('.eml'):
            with open(os.path.join(eml_folder, file_name), 'rb') as f:
                msg = BytesParser(policy=policy.default).parse(f)
                subject = msg['subject'] or ''
                body = None
                # Extract the body
                if msg.is_multipart():
                    # Iterate through parts in case of multipart emails
                    for part in msg.iter_parts():
                        content_type = part.get_content_type()
                        if content_type == 'text/plain':
                            body = part.get_payload(decode=True)  # Automatically decode Base64
                            if body:
                                body = body.decode('utf-8', errors='ignore')
                                break
                else:
                    # Handle single-part email
                    body = msg.get_payload(decode=True)  # Automatically decode Base64
                    if body:
                        body = body.decode('utf-8', errors='ignore')
                # Add parsed email to list
                emails.append({'subject': subject, 'body': body or ''})
    return emails

# Specify your directory
eml_folder = "../Mails"  
parsed_emails = parse_eml_files(eml_folder)

# Combine subject and body
def prepare_email_data(parsed_emails):
    combined_emails = [email['subject'] + " " + email['body'] for email in parsed_emails]
    return combined_emails

email_texts = prepare_email_data(parsed_emails)

def clean_text(text):
    text = text.lower()  # Mettre en minuscule
    #non ascii characters are removed for ex. corean and chinese characters are removed
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # supprimer les caractères non-alphanumériques à l'excpetion de @ . : // ? = & - _
    text = re.sub(r'[^\w@.:\/\?=&\-_]+', ' ', text) 
    text = re.sub(r'\s+', ' ', text)  # Supprimer les espaces multiples
    text = text.strip()  # Supprimer les espaces en début/fin
    return text

clean_email_texts = [clean_text(text) for text in email_texts]

#print clean_email_texts
for i in range(len(clean_email_texts)):
    print(clean_email_texts[i])
    print("-" * 50)

# Load the trained model and vectorizer
loaded_model = joblib.load("phishing_email_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Transform using the TF-IDF vectorizer (from training)
email_features = tfidf_vectorizer.transform(clean_email_texts)

predictions = loaded_model.predict(email_features)

# Map predictions to labels (assuming 0=Not Phishing, 1=Phishing)
labels = ["Non-Phishing", "Phishing"]
results = [labels[pred] for pred in predictions]

# Print results and save into a log file
with open("results_log.txt", "w") as f:
    for idx, email in enumerate(parsed_emails):
        f.write(f"Email {idx + 1}:\n")
        f.write(f"Subject: {email['subject']}\n")
        f.write(f"Prediction: {results[idx]}\n")
        f.write("-" * 50 + "\n")

for idx, email in enumerate(parsed_emails):
    print(f"Email {idx + 1}:")
    print("Subject:", email['subject'])
    print("Body:", email['body'])
    print("Prediction:", results[idx])
    print("-" * 50)

# Count the number of phishing and non-phishing emails
label_counts = Counter(results)
print(label_counts)

# show a confusion matrix of phishing / non phishing emails knowing that all emails are non-phishing
# Confusion matrix
cm = confusion_matrix(["Non-Phishing"] * len(results), results, labels=labels)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_dorray_dataset.png')
plt.show()

