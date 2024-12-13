import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re # Nettoyage de base du texte
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
import joblib
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# Charger le fichier CSV
file_path = "../enron_dataset/enron.csv"
data = pd.read_csv(file_path)

# # Afficher les premières lignes
# print(data.head())

# # Vérifier les colonnes
# print(data.columns)

# # Vérifier la distribution des labels
# print(data['label'].value_counts())

# Combiner "subject" et "body" dans une seule colonne
data['email_text'] = data['subject'] + " " + data['body']

# Garder uniquement les colonnes nécessaires
data = data[['email_text', 'label']]

# Supprimer les valeurs manquantes
data.dropna(inplace=True)

def clean_text(text):
    text = text.lower()  # Mettre en minuscule
    #non ascii characters are removed for ex. corean and chinese characters are removed
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # supprimer les caractères non-alphanumériques à l'excpetion de @ . : // ? = & - _
    text = re.sub(r'[^\w@.:\/\?=&\-_]+', ' ', text) 
    text = re.sub(r'\s+', ' ', text)  # Supprimer les espaces multiples
    text = text.strip()  # Supprimer les espaces en début/fin
    return text

data['email_text'] = data['email_text'].apply(clean_text)

print(data.head())

# Features et labels
X = data['email_text']
y = data['label']

# Diviser en ensemble d'entraînement et de test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Taille de l'ensemble d'entraînement :", len(X_train))
print("Taille de l'ensemble de test :", len(X_test))

# Initialiser le vectoriseur TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)

# Adapter sur l'ensemble d'entraînement et transformer les textes
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Taille des features TF-IDF (ensemble d'entraînement) :", X_train_tfidf.shape)

# Initialiser et entraîner le modèle
model = LogisticRegression(C=0.1)
model.fit(X_train_tfidf, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test_tfidf)

# Évaluer les performances
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Rapport de classification :\n", classification_report(y_test, y_pred))
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))

# Sauvegarder le modèle et le vectoriseur
joblib.dump(model, "phishing_email_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Générer les learning curves
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train_tfidf, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Calculer les moyennes et écarts-types pour l'entraînement et le test
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Tracer les learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')

# Ajouter des bandes d'incertitude pour l'écart-type
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')

# Ajouter les labels
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend(loc='best')
plt.grid(True)
plt.savefig("learning_curves_mail_model.png")
plt.show()

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)

# Tracer la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues' , xticklabels=['Non-Phishing', 'Phishing'], yticklabels=['Non-Phishing', 'Phishing'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix_mail_model.png")
plt.show()