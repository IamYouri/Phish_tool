import joblib

# Charger le modèle et le vectoriseur
loaded_model = joblib.load("phishing_email_model.pkl")
loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Nouvel email à tester
new_email = ["Dear collegue, I want the file for uploading the data. Please send it to me as soon as possible."]

# Nettoyer et vectoriser le texte
new_email_tfidf = loaded_vectorizer.transform(new_email)

# Prédire
prediction = loaded_model.predict(new_email_tfidf)
print("Phishing" if prediction[0] == 1 else "Non-Phishing")