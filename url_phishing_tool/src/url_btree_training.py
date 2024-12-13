import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import xgboost as xgb

# Charger la dataset
data = pd.read_csv('../phishing_dataset/Phishing_Legitimate_full.csv')

# Sélectionner les features simples
selected_features = ['NumDots', 'NumDash', 'UrlLength', 'PathLevel', 
                     'NumQueryComponents', 'HostnameLength', 
                     'DoubleSlashInPath', 'AtSymbol', 'IpAddress', 'NoHttps']
X = data[selected_features]
y = data['CLASS_LABEL']

# Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser les données en entraînement et test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialiser le modèle XGBoost
model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)

# Entraîner le modèle
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluer le modèle
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Sauvegarder le modèle
joblib.dump(model, './simple_model_results/xgboost_model.pkl')
joblib.dump(scaler, './simple_model_results/scaler_xgboost.pkl')

# Afficher l'accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualisation de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Phishing", "Phishing"], yticklabels=["Non-Phishing", "Phishing"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.savefig('./simple_model_results/confusion_matrix_xgboost.png')
plt.show()

# Afficher l'importance des features
feature_importance = model.feature_importances_
features = selected_features

importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importances - XGBoost')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('./simple_model_results/feature_importances_xgboost.png')
plt.show()
