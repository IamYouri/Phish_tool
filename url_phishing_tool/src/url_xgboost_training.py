import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBClassifier  # Import XGBoost

# Charger les datasets
original_data = pd.read_csv('../phishing_dataset/Phishing_Legitimate_full.csv')
new_data = pd.read_csv('../phishing_dataset/transformed_phishing_urls.csv')

# Combiner les deux datasets
combined_data = pd.concat([original_data, new_data], ignore_index=True)

# Réentraîner le modèle
selected_features = ['NumDots', 'NumDash', 'UrlLength', 'PathLevel', 
                     'NumQueryComponents', 'HostnameLength', 
                     'DoubleSlashInPath', 'AtSymbol', 'IpAddress', 'NoHttps']
X = combined_data[selected_features]
y = combined_data['CLASS_LABEL']

# Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Entraîner le modèle avec XGBoost
model = XGBClassifier(
    n_estimators=200,        # Nombre d'arbres
    max_depth=8,             # Profondeur maximale des arbres
    learning_rate=0.1,       # Taux d'apprentissage
    scale_pos_weight=1,      # Ajustement pour les classes déséquilibrées
    use_label_encoder=False, # Désactiver l'encodage des labels de XGBoost (déprécié)
    random_state=42
)

model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Calculer et afficher la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Phishing', 'Phishing'], yticklabels=['Non-Phishing', 'Phishing'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('./models/confusion_matrix_xgboost.png')
plt.show()

# Sauvegarder le modèle et le scaler
# joblib.dump(model, './models/xgboost_model.pkl')
# joblib.dump(scaler, './models/scaler_xgboost.pkl')

from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

# Générer les learning curves
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Calculer les moyennes et écarts-types pour les scores d'entraînement et de validation croisée
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Tracer les learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.title('Learning Curves for XGBoost')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('./models/learning_curves_xgboost.png')
plt.show()


