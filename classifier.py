import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC

# Verileri yükle
data = np.load('combined_feature_vectors.npy')
X = data[:, :-1]  # Feature vectors
y = data[:, -1]   # Star ratings

bins = [0, 1, 2, 3, 4, 5]  # Define the bin edges
labels = ['Low', 'Low-Med', 'Medium', 'Med-High',
          'High']  # Assign labels to the bins

# Kesikli kategorileri içeren yeni bir sütun oluşturun
y_discrete = pd.cut(y, bins=bins, labels=labels, include_lowest=True)

# Alanları sayısal değerlere eşleyin
label_mapping = {'Low': 1, 'Low-Med': 2, 'Medium': 3, 'Med-High': 4, 'High': 5}
y_discrete_numeric = y_discrete.map(label_mapping)

# Veriyi eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(
    X, y_discrete_numeric, test_size=0.2, random_state=42)

# Ridge Classifier
model = RidgeClassifier()

# Modeli eğitin
model.fit(X_train, y_train)

# Test veri setini kullanarak modeli değerlendirin
y_pred = model.predict(X_test)

# Modelin performansını değerlendirin
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)

# Eğitilen modeli kaydedin
joblib.dump(model, 'Classifiers/RidgeClassifier.joblib')
