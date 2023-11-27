from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import joblib

# Eğittiğimiz modelleri yükle
ridge_classifier = joblib.load('Classifiers/RidgeClassifier.joblib')
passive_aggressive_classifier = joblib.load(
    'Classifiers/PassiveAggressive.joblib')
random_forest = joblib.load('Classifiers/RandomForest.joblib')
gradient_boosting_classifier = joblib.load(
    'Classifiers/GradientBoosting.joblib')
perceptron = joblib.load('Classifiers/Perceptron.joblib')
sgd_classifier = joblib.load('Classifiers/SGDClassifier.joblib')
decision_tree = joblib.load('Classifiers/DecisionTree.joblib')
linear_svc = joblib.load('Classifiers/LinearSVC.joblib')
logistic_regression = joblib.load('Classifiers/LogisticRegression.joblib')

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

# Eğitim veri setini ve etiketleri yükleyin (bu kısımları kendi veri setinizle değiştirin)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_discrete_numeric, test_size=0.2, random_state=42)

# Ensemble modelini oluşturun
ensemble_model = VotingClassifier(estimators=[
    ('ridge', ridge_classifier),
    ('passive_aggressive', passive_aggressive_classifier),
    ('random_forest', random_forest),
    ('gradient_boosting', gradient_boosting_classifier),
    ('perceptron', perceptron),
    ('sgd', sgd_classifier),
    ('decision_tree', decision_tree),
    ('linear_svc', linear_svc),
    ('logistic_regression', logistic_regression)
], voting='hard')  # 'hard' kullanarak çoğunluk oylaması yapabilirsiniz

# Ensemble modelini eğitin
ensemble_model.fit(X_train, y_train)

# Test veri setini kullanarak modeli değerlendirin
y_pred = ensemble_model.predict(X_test)

# Accuracy'yi yazdırın (veya başka bir performans metriğini seçebilirsiniz)
accuracy = accuracy_score(y_test, y_pred)
print(f'Ensemble Model Accuracy: {accuracy}')
