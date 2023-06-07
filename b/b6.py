# K-Nearest Neighbors model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Load Training Data
data = pd.read_csv('TrainingDataMulti.csv')

# Separate Features and Labels
features = data.iloc[:, :128]
labels = data.iloc[:, 128]

# Split the data into training and test set
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train Model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(features_train, labels_train)

# Predict on Test Data
predicted_labels = model.predict(features_test)

# Accuracy
accuracy = accuracy_score(labels_test, predicted_labels)
print(f'Accuracy: {accuracy}')

# F1 Score
f1 = f1_score(labels_test, predicted_labels, average='weighted')
print(f'F1 Score: {f1}')

# Precision
precision = precision_score(labels_test, predicted_labels, average='weighted')
print(f'Precision: {precision}')

# Recall
recall = recall_score(labels_test, predicted_labels, average='weighted')
print(f'Recall: {recall}')

# Error Rate
error_rate = 1 - accuracy
print(f'Error Rate: {error_rate}')

# Area Under the ROC Curve
probabilities = model.predict_proba(features_test)
auc = roc_auc_score(labels_test, probabilities, multi_class='ovr')
print(f'Area Under the ROC Curve: {auc}')
