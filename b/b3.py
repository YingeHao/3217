#xgboost model

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load Training Data
data = pd.read_csv('TrainingDataMulti.csv')

# Separate Features and Labels
features = data.iloc[:, :128]
labels = data.iloc[:, 128]

# Split the data into training and test set
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train Model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', subsample=0.8, n_estimators=700, max_depth=5, learning_rate=0.1, colsample_bytree=0.6)
model.fit(features_train, labels_train)

# Predict on Test Data
predicted_labels = model.predict(features_test)

# Accuracy
accuracy = accuracy_score(labels_test, predicted_labels)
print(f'Accuracy: {accuracy}')

# Precision
precision = precision_score(labels_test, predicted_labels, average='weighted')
print(f'Precision: {precision}')

# Recall
recall = recall_score(labels_test, predicted_labels, average='weighted')
print(f'Recall: {recall}')

# F1 Score
f1 = f1_score(labels_test, predicted_labels, average='weighted')
print(f'F1 Score: {f1}')

# Confusion Matrix
conf_mat = confusion_matrix(labels_test, predicted_labels)
print('Confusion Matrix:\n', conf_mat)

# Plot Confusion Matrix
plt.figure(figsize=(10,7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap="Greens")
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title("confusion matrix")
plt.show()
