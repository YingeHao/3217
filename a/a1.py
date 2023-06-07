# Training a random forest model and evaluating its accuracy


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training data
data = pd.read_csv('TrainingDataBinary.csv')

# Separate features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(max_depth=30, max_features='log2', min_samples_leaf=1, n_estimators=200)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

# Output evaluation metrics
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}".format(np.mean(precision)))
print("Recall: {:.2f}".format(np.mean(recall)))
print("F1 Score: {:.2f}".format(f1))
print("Confusion Matrix:\n", confusion_mat)

# Plot confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

