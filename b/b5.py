#Neural Networks model


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('TrainingDataMulti.csv')
X = data.iloc[:, 0:128].values
y = data.iloc[:, 128].values

# Data Preprocessing
encoder = OneHotEncoder()
y = encoder.fit_transform(y.reshape(-1, 1)).toarray()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Define model
model = Sequential()
model.add(Dense(64, input_dim=128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=20)

# Evaluate model
y_pred = model.predict(X_test)

# Convert y_test and y_pred to labels from one hot encoding
y_test_argmax = y_test.argmax(axis=1)
y_pred_argmax = y_pred.argmax(axis=1)

# Accuracy
accuracy = accuracy_score(y_test_argmax, y_pred_argmax)
print(f'Accuracy: {accuracy}')

# F1 Score
f1 = f1_score(y_test_argmax, y_pred_argmax, average='weighted')
print(f'F1 Score: {f1}')

# Precision
precision = precision_score(y_test_argmax, y_pred_argmax, average='weighted')
print(f'Precision: {precision}')

# Recall
recall = recall_score(y_test_argmax, y_pred_argmax, average='weighted')
print(f'Recall: {recall}')

# Error Rate
error_rate = 1 - accuracy
print(f'Error Rate: {error_rate}')

# Area Under the ROC Curve
auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
print(f'Area Under the ROC Curve: {auc}')
