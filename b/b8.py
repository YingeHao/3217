# Test data using the xgboost model


import pandas as pd
from xgboost import XGBClassifier

# Load Training Data
train_data = pd.read_csv('TrainingDataMulti.csv', header=None)

# Separate Features and Labels in Training Data
train_features = train_data.iloc[:, :-1]
train_labels = train_data.iloc[:, -1]

# Load Testing Data
test_data = pd.read_csv('TestingDataMulti.csv', header=None)

# Create XGBoost classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', subsample=0.8, n_estimators=700, max_depth=5, learning_rate=0.1, colsample_bytree=0.6)

# Train Model
model.fit(train_features, train_labels)

# Predict on Test Data
predicted_labels = model.predict(test_data)

# Add predicted labels to Testing Data
test_data['Predicted Label'] = predicted_labels

# Save Testing Data with Predicted Labels to CSV file
test_data.to_csv('TestingResultsMulti.csv', index=False, header=False)
