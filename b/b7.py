# Adjusting the parameters of the random forest model using RandomizedSearch


import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score

# Load Training Data
data = pd.read_csv('TrainingDataMulti.csv')

# Separate Features and Labels
features = data.iloc[:, :128]
labels = data.iloc[:, 128]

# Split the data into training and test set
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define parameter distributions
param_dist = {
    'n_estimators': [500, 600, 700],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 1.0]
}

# Create XGBoost classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Perform randomized search
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, scoring='accuracy', cv=3, random_state=42, verbose=1, n_jobs=-1)
random_search.fit(features_train, labels_train)

# Best parameters
best_params = random_search.best_params_
print("Best Parameters: ", best_params)

# Evaluate on test set
best_model = random_search.best_estimator_
predicted_labels = best_model.predict(features_test)
accuracy = accuracy_score(labels_test, predicted_labels)
print("Accuracy: ", accuracy)
