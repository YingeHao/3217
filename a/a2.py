# Test data using the random forest model



import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the data
train_data = pd.read_csv('TrainingDataBinary.csv', header=None)
test_data = pd.read_csv('TestingDataBinary.csv', header=None)

# Prepare the data
X_train = train_data.iloc[:, :-1]  # Training set features
y_train = train_data.iloc[:, -1]   # Training set labels
X_test = test_data.iloc[:, :]      # Testing set features

# Choose and train the model
model = RandomForestClassifier(max_depth=30, max_features='log2', min_samples_leaf=1, n_estimators=200)
model.fit(X_train, y_train)

# Predict labels for the testing data
predictions = model.predict(X_test)

# Add the predicted labels as the last column to the testing data
test_data['Label'] = predictions

# Generate the output file (TestingResultsBinary.csv)
test_data.to_csv('TestingResultsBinary.csv', index=False, header=False)
