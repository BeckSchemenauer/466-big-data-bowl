import joblib
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from RandomForest.rf_helper import get_normalized_data
from matplotlib import pyplot as plt
import numpy as np

# Load and prepare data
seconds_after_snap = 3

# Load normalized data
X_normalized, y = get_normalized_data(
    csv_file=f'../AfterSnap/after_snap_{seconds_after_snap + 1}.csv',
    five_yard_grouping=False,
    cols=['yardsToGo', 'absoluteYardlineNumber', 'preSnapHomeTeamWinProbability', 'teamAbbr',
          'preSnapVisitorTeamWinProbability', 'nflId', 'x', 'expectedPoints', 'gameClock', 'y']
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=5)

# Initialize the XGBoost classifier with modified hyperparameters for longer training
model = xgb.XGBClassifier(
    objective='binary:logistic',
    tree_method='gpu_hist',  # Use GPU if available
    n_estimators=100,         # Increased number of trees for longer training
    max_depth=3,              # Depth of the trees
    learning_rate=0.15,       # Lower learning rate for gradual learning
    subsample=0.8,            # Subsample ratio of the training instances
    colsample_bytree=0.9,     # Subsample ratio of columns when constructing each tree
    min_child_weight=10       # Minimum sum of instance weight (hessian) in a child
)

# Perform 5-Fold Cross-Validation
cv_folds = 5
cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')

# Print the cross-validation results
print(f"Cross-Validation Accuracy Scores for {cv_folds} folds: {cv_scores}")
print(f"Mean Accuracy: {np.mean(cv_scores):.4f}")
print(f"Standard Deviation: {np.std(cv_scores):.4f}")

# Fit the model to the training data
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

y_pred = model.predict(X_test)

# Calculate accuracy and print the classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model to a file
model_filename = f"models/xgb_model_{seconds_after_snap}_seconds_accuracy_{round(accuracy, 4)}.pkl"
with open(model_filename, "wb") as file:
    joblib.dump(model, file)
