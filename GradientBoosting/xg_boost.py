import joblib
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from RandomForest.rf_helper import get_normalized_data

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
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Initialize the XGBoost classifier for binary classification
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='gpu_hist',  # Use GPU if available
    n_estimators=100
)

# Define a grid of hyperparameters to search over
param_grid = {
    'max_depth': [3,],  # Depth of the trees
    'learning_rate': [0.1],  # Learning rate
    'subsample': [0.9],  # Subsample ratio of the training instances
    'colsample_bytree': [0.9],  # Subsample ratio of columns when constructing each tree
    'gamma': [0.1,],  # Minimum loss reduction required to make a further partition
    'min_child_weight': [10,]  # Minimum sum of instance weight (hessian) in a child
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best accuracy score
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Print out each set of parameters and corresponding accuracy
print("Grid Search Results:")
for params, mean_score, scores in zip(grid_search.cv_results_['params'],
                                      grid_search.cv_results_['mean_test_score'],
                                      grid_search.cv_results_['std_test_score']):
    print(f"Params: {params} | Mean Accuracy: {mean_score:.4f} | Std Accuracy: {scores:.4f}")

# Output the best parameters and best accuracy score
print("\nBest Hyperparameters found:")
print(f"Best Parameters: {best_params}")
print(f"Best Accuracy: {best_accuracy:.4f}")

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict the labels
y_pred = best_model.predict(X_test)

# Generate classification report
report = classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"])
print("Classification Report:")
print(report)

# Save the best model to a file
joblib.dump(best_model, 'best_xgb_model.pkl')
