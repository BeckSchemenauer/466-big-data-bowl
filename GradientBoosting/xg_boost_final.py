import joblib
import xgboost as xgb
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from RandomForest.rf_helper import get_normalized_data
from sklearn.model_selection import learning_curve
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
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Initialize the XGBoost classifier with modified hyperparameters for longer training
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='gpu_hist',  # Use GPU if available
    n_estimators=100,         # Increased number of trees for longer training
    max_depth=3,              # Depth of the trees
    learning_rate=0.05,       # Lower learning rate for gradual learning
    subsample=0.9,            # Subsample ratio of the training instances
    colsample_bytree=0.9,     # Subsample ratio of columns when constructing each tree
    gamma=0.1,                # Minimum loss reduction required to make a further partition
    min_child_weight=10       # Minimum sum of instance weight (hessian) in a child
)

# Fit the model to the training data with evaluation
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    model, X_normalized, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, label='Training Accuracy')
plt.plot(train_sizes, test_mean, label='Validation Accuracy')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.show()

# Now calculate predictions and evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
