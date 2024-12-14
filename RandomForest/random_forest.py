from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from rf_helper import generate_report, get_normalized_data
import pickle
import numpy as np

# 5 or 10 yard groupings
five_yard_grouping = False
seconds_after_snap = 3


X_normalized, y = get_normalized_data(csv_file=f'../AfterSnap/after_snap_{seconds_after_snap+1}.csv',
                                      five_yard_grouping=five_yard_grouping,
                                      cols=['yardsToGo', 'absoluteYardlineNumber', 'preSnapHomeTeamWinProbability',
                                            'teamAbbr', 'preSnapVisitorTeamWinProbability', 'nflId', 'x',
                                            'expectedPoints', 'gameClock', 'y'])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=5)

# set up the model
model = RandomForestClassifier(
    n_estimators=80,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=4
)

# Perform Cross-Validation
cv_folds = 5
cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')

# Print Cross-Validation results
print(f"Cross-Validation Accuracy Scores for {cv_folds} folds: {cv_scores}")
print(f"Mean Accuracy: {np.mean(cv_scores):.4f}")
print(f"Standard Deviation: {np.std(cv_scores):.4f}")

# Train the model on the full training set
model.fit(X_train, y_train)

# Make predictions and calculate accuracy on the test set
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# Save the model to a pickle file
model_filename = f"models/{seconds_after_snap}_seconds_{'five' if five_yard_grouping else 'ten'}_yard_grouping_{round(test_accuracy, 4)}%.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(model, file)

# Generate the report
generate_report(model, X_normalized, X_test, y_test, five_yard_grouping)
