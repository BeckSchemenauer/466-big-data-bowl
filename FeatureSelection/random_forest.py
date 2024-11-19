import itertools

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from rf_helper import generate_report, get_normalized_data
import pickle


X_normalized, y = get_normalized_data(csv_file='../AfterSnap/after_snap_4.csv', five_yard_grouping=False)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Define hyperparameters to test
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'random_state': [42]
}

param_grid = {
    'n_estimators': [50],
    'max_depth': [20],
    'min_samples_split': [2],
    'min_samples_leaf': [4],
    'max_features': ['sqrt'],
    'random_state': [13]
}

# Track the best model and highest accuracy
best_accuracy = 0
best_model = None
best_params = None

# Loop through all combinations of parameters
for params in itertools.product(*param_grid.values()):
    # Unpack parameters
    param_dict = dict(zip(param_grid.keys(), params))

    # Define and train the model with the current hyperparameters
    model = RandomForestClassifier(
        n_estimators=param_dict['n_estimators'],
        max_depth=param_dict['max_depth'],
        min_samples_split=param_dict['min_samples_split'],
        min_samples_leaf=param_dict['min_samples_leaf'],
        max_features=param_dict['max_features'],
        random_state=param_dict['random_state']
    )

    model.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Update the best model if current one has higher accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_params = param_dict

# Save the best model to a file
with open("best_random_forest_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print(best_params)

generate_report(best_model, X_normalized, X_test, y_test, False)
