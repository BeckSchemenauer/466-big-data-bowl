from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# Define a function to categorize x_offset
def categorize_x_offset(x_offset):
    if x_offset < 0:
        return 0
    elif 0 <= x_offset <= 5:
        return 1
    elif 5 < x_offset <= 10:
        return 2
    elif 10 < x_offset <= 15:
        return 3
    elif 15 < x_offset <= 20:
        return 3
    else:
        return 5


def read_category(category):
    if category == 0:
        return 'negative'
    elif category == 1:
        return '0-5'
    elif category == 2:
        return '5-10'
    elif category == 3:
        return '10-15'
    elif category == 4:
        return '15-20'
    else:
        return '20+'


# Function to convert mm:ss to seconds
def convert_to_seconds(time_str):
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds


# Load the CSV file
csv_file = '../AfterSnap/after_snap_3.csv'
data = pd.read_csv(csv_file)

# Drop id columns
data = data.drop(['gameId', 'playId', 'frameId', 'displayName', 'frameType', 'position', 'y_offset'], axis=1)

# Prepare lists to hold the input-output pairs
X = []
y = []

# Iterate through the data in pairs
for i in range(0, len(data) - 1, 2):
    # First row in the pair is the input
    input_row = data.iloc[i].drop('x_offset')  # Drop x_offset if it's in the feature row
    # Second row in the pair, we take the target variable `x_offset`
    x_offset_value = data.iloc[i + 1]['x_offset']

    # Append to X and y, with y being the categorized class
    X.append(input_row)
    y.append(categorize_x_offset(x_offset_value))

# Convert X and y to DataFrames
X = pd.DataFrame(X)
y = pd.Series(y)

# One hot encoding and normalization
X_normalized = pd.get_dummies(X, columns=['offenseFormation', 'receiverAlignment', 'pff_passCoverage', 'pff_manZone'], prefix=['offenseFormation', 'receiverAlignment', 'pff_passCoverage', 'pff_manZone'], drop_first=True)
X_normalized['playDirection'] = np.where(X_normalized['playDirection'] == 'left', 1, 0)
X_normalized['gameClock'] = X_normalized['gameClock'].apply(convert_to_seconds)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Define and train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Calculate overall accuracy
y_pred = model.predict(X_test)
overall_accuracy = accuracy_score(y_test, y_pred)
print('Overall Accuracy:', overall_accuracy)

# Map y_test and y_pred to categorical labels
y_test_labels = [read_category(category) for category in y_test]
y_pred_labels = [read_category(category) for category in y_pred]

# Print classification report for per-class accuracy
print('\nClassification Report:')
print(classification_report(y_test_labels, y_pred_labels))

# Calculate accuracy by category
category_counts = defaultdict(lambda: {'correct': 0, 'total': 0})

for true_label, pred_label in zip(y_test_labels, y_pred_labels):
    category_counts[true_label]['total'] += 1
    if true_label == pred_label:
        category_counts[true_label]['correct'] += 1

print("\nAccuracy by Category:")
for category, counts in category_counts.items():
    category_accuracy = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
    print(f"{category}: {category_accuracy:.2f}")

# Plot feature importance
feature_importance = model.feature_importances_
sorted_idx = feature_importance.argsort()
plt.figure(figsize=(20, 12))
plt.barh(X_normalized.columns[sorted_idx], feature_importance[sorted_idx], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Random Forest Classifier')
plt.show()
