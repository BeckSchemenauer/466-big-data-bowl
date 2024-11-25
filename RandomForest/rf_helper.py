from collections import defaultdict
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report


def group_by_five(x_offset):
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


def group_by_ten(x_offset):
    if x_offset < 10:
        return 0
    else:
        return 1


# Function to convert mm:ss to seconds
def convert_to_seconds(time_str, quarter):
    minutes, seconds = map(int, time_str.split(':'))

    # if quarter == 1 or quarter == 3:
    #     return minutes * 60 + seconds + 900

    return minutes * 60 + seconds


def get_normalized_data(csv_file, five_yard_grouping):
    # Load the CSV file
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
        if five_yard_grouping:
            y.append(group_by_five(x_offset_value))
        else:
            y.append(group_by_ten(x_offset_value))

    # Convert X and y to DataFrames
    X = pd.DataFrame(X)
    y = pd.Series(y)

    # One hot encoding and normalization
    X_normalized = pd.get_dummies(X,
                                  columns=['offenseFormation', 'receiverAlignment', 'pff_passCoverage', 'pff_manZone',
                                           'teamAbbr'],
                                  prefix=['offenseFormation', 'receiverAlignment', 'pff_passCoverage', 'pff_manZone',
                                          'teamAbbr'], drop_first=True)
    X_normalized['playDirection'] = np.where(X_normalized['playDirection'] == 'left', 1, 0)
    X_normalized['gameClock'] = X_normalized.apply(lambda row: convert_to_seconds(row['gameClock'], row['quarter']),
                                                   axis=1)

    return X_normalized, y


def read_five_yard_category(category):
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


def read_ten_yard_category(category):
    if category == 0:
        return '0-10'
    elif category == 1:
        return '10+'


def generate_report(model, X_normalized, X_test, y_test, five_yard_grouping):
    # Calculate overall accuracy
    y_pred = model.predict(X_test)
    overall_accuracy = accuracy_score(y_test, y_pred)
    print('Overall Accuracy:', overall_accuracy)

    if five_yard_grouping:
        # Map y_test and y_pred to categorical labels
        y_test_labels = [read_five_yard_category(category) for category in y_test]
        y_pred_labels = [read_five_yard_category(category) for category in y_pred]
    else:
        # Map y_test and y_pred to categorical labels
        y_test_labels = [read_ten_yard_category(category) for category in y_test]
        y_pred_labels = [read_ten_yard_category(category) for category in y_pred]

    # Print classification report for per-class accuracy
    print('\nClassification Report:')
    print(classification_report(y_test_labels, y_pred_labels, zero_division=1))

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

    # Define feature groups
    feature_groups = {
        'offenseFormation': [col for col in X_normalized.columns if col.startswith('offenseFormation_')],
        'receiverAlignment': [col for col in X_normalized.columns if col.startswith('receiverAlignment_')],
        'pff_passCoverage': [col for col in X_normalized.columns if col.startswith('pff_passCoverage_')],
        'pff_manZone': [col for col in X_normalized.columns if col.startswith('pff_manZone_')],
        'teamAbbr': [col for col in X_normalized.columns if col.startswith('teamAbbr_')]
    }

    # Calculate importances for each group
    group_importances = {}
    all_grouped_features = set()
    for group_name, features in feature_groups.items():
        group_importances[group_name] = feature_importance[
            sorted_idx[np.isin(X_normalized.columns[sorted_idx], features)]].sum()
        all_grouped_features.update(features)

    # Add ungrouped features individually
    ungrouped_features = [col for col in X_normalized.columns if col not in all_grouped_features]
    for feature in ungrouped_features:
        feature_idx = np.where(X_normalized.columns == feature)[0][0]
        group_importances[feature] = feature_importance[feature_idx]

    # Sort the final importances for better visualization
    sorted_importances = dict(sorted(group_importances.items(), key=lambda item: item[1]))

    # Plot the feature group importances
    plt.figure(figsize=(20, 12))
    plt.barh(list(sorted_importances.keys()), list(sorted_importances.values()), color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance (Grouped and Individual Features) in Random Forest Classifier')

    # Decrease font size of feature labels
    plt.yticks(fontsize=10)

    # Add value labels to the bars
    for i, v in enumerate(sorted_importances.values()):
        plt.text(v + 0.01, i, str(round(v, 3)), color='black', va='baseline', fontweight='bold', fontsize=12)

    plt.show()
