from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from rf_helper import generate_report, get_normalized_data
import pickle

# 5 or 10 yard groupings
five_yard_grouping = True
seconds_after_snap = 2


X_normalized, y = get_normalized_data(csv_file=f'../AfterSnap/after_snap_{seconds_after_snap+1}.csv',
                                      five_yard_grouping=five_yard_grouping,
                                      cols=['yardsToGo', 'absoluteYardlineNumber', 'preSnapHomeTeamWinProbability', 'teamAbbr', 'preSnapVisitorTeamWinProbability', 'nflId', 'x', 'expectedPoints', 'gameClock', 'y'])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Define and train the model with the current hyperparameters
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=13
)

model.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save the best model to a file
with open(f"models/{seconds_after_snap}_seconds_{"five" if five_yard_grouping else "ten"}_yard_grouping.pkl", "wb") as file:
    pickle.dump(model, file)

generate_report(model, X_normalized, X_test, y_test, True)
