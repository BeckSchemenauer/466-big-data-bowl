from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from rf_helper import generate_report, get_normalized_data

X_normalized, y = get_normalized_data('../AfterSnap/after_snap_3.csv')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Define and train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

generate_report(model, X_normalized, X_test, y_test)
