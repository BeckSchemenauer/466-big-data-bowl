import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

train_data = pd.read_csv('../AfterSnap/after_snap_3.csv')
train_data.drop(["displayName", "position", "playId", "gameId"], inplace=True, axis=1)
def convert_to_seconds(clock):
    minutes, seconds = map(int, clock.split(":"))
    return minutes * 60 + seconds

train_data['gameClock'] = train_data['gameClock'].apply(convert_to_seconds)

categorical_features = ['frameType', 'playDirection', 'inMotionAtBallSnap', 'shiftSinceLineset', 'motionSinceLineset', 'offenseFormation', 'receiverAlignment', 'pff_passCoverage', 'pff_manZone']
encoder = OneHotEncoder()

# Perform one-hot encoding on categorical features
encoded_categoricals = encoder.fit_transform(train_data[categorical_features]).toarray()
encoded_df = pd.DataFrame(encoded_categoricals, columns=encoder.get_feature_names_out())

train_data.drop(categorical_features, axis=1, inplace=True)
processed_data = pd.concat([train_data, encoded_df], axis=1)


# Split into inputs and outputs
inputs = processed_data.iloc[::2].reset_index(drop=True)
outputs = processed_data.iloc[1::2].reset_index(drop=True)

X = inputs.drop(['x_offset', 'y_offset'], axis=1)
y = outputs[['x_offset', 'y_offset']]
#groups = inputs["nflId"]

nan_indices = X[X.isna().any(axis=1)].index

# Drop the rows with NaN in X and the corresponding rows in y
X = X.drop(nan_indices).reset_index(drop=True)
y = y.drop(nan_indices).reset_index(drop=True)

'''potentially make stratified'''
X_train, X_test, y_train, y_test = train_test_split(
    X.to_numpy(), y.to_numpy(), test_size=0.2, random_state=10
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class FFNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.batch_norm3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.batch_norm3(x)
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x


def is_correct(x_pred, y_pred, x_true, y_true, threshold=2):
    print(x_pred, y_pred, x_true, y_true)
    return abs(x_pred - x_true) <= threshold and abs(y_pred - y_true) <= threshold

# Instantiate the model
input_size = X_train_tensor.shape[1]  # Number of features
output_size = y_train_tensor.shape[1]  # Number of target variables (x, y)
model = FFNN(input_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Evaluate accuracy on the test set based on the threshold
model.eval()
test_loss = 0.0
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        test_loss += loss.item()

        for i in range(len(y_batch)):
            x_true, y_true = y_batch[i]
            x_pred, y_pred = outputs[i]
            if is_correct(x_pred.item(), y_pred.item(), x_true.item(), y_true.item(), threshold=2):
                correct_predictions += 1
            total_predictions += 1

# Calculate accuracy and print results
accuracy = correct_predictions / total_predictions
print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {accuracy:.4f}")