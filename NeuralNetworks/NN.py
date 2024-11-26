import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from RandomForest.rf_helper import get_normalized_data

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load and prepare data
seconds_after_snap = 3

X_normalized, y = get_normalized_data(
    csv_file=f'../AfterSnap/after_snap_{seconds_after_snap + 1}.csv',
    five_yard_grouping=False,
    cols=['yardsToGo', 'absoluteYardlineNumber', 'preSnapHomeTeamWinProbability', 'teamAbbr',
          'preSnapVisitorTeamWinProbability', 'nflId', 'x', 'expectedPoints', 'gameClock', 'y']
)

# Standardize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_normalized)

# Convert to PyTorch tensors and move them to the appropriate device
X_tensor = torch.tensor(X_normalized, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device)  # Ensure y is a float tensor for BCELoss

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)


# Define the neural network
class BinaryClassificationNN(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassificationNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


# Model, loss, and optimizer
input_size = X_train.shape[1]
model = BinaryClassificationNN(input_size).to(device)  # Move model to device
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
batch_size = 32
for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train), batch_size):
        # Mini-batch
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        # Forward pass
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print epoch loss
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate on test data
model.eval()
with torch.no_grad():
    y_pred = model(X_test).squeeze()
    y_pred_labels = (y_pred >= 0.5).float()
    accuracy = (y_pred_labels == y_test).sum().item() / len(y_test)
    print(f'Accuracy: {accuracy:.4f}')
