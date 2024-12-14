import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR

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
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


# Model, loss, and optimizer
input_size = X_train.shape[1]
model = BinaryClassificationNN(input_size).to(device)  # Move model to device
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0004)

# Initialize lists to store metrics
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# Training loop
epochs = 30
batch_size = 16

# Add a variable to track the best test accuracy
best_test_accuracy = 0.0
best_model_path = "best_model.pth"

#scheduler = StepLR(optimizer, step_size=10, gamma=0.01)  # Reduce LR by 5% every 10 epochs

# Training loop
for epoch in range(epochs):
    # Training phase
    model.train()
    epoch_train_loss = 0
    correct_train = 0
    total_train = 0

    for i in range(0, len(X_train), batch_size):
        # Mini-batch
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        # Forward pass
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        epoch_train_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy for the batch
        predictions = (outputs >= 0.5).float()
        correct_train += (predictions == y_batch).sum().item()
        total_train += y_batch.size(0)

    train_losses.append(epoch_train_loss / len(X_train))
    train_accuracies.append(correct_train / total_train)

    # Validation phase
    model.eval()
    epoch_test_loss = 0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        outputs = model(X_test).squeeze()
        test_loss = criterion(outputs, y_test)
        epoch_test_loss = test_loss.item()

        # Compute accuracy for the test set
        predictions = (outputs >= 0.5).float()
        correct_test += (predictions == y_test).sum().item()
        total_test += y_test.size(0)

    test_losses.append(epoch_test_loss / len(X_test))
    test_accuracies.append(correct_test / total_test)

    # Save model if the test accuracy improves
    if test_accuracies[-1] > best_test_accuracy:
        best_test_accuracy = test_accuracies[-1]
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with test accuracy: {best_test_accuracy:.4f}")

    # Print metrics for the epoch
    print(f"Epoch [{epoch + 1}/{epochs}], "
          f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, "
          f"Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}")

    #scheduler.step()


# Plot training vs test accuracy
plt.plot(range(1, epochs + 1), train_accuracies, label="Training Accuracy")
plt.plot(range(1, epochs + 1), test_accuracies, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Test Accuracy")
plt.legend()

plt.show()

# Load the best model for evaluation
model.load_state_dict(torch.load(best_model_path))
model.eval()

# Evaluate on the test data using the best model
with torch.no_grad():
    y_pred = model(X_test).squeeze()
    y_pred_labels = (y_pred >= 0.5).float()

# Convert tensors back to CPU for sklearn compatibility
y_test_cpu = y_test.cpu().numpy()
y_pred_labels_cpu = y_pred_labels.cpu().numpy()

# Print classification report
report = classification_report(y_test_cpu, y_pred_labels_cpu, target_names=["Class 0", "Class 1"])
print("Classification Report with Best Model:")
print(report)
