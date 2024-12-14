import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import product
import json
from RandomForest.rf_helper import get_normalized_data

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# Function to create a network dynamically
class DynamicNN(nn.Module):
    def __init__(self, input_size, hidden_layers, activation_function):
        super(DynamicNN, self).__init__()
        layers = []
        current_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_size, hidden_size))
            if activation_function == 'ReLU':
                layers.append(nn.ReLU())
            elif activation_function == 'Tanh':
                layers.append(nn.Tanh())
            elif activation_function == 'Sigmoid':
                layers.append(nn.Sigmoid())
            current_size = hidden_size
        layers.append(nn.Linear(current_size, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Function to train and evaluate a model
def train_and_evaluate(model, optimizer, criterion, X_train, y_train, X_test, y_test, epochs, batch_size):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).squeeze()
        y_pred_labels = (y_pred >= 0.5).float()
        accuracy = (y_pred_labels == y_test).sum().item() / len(y_test)
    return accuracy


# Load and prepare data
seconds_after_snap = 4

X_normalized, y = get_normalized_data(
    csv_file=f'../AfterSnap/after_snap_{seconds_after_snap + 1}.csv',
    five_yard_grouping=False,
    cols=['yardsToGo', 'absoluteYardlineNumber', 'preSnapHomeTeamWinProbability', 'teamAbbr',
          'preSnapVisitorTeamWinProbability', 'nflId', 'x', 'expectedPoints', 'gameClock', 'y']
)

# Standardize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_normalized)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_normalized, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y.values, dtype=torch.float32).to(device)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Hyperparameter search space
hidden_layer_configs = [[64, 32]]  # Focus on simpler architectures
activation_functions = ['ReLu']  # Centered around Tanh
learning_rates = [.0001]  # Close to the best value
batch_sizes = [16]  # Small variations around 32
epochs = 50

# Store results
results = []

# Iterate over all combinations of hyperparameters
for hidden_layers, activation_function, lr, batch_size in product(
        hidden_layer_configs, activation_functions, learning_rates, batch_sizes):
    # Create model
    input_size = X_train.shape[1]
    model = DynamicNN(input_size, hidden_layers, activation_function)

    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Train and evaluate
    accuracy = train_and_evaluate(model, optimizer, criterion, X_train, y_train, X_test, y_test, epochs, batch_size)
    print(
        f"Hidden Layers: {hidden_layers}, Activation: {activation_function}, LR: {lr}, Batch Size: {batch_size}, Accuracy: {accuracy:.4f}")

    # Save results
    results.append({
        'hidden_layers': hidden_layers,
        'activation_function': activation_function,
        'learning_rate': lr,
        'batch_size': batch_size,
        'accuracy': accuracy,
        'model_state_dict': model.state_dict()  # Save model weights
    })

# Sort results by accuracy and keep the top 5
results = sorted(results, key=lambda x: x['accuracy'], reverse=True)[:5]

# Save the top 5 models and parameters
with open('top_models.json', 'w') as f:
    json.dump([{
        'hidden_layers': r['hidden_layers'],
        'activation_function': r['activation_function'],
        'learning_rate': r['learning_rate'],
        'batch_size': r['batch_size'],
        'accuracy': r['accuracy']
    } for r in results], f, indent=4)

# Save model weights to individual files
for i, result in enumerate(results):
    torch.save(result['model_state_dict'], f"model_{i + 1}.pth")

print("Top 5 models and their parameters have been saved!")
