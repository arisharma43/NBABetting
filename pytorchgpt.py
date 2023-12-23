import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

gamedetails=open("games_details.csv","r")
games=open("games.csv","r")
players=open("players.csv","r")
ranking=open("ranking.csv","r")
teams=open("teams.csv","r")




# Sample data (replace this with your dataset)
# For simplicity, I'm assuming you have features like 'average_points', 'minutes_played', etc.
# and the target variable is 'points_scored'.
# Make sure to replace this with your actual dataset.
data = {
    'average_points': [20.5, 18.2, 22.1, 15.9, 21.3],
    'minutes_played': [30, 28, 32, 25, 31],
    'points_scored': [25, 22, 28, 20, 27],
}

df = pd.DataFrame(data)

# Split the data into features (X) and target variable (y)
X = df[['average_points', 'minutes_played']].values
y = df['points_scored'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# Define a simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
input_size = X_train.shape[1]
model = LinearRegressionModel(input_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training the model
num_epochs = 1000

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor.view(-1, 1))

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor.view(-1, 1))

print(f'Test Loss: {test_loss.item():.4f}')

# Make predictions on new data
new_data = np.array([[23.0, 29.0]])  # Replace this with the new data you want to predict
new_data_scaled = scaler.transform(new_data)
new_data_tensor = torch.FloatTensor(new_data_scaled)

model.eval()
with torch.no_grad():
    prediction = model(new_data_tensor)
    print(f'Predicted Points: {prediction.item():.2f}')