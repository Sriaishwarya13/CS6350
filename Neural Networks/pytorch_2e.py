import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, filepath):
        features, labels = [], []
        with open(filepath, 'r') as file:
            for line in file:
                data = line.strip().split(",")
                data = list(map(np.float32, data))
                features.append(data[:-1])
                labels.append(data[-1])
        
        self.features = np.array(features)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

train_data = CustomDataset('../data/bank-note/train.csv')
test_data = CustomDataset('../data/bank-note/train.csv')

batch_size = 10
train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

for features, labels in test_loader:
    print("Features:", features)
    print("Labels:", labels)
    break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

def train_one_epoch(loader, model, loss_fn, optimizer):
    model.train()
    epoch_loss = []
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        predictions = model(inputs)
        loss = loss_fn(predictions.view(-1), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            epoch_loss.append(loss.item())
    avg_loss = np.mean(epoch_loss)
    print(f"Training loss: {avg_loss:.6f}")
    return epoch_loss

def evaluate_model(loader, model, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            total_loss += loss_fn(predictions.view(-1), targets).item()
    avg_test_loss = total_loss / len(loader)
    print(f"Test loss: {avg_test_loss:.6f}\n")

def xavier_initializer(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        layer.bias.data.fill_(0.01)

def he_initializer(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight)
        layer.bias.data.fill_(0.01)

hidden_layer_sizes = [5, 10, 25, 50, 100]
layer_depths = [3, 5, 9]
activation_functions = [(nn.ReLU(), he_initializer, "ReLU"), (nn.Tanh(), xavier_initializer, "Tanh")]

for activation_fn, init_fn, activation_name in activation_functions:
    print(f"Using activation function: {activation_name}")
    for hidden_size in hidden_layer_sizes:
        for depth in layer_depths:
            print(f"Architecture: {depth}-layer, {hidden_size}-units per layer")
            
            class MLPModel(nn.Module):
                def __init__(self):
                    super(MLPModel, self).__init__()
                    self.input_layer = nn.Sequential(nn.Linear(4, hidden_size), activation_fn)
                    self.hidden_layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, hidden_size), activation_fn) for _ in range(depth-2)])
                    self.output_layer = nn.Linear(hidden_size, 1)

                def forward(self, x):
                    x = self.input_layer(x)
                    for layer in self.hidden_layers:
                        x = layer(x)
                    return self.output_layer(x)

            model = MLPModel().to(device)
            model.apply(init_fn)

            loss_function = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            all_train_losses = []
            epochs = 15
            for epoch in range(epochs):
                print(f"Epoch {epoch+1}/{epochs}", end=' ')
                train_losses = train_one_epoch(train_loader, model, loss_function, optimizer)
                all_train_losses.extend(train_losses)

            plt.figure()
            plt.plot(all_train_losses)
            plt.title(f"PyTorch: {depth}-layer, {hidden_size}-units model")
            plt.xlabel("Iteration")
            plt.ylabel("Squared Error")
            plt.savefig(f"./out/model_{activation_name}_depth{depth}_width{hidden_size}.png")
            plt.close()

            evaluate_model(test_loader, model, loss_function)

print("Training and testing complete. Plots saved in './out/'")
