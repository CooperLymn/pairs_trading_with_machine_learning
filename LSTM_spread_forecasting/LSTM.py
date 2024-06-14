import torch
import torch.nn as nn
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, out_features=1):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, out_features=1):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=4, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.lstm = nn.LSTM(input_size=8, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0, 2, 1) #cnn takes input of shape (batch_size, channels, seq_len)
        x = self.cnn(x)
        x = x.permute(0, 2, 1) # lstm takes input of shape (batch_size, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class LSTM_CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, out_features=1):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=4, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(out_features=16),
            nn.ReLU()
        )
        self.fc = nn.Linear(in_features=16, out_features=1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out.permute(0, 2, 1)
        out = self.cnn(out)
        out = self.fc(out)
        return out

def train_one_epoch(model, train_loader, optimizer, device):
    model.train(True)
    loss_function = nn.MSELoss()
    running_loss = 0.0
    output_interval = 10
    count = 0
    for batch_idx, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        count += len(x_batch)

        optimizer.zero_grad()
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item() * len(x_batch)
        loss.backward()
        optimizer.step()

    avg_loss = running_loss / count
    print(f"Training loss :{avg_loss:.6f}")
    return avg_loss


def val_one_epoch(model, val_loader, device):
    model.eval()
    loss_function = nn.MSELoss()
    running_loss = 0.0
    count = 0
    for batch_idx, batch in enumerate(val_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        count += len(x_batch)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item() * len(x_batch)

    avg_loss = running_loss / count
    print(f"Validation loss :{avg_loss:.6f}")
    print("*************************************************")
    return avg_loss
