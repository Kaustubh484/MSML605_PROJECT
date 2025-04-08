import torch

class FraudDetectionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(29, 32)
        self.relu_1 = torch.nn.ReLU()
        self.layer_2 = torch.nn.Linear(32, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu_1(self.layer_1(x))
        x = self.sigmoid(self.layer_2(x))

        return x

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)