import torch
import argparse
import pandas as pd
import os
from clearml import Task
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from model import FraudDetectionModel, CustomDataset
from hyperparameters import num_epochs, batch_size, learning_rate
# Set up ClearML connection from environment variables
os.environ['CLEARML_API_ACCESS_KEY'] = os.getenv('CLEARML_API_ACCESS_KEY', '')
os.environ['CLEARML_API_SECRET_KEY'] = os.getenv('CLEARML_API_SECRET_KEY', '')
os.environ['CLEARML_API_HOST'] = os.getenv('CLEARML_API_HOST', 'https://app.clear.ml') 

task = Task.init(
    project_name="Fraud Detection",  # Name of your project in ClearML UI
    task_name="PyTorch Model Training",  # Name of this experiment
)
logger = task.get_logger()
def train(num_epochs, criterion, optimizer, train_loader, test_loader, device):
    for epoch in range(num_epochs):
        train_correct = 0
        train_loss = 0
        train_size = 0
        
        test_correct = 0
        test_loss = 0
        test_size = 0
        
        for features, labels in train_loader:
            features = features.to(device).float()
            labels = labels.to(device).float()
            optimizer.zero_grad()

            probs = model(features).squeeze()
            predictions = (probs > 0.5).float()

            loss = criterion(probs, labels)
            
            loss.backward()
            optimizer.step()

            correct = (predictions == labels).sum()
            
            train_correct += correct
            train_loss += loss.item()
            train_size += len(labels)

        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device).float()
                labels = labels.to(device).float()
    
                probs = model(features).squeeze()
                predictions = (probs > 0.5).float()

                loss = criterion(probs, labels)
    
                correct = (predictions == labels).sum()
    
                test_correct += correct
                test_loss += loss.item()
                test_size += len(labels)

        print(f"Epoch: {epoch + 1:3d}/{num_epochs} | Train Loss : {train_loss / train_size:.4f} | Train Accuracy : {train_correct / train_size:.2f}",
                f"| Test Loss : {test_loss / test_size:.4f} | Test Accuracy : {test_correct / test_size:.2f}")
        

        # Log training loss
        logger.report_scalar(
            title="Loss",         # Category (group)
            series="Train",       # Series (line)
            value=train_loss,     # The actual metric value
            iteration=epoch       # X-axis (epoch or step)
        )

        # Log validation loss
        logger.report_scalar(
            title="Loss",
            series="Validation",
            value=test_loss,
            iteration=epoch
        )

       
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="creditcard_2023.csv")
args = parser.parse_args()

df = pd.read_csv(args.dataset)



drop_columns = ["id"]
df = df.drop(columns = drop_columns)

min_max_scaler = MinMaxScaler()
df["Amount"] = min_max_scaler.fit_transform(df["Amount"].values.reshape(-1 ,1))

X = df.drop(columns = "Class").values
y = df["Class"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = FraudDetectionModel().to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

train(num_epochs, criterion, optimizer, train_loader, test_loader, device)
model= model.to("cpu")
torch.save(model.state_dict(), "fraud_model.pth")
