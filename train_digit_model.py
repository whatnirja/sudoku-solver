import torch
import torchvision
import torch.nn as nn
from torchvision import transforms

transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
  root="./data", 
  train=True, 
  download=True, 
  transform=transform
)
test_dataset = torchvision.datasets.MNIST(
  root="./data",
  train=False,
  download=True,
  transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Training samples:", len(train_dataset))
print("Test samples:", len(test_dataset))

class Net(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
    self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

    self.pool = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(1568, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = torch.relu(self.conv1(x))
    x = self.pool(x)

    x = torch.relu(self.conv2(x))
    x = self.pool(x)

    x = x.view(x.size(0), -1)

    x = torch.relu(self.fc1(x))
    x = self.fc2(x)

    return x

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)