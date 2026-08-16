import torch
import torchvision
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