from train_digit_model import Net
import torch 
import torchvision
from torchvision import transforms
import torch.nn as nn

model = Net()
model.load_state_dict(torch.load("digit_model.pth"))
model.eval()

test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
image, label = test_dataset[0]

image = image.unsqueeze(0)

with torch.no_grad():
  output = model(image)
  predicted = output.argmax(dim=1).item()

print("True label:", label)
print("Predicted:", predicted)