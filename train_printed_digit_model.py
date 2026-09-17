import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from train_digit_model import Net

if __name__ == "__main__":
    augment = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor()
    ])

    # transform = transforms.Compose([
    #     transforms.Grayscale(),
    #     transforms.ToTensor()
    # ])

    # full_dataset = torchvision.datasets.ImageFolder("printed_digits", transform=transform)

    mnist_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=augment)
    printed_dataset = torchvision.datasets.ImageFolder("printed_digits", transform=augment)

    combined_dataset = torch.utils.data.ConcatDataset([
        mnist_dataset,
        printed_dataset
    ])

    train_size = int(0.8 * len(combined_dataset))
    test_size = len(combined_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(combined_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f"Test accuracy: {correct/total*100:.2f}%")

    torch.save(model.state_dict(), "digit_model.pth")
    print("Model saved to digit_model.pth")