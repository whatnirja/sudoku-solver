from train_digit_model import Net
import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn

model = Net()
model.load_state_dict(torch.load("digit_model.pth"))
model.eval()

