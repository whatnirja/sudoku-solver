from train_digit_model import Net
from grid_detector import detect_and_warp_grid, split_into_cells 
import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn

def center_digit(cell_gray):
  _, thresh = cv2.threshold(cell_gray, 128, 255, cv2.THRESH_BINARY_INV)
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  if not contours:
    return np.zeros((28, 28), dtype=np.uint8)

  largest = max(contours, key=cv2.contourArea)
  x, y, w, h = cv2.boundingRect(largest)
  digit = cell_gray[y:y+h, x:x+w]

  scale = 20.0 / max(w, h)
  new_w, new_h = int(w * scale), int(h * scale)
  digit_resized = cv2.resize(digit, (new_w, new_h))

  canvas = np.zeros((28, 28), dtype=np.uint8)
  x_offset = (28 - new_w) // 2
  y_offset = (28 - new_h) // 2
  canvas[y_offset:y_offset+new_h, x_offset:x_offset + new_w] = digit_resized

  cv2.imshow("Cropped digit", digit)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  return canvas

model = Net()
model.load_state_dict(torch.load("digit_model.pth"))
model.eval()

image = cv2.imread('test2.jpg')
warped = detect_and_warp_grid(image)
cells = split_into_cells(warped)

cell = cells[0]
# resized = cv2.resize(cell, (28, 28))
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
gray_full = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
gray = center_digit(gray_full)

cv2.imshow("What the model sees", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
tensor = torch.from_numpy(gray)
tensor = tensor.float() / 255.0
tensor = tensor.unsqueeze(0).unsqueeze(0)

with torch.no_grad():
  output = model(tensor)
  predicted = output.argmax(dim=1).item()

print("Predicted:", predicted)  