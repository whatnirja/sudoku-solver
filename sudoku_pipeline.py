from train_digit_model import Net
from grid_detector import detect_and_warp_grid, split_into_cells 
import cv2
import numpy as np
import torch

def center_digit(cell_gray):
  _, thresh = cv2.threshold(cell_gray, 128, 255, cv2.THRESH_BINARY_INV)

  kernel = np.ones((2, 2), np.uint8)
  thresh = cv2.erode(thresh, kernel, iterations=1)

  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  for c in contours:
    area = cv2.contourArea(c)
    box = cv2.boundingRect(c)
    print("contour area: ", area)
    print("bounding box: ", box)

  if not contours:
    return None

  largest = max(contours, key=cv2.contourArea)
  x, y, w, h = cv2.boundingRect(largest)
  # digit = cell_gray[y:y+h, x:x+w]
  digit = thresh[y:y+h, x:x+w]

  scale = 20.0 / max(w, h)
  new_w, new_h = int(w * scale), int(h * scale)
  digit_resized = cv2.resize(digit, (new_w, new_h))

  # canvas = np.zeros((28, 28), dtype=np.uint8)
  # x_offset = (28 - new_w) // 2
  # y_offset = (28 - new_h) // 2
  # canvas[y_offset:y_offset+new_h, x_offset:x_offset + new_w] = digit_resized

  canvas = np.zeros((28, 28), dtype=np.uint8)
  x_offset = (28 - new_w) // 2
  y_offset = (28 - new_h) // 2
  canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized

  M = cv2.moments(canvas)
  if M["m00"] != 0:
      cx = int(M["m10"] / M["m00"])
      cy = int(M["m01"] / M["m00"])
      shift_x = 14 - cx
      shift_y = 14 - cy
      shift_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
      canvas = cv2.warpAffine(canvas, shift_matrix, (28, 28))

  # cv2.imshow("Cropped digit", digit)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

  return canvas

model = Net()
model.load_state_dict(torch.load("digit_model.pth"))
model.eval()

image = cv2.imread('test2.jpg')

print("Image shape:", image.shape)

warped = detect_and_warp_grid(image)
cells = split_into_cells(warped)

# cell = cells[14]
# gray_full = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
# gray = center_digit(gray_full)

# print("Corner pixel (background):", gray[0][0])
# print("Center pixel (should be digit):", gray[14][14])

# cv2.imshow("What the model sees", gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# tensor = torch.from_numpy(gray)
# tensor = tensor.float() / 255.0
# tensor = tensor.unsqueeze(0).unsqueeze(0)

# print("Tensor min:", tensor.min().item(), "max:",
#  tensor.max().item(), "mean:", tensor.mean().item())


# with torch.no_grad():
#   output = model(tensor)
#   predicted = output.argmax(dim=1).item()

# print("Predicted:", predicted)  

test_indices = range(81)

for idx in test_indices:
    cell = cells[idx]
    gray_full = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    gray = center_digit(gray_full)

    if gray is None:
        print(f"Cell {idx}: blank (0)")
        continue

    tensor = torch.from_numpy(gray)
    tensor = tensor.float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        predicted = output.argmax(dim=1).item()

    print(f"Cell {idx}: Predicted {predicted}")