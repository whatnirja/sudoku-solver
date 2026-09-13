import cv2
import torch
from grid_detector import order_corners, detect_and_warp_grid, split_into_cells
from sudoku_pipeline import recognize_board
from solver import solve, is_board_valid, combine_boards
from train_digit_model import Net

cap = cv2.VideoCapture(0)
counter = 0
solved = None

model = Net()
model.load_state_dict(torch.load("digit_model.pth"))
model.eval()

boards_collected = []

while True:
  ret, frame = cap.read()
  if not ret:
    break

  warped, grid_contour = detect_and_warp_grid(frame)

  if warped is not None:
    counter += 1
    if counter >= 35 and len(boards_collected) < 5:
      cells = split_into_cells(warped)
      board = recognize_board(cells, model)
      boards_collected.append(board)
      print("Boards collected so far:", len(boards_collected))
  else:
      counter = 0
      solved = None
      boards_collected = []

  if len(boards_collected) == 5 and solved is None:
    combined = combine_boards(boards_collected)
    print("Combined board:", combined)
    if is_board_valid(combined):
        solve(combined)
        solved = combined
    else:
        print("Combined board is invalid")
        boards_collected = []
  

  print("Counter:", counter)

  if warped is None:
    cv2.imshow("feed", frame)
  else: 
    cv2.drawContours(frame, [grid_contour], -1, (0, 255, 0), 3)
    cv2.imshow("feed", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cv2.destroyAllWindows()


