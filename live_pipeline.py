import copy
import cv2
import numpy as np
import torch
from grid_detector import order_corners, detect_and_warp_grid, split_into_cells
from sudoku_pipeline import recognize_board
from solver import solve, is_board_valid, combine_boards
from train_digit_model import Net
from board_repair import repair_board

cap = cv2.VideoCapture(0)
counter = 0
solved = None

model = Net()
model.load_state_dict(torch.load("digit_model.pth"))
model.eval()

boards_collected = []
confidence_boards_collected = []

last_good_matrix = None
frames_since_good_matrix = 0
MAX_GRACE_FRAMES = 10

while True:
  ret, frame = cap.read()
  if not ret:
    break

  warped, grid_contour, matrix = detect_and_warp_grid(frame)

  # if warped is not None:
  #   counter += 1
  #   if counter >= 35 and len(boards_collected) < 5:
  #     cells = split_into_cells(warped)
  #     board, confidence_board = recognize_board(cells, model)
  #     boards_collected.append(board)
  #     confidence_boards_collected.append(confidence_board)
  #     print("Boards collected so far:", len(boards_collected))
  # else:
  #   counter = 0
  #   solved = None
  #   boards_collected = []
  #   confidence_boards_collected = []
  if warped is not None:
    counter += 1
    last_good_matrix = matrix
    frames_since_good_matrix = 0
    if counter >= 35 and solved is None:
      cells = split_into_cells(warped)
      board, confidence_board = recognize_board(cells, model)
      if is_board_valid(board):

        original_board = copy.deepcopy(board)

        if solve(board):
          solved = board
          print("Solved:", board)
          canvas = np.zeros((450, 450, 3), dtype=np.uint8)
          cell_size = 450 // 9

          for row in range(9):
            for col in range(9):
              if original_board[row][col] == 0:
                digit = str(solved[row][col])
                x = col * cell_size + 15
                y = row * cell_size + 35
                cv2.putText(canvas, digit, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
  else:
    counter = 0
    frames_since_good_matrix += 1

  if solved is not None and frames_since_good_matrix < MAX_GRACE_FRAMES:
    use_matrix = matrix if warped is not None else last_good_matrix
    inverse_matrix = np.linalg.inv(use_matrix)
    warped_digits = cv2.warpPerspective(canvas, inverse_matrix, (frame.shape[1], frame.shape[0])) # frame shape give width and height resp to match the live frame's actual size
    gray_digits = cv2.cvtColor(warped_digits, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_digits, 1, 255, cv2.THRESH_BINARY)

    frame[mask == 255] = warped_digits[mask == 255]

    # cv2.imshow("warped_digita", warped_digits)
    # cv2.waitKey(0)
    print("=" * 20)
    print("solved!!")
    for row in solved:
      print(row)
    print("=" * 20)
    # break
  # else:
  #   print("Invalid single frame:", board)

  # if len(boards_collected) == 5 and solved is None:
  #   combined = combine_boards(boards_collected)
  #   combined_confidence = combine_boards(confidence_boards_collected)

  #   print("Combined board:", combined)
  #   print("Combined confidence board:", combined_confidence)
  #   if is_board_valid(combined):
  #     solve(combined)
  #     solved = combined
  #   else:
  #     repaired, fixed = repair_board(combined, combined_confidence)
  #     if fixed:
  #       solve(repaired)
  #       solved = repaired
  #     else:
  #       boards_collected = []
  #       confidence_boards_collected = []

  # print("Counter:", counter)

  if warped is None:
    cv2.imshow("feed", frame)
  else:
    cv2.drawContours(frame, [grid_contour], -1, (0, 255, 0), 3)
    cv2.imshow("feed", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cv2.destroyAllWindows()