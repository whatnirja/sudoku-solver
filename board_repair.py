from train_digit_model import Net
from grid_detector import detect_and_warp_grid, split_into_cells
from sudoku_pipeline import recognize_board
from solver import is_valid, is_board_valid, solve, print_board
import torch
import cv2

def find_conflicts(board):
  conflicts = []

  for row in range(9):
    for col in range(9):
      if board[row][col] != 0:
        num = board[row][col]
        board[row][col] = 0
        if not is_valid(board, row, col, num):
          conflicts.append((row, col))
        board[row][col] = num 

  return conflicts

def repair_board(board, confidence_board):
  conflicts = find_conflicts(board)
  print("Conflicts found:", conflicts)

  for (row, col) in conflicts:
      original = board[row][col]
      alternative = confidence_board[row][col]
      print(f"Trying ({row},{col}): {original} -> {alternative}")

      board[row][col] = 0

      if is_valid(board, row, col, alternative):
          board[row][col] = alternative
          print("  Kept swap")
      else:
          board[row][col] = original
          print("  Reverted swap")

  solved = is_board_valid(board)
  return board, solved

if __name__ == "__main__":
  image = cv2.imread('test2.jpg')
  model = Net()
  model.load_state_dict(torch.load("digit_model.pth"))
  model.eval()

  warped, grid_contour, matrix = detect_and_warp_grid(image)
  cells = split_into_cells(warped)

  board, confidence_board = recognize_board(cells, model)
  print_board(board)

  repaired_board, solved = repair_board(board, confidence_board)
  print("Repaired, board valid:", solved)
  print_board(repaired_board)

  if solved:
    fully_solved = solve(repaired_board)
    print("Fully solved:", fully_solved)
    print_board(repaired_board)