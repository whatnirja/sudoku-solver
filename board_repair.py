from solver import is_valid

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