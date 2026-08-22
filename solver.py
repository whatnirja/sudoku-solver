def solve(board):
  if find_empty(board) != None:
    row, col = find_empty(board)
    for num in range(1, 10):
      if is_valid(board, row, col, num):
        board[row][col] = num
        if solve(board):
          return True
        else:
          board[row][col] = 0
    return False
  else:
    return True


def is_valid(board, row, col, num):
  for i in range(9):
    if board[row][i] == num:
      return False

  for i in range(9):
    if board[i][col] == num:
      return False

# dont get it
  bow_row_start = (row // 3) * 3
  bow_col_start = (col // 3) * 3

  for i in range(3):
    for j in range(3):
      if board[bow_row_start + i][bow_col_start + j] == num:
        return False
  
  return True

def find_empty(board):
  for row in range(9):
    for col in range(9):
      if board[row][col] == 0:
        return row, col
  return None