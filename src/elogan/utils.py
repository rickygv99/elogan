import numpy as np

square_to_idx = {
    ".": 0,
    "r": 1,
    "n": 2,
    "b": 3,
    "q": 4,
    "k": 5,
    "p": 6,
    "R": 7,
    "N": 8,
    "B": 9,
    "Q": 10,
    "K": 11,
    "P": 12,
}

def preprocess(board):
  board_as_list = str(board).split()
  board_as_int_list = [square_to_idx[square] for square in board_as_list]
  board_np = np.array(board_as_int_list)
  return board_np

