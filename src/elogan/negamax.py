import torch
import numpy as np

from utils import preprocess


def negamax(board, generator, depth=0):
  if depth == 3:
    state = preprocess(board)
    state = np.expand_dims(state, axis=0)
    state = torch.from_numpy(state).float()
    x = generator(state)
    return x
  
  scores = []
  possible_moves = [m for m in board.legal_moves]
  for move in possible_moves:
    board.push(move)
    scores.append(negamax(board.copy(), depth+1))
    board.pop()
  if depth == 0:
    return possible_moves[scores.index(max(scores))]
  else:
    return max(scores)
