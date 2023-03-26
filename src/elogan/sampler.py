import random
import io

import chess.pgn
import torch
from torch import nn
import numpy as np

from data_creation import get_data
from utils import preprocess


df, train_idxs, val_idxs, test_idxs = get_data()


def get_generated_samples(num_samples, generator):
  generated_samples = []

  for i in range(num_samples):
    while True:
      try:
        data_idx = train_idxs[random.randint(0, len(train_idxs))]
        elo = int((df["white_rating"][data_idx] + df["black_rating"][data_idx]) / 2)

        game_pgn = df["pgn"][data_idx]
        game = chess.pgn.read_game(io.StringIO(game_pgn))
        board = game.board()

        moves = [m for m in game.mainline_moves()]
        num_moves = len(moves)
        rand_pos_idx = random.randint(0, num_moves - 1) # -1 because we don't want the end position
      except Exception:
        continue
      break

    for i, move in enumerate(moves[:rand_pos_idx]):
      board.push(move)

    scores = []
    possible_moves = [m for m in board.legal_moves]

    for move in possible_moves:
      board.push(move)
      state = preprocess(board)
      state = np.expand_dims(state, axis=0)
      state = torch.from_numpy(state).float()
      scores.append(generator(state))
      board.pop()
    
    pred_move = possible_moves[scores.index(max(scores))]

    board_after_move = board.copy()
    board_after_move.push(pred_move)

    sample = np.concatenate([[elo], preprocess(board), preprocess(board_after_move)])
    sample = torch.from_numpy(sample).float()

    generated_samples.append(sample)
  
  return torch.stack(generated_samples)


def get_real_samples(num_samples):
  real_samples = []

  for i in range(num_samples):
    while True:
      try:
        data_idx = train_idxs[random.randint(0, len(train_idxs))]
        elo = int((df["white_rating"][data_idx] + df["black_rating"][data_idx]) / 2)

        game_pgn = df["pgn"][data_idx]
        game = chess.pgn.read_game(io.StringIO(game_pgn))
        board = game.board()

        moves = [m for m in game.mainline_moves()]
        num_moves = len(moves)
        rand_pos_idx = random.randint(0, num_moves - 1) # -1 because we don't want the end position
      except Exception:
        continue
      break

    for i, move in enumerate(moves[:rand_pos_idx]):
      board.push(move)
    
    board_after_move = board.copy()
    board_after_move.push(moves[rand_pos_idx])

    sample = np.concatenate([[elo], preprocess(board), preprocess(board_after_move)])
    sample = torch.from_numpy(sample).float()

    real_samples.append(sample)
  
  return torch.stack(real_samples)
