import random

import chess.pgn
import pandas as pd


def get_data():
    df = pd.read_csv("../club_games_data.csv")

    df = df[df['rules']=='chess']
    df.drop('rules',axis=1,inplace=True)

    shuffled_idxs = list(range(len(df["pgn"])))
    random.shuffle(shuffled_idxs)
    train_idxs = shuffled_idxs[ : int(len(shuffled_idxs) * 0.70)]
    val_idxs = shuffled_idxs[int(len(shuffled_idxs) * 0.70) : int(len(shuffled_idxs) * 0.85)]
    test_idxs = shuffled_idxs[int(len(shuffled_idxs) * 0.85) : ]

    print(len(train_idxs))
    print(len(val_idxs))
    print(len(test_idxs))

    return df, train_idxs, val_idxs, test_idxs
