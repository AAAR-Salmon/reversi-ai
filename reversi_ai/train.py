#!/usr/bin/env python
import numpy as np
import scipy.special
import torch
import torch.utils.data
from reversi_game.reversi import Color, Reversi
from torch import nn

from reversi_ai.ffn import FFN


def train_loop(model, dataloader, loss_fn, optimizer) -> None:
    # 学習用関数の定義
    for x, turn, y in dataloader:
        pred = model(x, turn)
        loss = loss_fn(pred, y)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    ffn = FFN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ffn.parameters())

    # TODO: ======== モデルの読み込み ========

    # ======== データセットの生成 ========
    n_games = 100
    n_max_hands = 60
    # allocation 回数削減のため予め確保
    train_i = 0
    train_x = np.empty((n_games * n_max_hands, 8 * 8), dtype=np.float32)
    train_turn = np.empty((n_games * n_max_hands,), dtype=np.float32)
    train_y = np.empty((n_games * n_max_hands, 8 * 8), dtype=np.float32)
    label_to_onehot = np.eye(8 * 8)

    # (Row, Column) 座標と flatten 後の添字の相互変換用のやつ
    rc_coord_to_flat_index = np.arange(8 * 8).reshape((8, 8))
    flat_coord_to_rc_index = np.mgrid[0:8, 0:8].reshape((2, -1))

    for game_i in range(n_games):
        print(game_i)
        reversi = Reversi(8, 8)
        history_i = 0
        history_x = np.empty((n_max_hands, 8 * 8), dtype=np.float32)
        history_turn = np.empty((n_max_hands,), dtype=np.int32)
        history_hand = np.empty((n_max_hands,), dtype=np.int32)

        while reversi.turn != Color.NONE:
            # 評価値を計算
            evaluation_value: torch.Tensor = ffn(
                torch.tensor(reversi.board, dtype=torch.float),
                torch.tensor([[reversi.turn]], dtype=torch.float),
            ).flatten()

            # 置ける座標のリスト
            placeable_index = list(
                map(
                    rc_coord_to_flat_index.item,
                    reversi.get_placeable_coords(reversi.turn),
                )
            )

            # 置ける座標のうち評価値が高い座標が高い確率で得られる
            probability = scipy.special.softmax(
                evaluation_value[placeable_index].detach().numpy()
            )
            chosen_index = np.random.choice(placeable_index, p=probability)
            chosen_row, chosen_column = flat_coord_to_rc_index[:, chosen_index]

            # 手番の仮保存
            history_x[history_i] = reversi.board.flatten()
            history_turn[history_i] = reversi.turn
            history_hand[history_i] = chosen_index
            history_i += 1

            reversi.place_disk(chosen_row, chosen_column, reversi.turn)
            reversi.turn = reversi.forward_turn()

        # 勝者の判定
        dark_count = np.count_nonzero(reversi.board == Color.DARK)
        light_count = np.count_nonzero(reversi.board == Color.LIGHT)
        winner = Color.NONE
        if dark_count == light_count:
            continue  # 引き分けは学習しても無意味なので破棄
        elif dark_count > light_count:
            winner = Color.DARK
        else:
            winner = Color.LIGHT

        # データセットへの格納
        train_x[train_i : train_i + history_i] = history_x  # noqa: E203
        train_turn[train_i : train_i + history_i] = history_turn  # noqa: E203
        train_y[train_i : train_i + history_i] = np.array(  # noqa: E203
            [
                label_to_onehot[hand]
                if turn == winner
                else -label_to_onehot[hand]
                for turn, hand in zip(history_turn, history_hand)
            ],
            dtype=np.float32,
        )
        train_i += history_i

    # データセットの縮小
    train_x.resize((train_i, 8 * 8))
    train_turn.resize((train_i,))
    train_y.resize((train_i, 8 * 8))

    # ======== DataLoader の設定 ========
    train_x = torch.tensor(train_x, dtype=torch.float)
    train_turn = torch.tensor(train_turn, dtype=torch.float)
    train_y = torch.tensor(train_y, dtype=torch.float)

    dataset = torch.utils.data.TensorDataset(train_x, train_turn, train_y)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=True
    )

    # TODO: ======== 重みの最適化 ========

    # TODO: ======== モデルの保存 ========
