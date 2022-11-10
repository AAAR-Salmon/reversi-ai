#!/usr/bin/env python
import argparse
import datetime
import itertools

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


def show_progress(
    title: str,
    n_finished: int,
    n_target: int,
    length: int = 20,
    done: bool = False,
):
    l_finished = n_finished * length // n_target
    if done:
        print(
            f"\r{title} |{'='*(length+1)}|",
            f"{n_target} / {n_target} (100.0%)",
            end="\n",
        )
    else:
        print(
            f"\r{title} |{'='*l_finished}>{' '*(length-l_finished)}|",
            f"{n_finished} / {n_target} ({n_finished/n_target*100:.1f}%)",
            end="",
        )


def main(state_dict_path: str = None, number_of_option: int = 3):
    ffn = FFN()
    rng = np.random.default_rng()

    # ======== モデルの読み込み ========
    if state_dict_path is not None:
        ffn.load_state_dict(torch.load(state_dict_path))
    ffn.train()

    for epoch in itertools.count():
        print(f"epoch {epoch}")
        # ======== データセットの生成 ========
        n_games = 100
        n_max_hands = 60
        # allocation 回数削減のため予め確保
        train_i = 0
        train_x = np.empty((n_games * n_max_hands, 8 * 8), dtype=np.float32)
        train_turn = np.empty((n_games * n_max_hands,), dtype=np.float32)
        train_label = np.empty((n_games * n_max_hands,), dtype=np.int32)

        for game_i in range(n_games):
            show_progress("game", game_i, n_games)
            reversi = Reversi(8, 8)
            history_i = 0
            history_x = np.empty((n_max_hands, 8 * 8), dtype=np.float32)
            history_turn = np.empty((n_max_hands,), dtype=np.int32)
            history_hand = np.empty((n_max_hands,), dtype=np.int32)

            while reversi.turn != Color.NONE:
                # 置ける座標（flattened）
                coords_placeable = np.array(
                    reversi.get_placeable_coords(reversi.turn), np.int32
                )
                indices_placeable = np.ravel_multi_index(
                    coords_placeable.T, (8, 8)
                )

                # 評価値を計算
                evaluation_value: torch.Tensor = ffn(
                    torch.tensor(reversi.board, dtype=torch.float32),
                    torch.tensor([[reversi.turn]], dtype=torch.float32),
                ).flatten()

                # 置ける座標のうち、評価値が高い順に number_of_option 個に絞り込む
                evaluation_value = evaluation_value.detach().numpy()[
                    indices_placeable
                ]
                indices_high_eval = evaluation_value.argsort()[
                    -number_of_option:
                ]

                # 評価値が高い座標が高い確率で得られる
                probability = scipy.special.softmax(
                    evaluation_value[indices_high_eval]
                )
                chosen_index = rng.choice(
                    indices_placeable[indices_high_eval], p=probability
                )
                chosen_row, chosen_column = np.unravel_index(
                    chosen_index, (8, 8)
                )

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

            # 未使用領域をカットした view を作成
            history_x = history_x[:history_i]
            history_turn = history_turn[:history_i]
            history_hand = history_hand[:history_i]

            # 勝った側の手だけ残すようフィルタ
            history_x = history_x[history_turn == winner]
            history_hand = history_hand[history_turn == winner]
            n_hand_to_train = np.count_nonzero(history_turn == winner)

            # データセットへの格納
            train_x[
                train_i : train_i + n_hand_to_train  # noqa: E203
            ] = history_x
            train_turn[
                train_i : train_i + n_hand_to_train  # noqa: E203
            ] = winner
            train_label[
                train_i : train_i + n_hand_to_train  # noqa: E203
            ] = history_hand
            train_i += n_hand_to_train

        # データセットの縮小
        train_x.resize((train_i, 8 * 8))
        train_turn.resize((train_i,))
        train_label.resize((train_i,))

        show_progress("game", n_games, n_games, done=True)

        # ======== DataLoader の設定 ========
        train_x = torch.tensor(train_x, dtype=torch.float32)
        train_turn = torch.tensor(train_turn, dtype=torch.float32)
        train_label = torch.tensor(train_label, dtype=torch.int64)

        dataset = torch.utils.data.TensorDataset(
            train_x, train_turn, train_label
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=100, shuffle=True
        )

        # ======== 重みの最適化 ========
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(ffn.parameters())
        train_loop(ffn, data_loader, loss_fn, optimizer)

        # ======== モデルの保存 ========
        timestamp = datetime.datetime.utcnow().timestamp()
        torch.save(
            ffn.state_dict(),
            f"reversi_ffn_model-v2-{timestamp}.pickle",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unsupervised reversi AI trainer"
    )
    parser.add_argument(
        "-r",
        "--resume",
        help="resume training from saved model (pickle-format, state_dict)",
        required=False,
        metavar="FILE",
    )
    parser.add_argument(
        "-n",
        "--number-of-option",
        help="narrows options of choice to N",
        type=int,
        metavar="N",
        required=False,
        default=3,
    )
    args = parser.parse_args()
    main(state_dict_path=args.resume, number_of_option=args.number_of_option)
