#!/usr/bin/env python

import argparse
import time

import numpy as np
import requests as rq
import scipy
import torch
from reversi_game.color import Color
from reversi_game.reversi import Reversi

from reversi_ai.ffn import FFN


def register_user(server_url: str, user_name: str):
    header = {"Content-Type": "application/json"}
    res = rq.post(
        f"{server_url}/users", headers=header, json={"user_name": user_name}
    )
    return res.json()


def wait_assigned_room(server_url: str, user_id: str):
    while True:
        res = rq.get(f"{server_url}/users/{user_id}")
        status = res.json()["status"]
        if status is not None:
            return status
        time.sleep(1)


def get_user_color(server_url: str, room_id: str, user_id: str):
    room = rq.get(f"{server_url}/rooms/{room_id}").json()
    return Color.DARK if user_id == room["black"]["id"] else Color.LIGHT


def wait_turn_or_finish(server_url: str, user_id: str, room_id: str):
    while True:
        res = rq.get(f"{server_url}/rooms/{room_id}").json()
        if res["next"] is None or res["next"]["id"] == user_id:
            return res["next"], res["board"]
        time.sleep(1)


def decide_hand(
    model, board: np.ndarray[np.float32], turn: Color, number_of_option: int
):
    reversi = Reversi(8, 8)

    reversi.board = board
    coords_placeable = np.array(
        reversi.get_placeable_coords(reversi.turn), np.int32
    )
    indices_placeable = np.ravel_multi_index(coords_placeable.T, (8, 8))

    evaluation_value: torch.Tensor = model(
        torch.tensor(board, dtype=torch.float32),
        torch.tensor([[turn]], dtype=torch.float32),
    ).flatten()

    evaluation_value = evaluation_value.detach().numpy()[indices_placeable]
    indices_high_eval = evaluation_value.argsort()[-number_of_option:]

    probability = scipy.special.softmax(evaluation_value[indices_high_eval])
    chosen_index = np.random.default_rng().choice(
        indices_placeable[indices_high_eval], p=probability
    )
    chosen_row, chosen_column = np.unravel_index(chosen_index, (8, 8))
    return chosen_row, chosen_column


def place_disk(
    server_url: str, room_id: str, user_id: str, row: int, column: int
):
    rq.post(
        f"{server_url}/rooms/{room_id}",
        json={"user_id": user_id, "row": row, "column": column},
    )


def main(
    model_path: str,
    server_url: str,
    user_id: str | None,
    number_of_option: int,
):
    ffn = FFN()

    # ======== モデルの読み込み ========
    ffn.load_state_dict(torch.load(model_path))
    ffn.eval()

    # ======== user（プレイヤー）登録 ========
    user_name = "Salmon v2"
    if user_id is None:
        user_id = register_user(server_url, user_name)["id"]

    # ======== main loop ========
    while True:
        room_id = wait_assigned_room(server_url, user_id)
        color = get_user_color(server_url, room_id, user_id)
        while True:
            next_user, board = wait_turn_or_finish(
                server_url, user_id, room_id
            )
            if next_user is None:
                break
            row, column = map(
                int,
                decide_hand(
                    ffn,
                    np.array(board, dtype=np.float32),
                    color,
                    number_of_option,
                ),
            )
            place_disk(server_url, room_id, user_id, row, column)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTTP reversi AI client")
    parser.add_argument(
        "model", help="saved model to use (pickle-format, state_dict)"
    )
    parser.add_argument(
        "server_url", help="server url e.g. http://127.0.0.1:8000"
    )
    parser.add_argument(
        "-u", "--user-id", metavar="USER_ID", help="user_id of client"
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
    main(args.model, args.server_url, args.user_id, args.number_of_option)
