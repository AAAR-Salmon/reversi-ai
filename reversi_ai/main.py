#!/usr/bin/env python

import argparse
import time

import requests as rq
import torch
from reversi_game.color import Color

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


def main(model_path: str, server_url: str, user_id: str | None):
    ffn = FFN()

    # ======== モデルの読み込み ========
    ffn.load_state_dict(torch.load(model_path))
    ffn.eval()

    # ======== user（プレイヤー）登録 ========
    user_name = "FN-36K"
    if user_id is None:
        user_id = register_user(server_url, user_name)["id"]

    # ======== main loop ========
    while True:
        room_id = wait_assigned_room(server_url, user_id)
        color = get_user_color(server_url, room_id, user_id)


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
    args = parser.parse_args()
    main(args.model, args.server_url, args.user_id)
