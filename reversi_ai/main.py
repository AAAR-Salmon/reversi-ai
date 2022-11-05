#!/usr/bin/env python

import argparse

import requests as rq
import torch

from reversi_ai.ffn import FFN


def register_user(server_url: str, user_name: str):
    header = {"Content-Type": "application/json"}
    res = rq.post(
        f"{server_url}/users", headers=header, json={"user_name": user_name}
    )
    return res.json()


def main(model_path: str, server_url: str, user_id: str | None):
    ffn = FFN()

    # ======== モデルの読み込み ========
    ffn.load_state_dict(torch.load(model_path))
    ffn.eval()

    # ======== user（プレイヤー）登録 ========
    user_name = "FN-36K"
    if user_id is not None:
        user = {'id': user_id, 'name': user_name, 'status': None}
    else:
        user = register_user(server_url, user_name)


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
