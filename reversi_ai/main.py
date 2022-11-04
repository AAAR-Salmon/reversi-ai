#!/usr/bin/env python

import argparse

import torch

from reversi_ai.ffn import FFN


def main(model_path: str):
    ffn = FFN()

    # ======== モデルの読み込み ========
    ffn.load_state_dict(torch.load(model_path))
    ffn.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTTP reversi AI client")
    parser.add_argument(
        "model", help="saved model to use (pickle-format, state_dict)"
    )
    args = parser.parse_args()
    main(args.model)
