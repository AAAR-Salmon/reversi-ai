#!/usr/bin/env python

import argparse


def main():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTTP reversi AI client")
    parser.add_argument(
        "model", help="saved model to use (pickle-format, state_dict)"
    )
    args = parser.parse_args()
    main()
