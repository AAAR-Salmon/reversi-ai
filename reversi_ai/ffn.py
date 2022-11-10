import torch
from torch import nn


# FFN の定義
class FFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # モデルの重みの定義
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 95)
        self.fc2 = nn.Linear(96, 64)

    def forward(self, x: torch.Tensor, turn: torch.Tensor) -> torch.Tensor:
        # 入力から出力を計算する
        x = x.reshape(-1, 64)
        x = self.fc1(x)
        x = self.relu(x)
        x = torch.cat(
            (self.flatten(x), turn.reshape(-1, 1)), dim=1
        )  # (95,) + (1,) -> (96,)
        x = self.fc2(x)
        return x
