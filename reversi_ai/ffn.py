import torch
from torch import nn


# FFN の定義
class FFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # モデルの重みの定義
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=5, padding=2)
        self.convt = nn.ConvTranspose2d(8, 1, kernel_size=5)
        self.fc = nn.Linear(65, 64)

    def forward(self, x: torch.Tensor, turn: torch.Tensor) -> torch.Tensor:
        # 入力から出力を計算する
        x = x.reshape(-1, 1, 8, 8)
        x = self.conv1(x)  # (1, 8, 8) -> (16, 4, 4)
        x = self.relu(x)
        x = self.conv2(x)  # (16, 4, 4) -> (8, 4, 4)
        x = self.relu(x)
        x = self.convt(x)  # (8, 4, 4) -> (1, 8, 8)
        x = self.relu(x)
        x = torch.cat(
            (self.flatten(x), turn.reshape(-1, 1)), dim=1
        )  # (1, 8, 8) + (1,) -> (65,)
        x = self.fc(x)  # (65,) -> (64,)
        return x
