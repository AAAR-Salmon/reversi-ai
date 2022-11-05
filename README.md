# reversi-ai

[PyTorch](https://github.com/pytorch/pytorch) によって実装された
畳み込みニューラルネットワークによるリバーシAIの自己教師あり学習と、
[Wansuko-cmd/reversi-server](https://github.com/Wansuko-cmd/reversi-server)
のためのクライアント。

## 利用方法

[Poetry](https://github.com/python-poetry/poetry)
によって依存関係を管理している。

```sh
poetry install
```

で依存パッケージをインストールし、venv を構成する。

### 学習

```sh
poetry run ./reversi_ai/train.py
```

で学習を開始し、モデルの state_dict をカレントディレクトリに保存する。

### クライアントの実行

サーバが `http://192.168.x.y:8000` で実行されており、
保存した state_dict がカレントディレクトリの
`reversi_ffn_model-xxxxxxxxxx.yyyyyy.pickle` にあるなら、

```sh
poetry run ./reversi_ai/main.py reversi_ffn_model-xxxxxxxxxx.yyyyyy.pickle http://192.168.x.y:8000
```

で実行できる。

何らかの原因でクライアントが停止した場合は、
`-u/--user-id` オプションにより前回のユーザIDを使うことができる。

前回のユーザIDが `123e4567-e89b-12d3-a456-426614174000` ならば、

```sh
poetry run ./reversi_ai/main.py --user-id 123e4567-e89b-12d3-a456-426614174000 reversi_ffn_model-xxxxxxxxxx.yyyyyy.pickle http://192.168.x.y:8000
```

## ライセンス

[Apache License Version 2.0](./LICENSE) に基づく。

## 動作環境

* Windows 10 21H2 19044.2130
* [Windows Subsystem for Linux](https://learn.microsoft.com/ja-jp/windows/wsl/install) 2
* CPython 3.10.8
* Poetry v1.2.2
