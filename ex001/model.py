import torch
import torch.nn as nn


class MCLR(nn.Module):
    """
    L2正則化付き多項ロジスティック回帰モデル (Multinomial Logistic Regression)
    強凸設定 (Convex setting) で使用される線形分類器 (入力784次元 -> 出力10クラス, パラメータ数 d = 7,850)。
    """

    def __init__(self, input_dim: int = 784, num_classes: int = 10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.linear(x)


def get_flat_params(model: nn.Module) -> torch.Tensor:
    """モデルの全パラメータを1次元ベクトルとして取得する。"""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    """1次元ベクトルからモデルの全パラメータを設定する。"""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset : offset + numel].view_as(p))
        offset += numel
