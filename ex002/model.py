from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompactCNN(nn.Module):
    """
    論文 Table 1 (MNIST FOR CNN, 200ラウンド通信量 9,968,000 B -> パラメータ数 d = 6,230) に準拠したCNNモデル。
    Conv1(1->5, 5x5): 130
    Conv2(5->5, 5x5): 630
    FC1(80->60): 4,860
    FC2(60->10): 610
    合計: 6,230 パラメータ
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=5)
        self.fc1 = nn.Linear(5 * 4 * 4, 60)
        self.fc2 = nn.Linear(60, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


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


def get_flat_grads(model: nn.Module) -> torch.Tensor:
    """モデルの全勾配を1次元ベクトルとして取得する。"""
    return torch.cat([p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1) for p in model.parameters()])


def batched_cnn_local_sgd(
    x: torch.Tensor,
    packed_clients: dict,
    tau: int,
    batch_size: int,
    lr: float,
    fixed_batches: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    CompactCNN (d=6,230) において全クライアント (N=20) の tau ステップ局所SGDを
    F.conv2d(..., groups=N) + torch.bmm を用いて GPU 上で一括並列計算し、
    サーバ集約モデル v_k = sum_i p_i * x_{k, tau}^{(i)} を返す。
    """
    N = int(packed_clients["num_clients"])
    X_pad = packed_clients["X_pad"]      # (N, max_n, 1, 28, 28)
    y_pad = packed_clients["y_pad"]      # (N, max_n)
    n_vec = packed_clients["n_vec"]      # (N,)
    weights = packed_clients["weights"]  # (N,)

    xi = x.detach().unsqueeze(0).expand(N, -1).clone().requires_grad_(True)
    client_ids = torch.arange(N, device=x.device).unsqueeze(1)

    for j in range(tau):
        with torch.no_grad():
            if fixed_batches is None:
                idx = (torch.rand((N, batch_size), device=x.device) * n_vec.unsqueeze(1)).long()
                xb = X_pad[client_ids, idx]  # (N, bs, 1, 28, 28)
                yb = y_pad[client_ids, idx]  # (N, bs)
            else:
                xb = fixed_batches[0][j]
                yb = fixed_batches[1][j]
            bs = xb.size(1)
            x_in = xb.permute(1, 0, 2, 3, 4).reshape(bs, N, 28, 28)

        w1 = xi[:, 0:125].reshape(N * 5, 1, 5, 5)
        b1 = xi[:, 125:130].reshape(N * 5)
        h1 = F.relu(F.max_pool2d(F.conv2d(x_in, w1, b1, groups=N), 2))

        w2 = xi[:, 130:755].reshape(N * 5, 5, 5, 5)
        b2 = xi[:, 755:760].reshape(N * 5)
        h2 = F.relu(F.max_pool2d(F.conv2d(h1, w2, b2, groups=N), 2))  # (bs, N*5, 4, 4)

        h2_flat = h2.view(bs, N, 80).permute(1, 0, 2)  # (N, bs, 80)
        w3 = xi[:, 760:5560].reshape(N, 60, 80)
        b3 = xi[:, 5560:5620].reshape(N, 1, 60)
        h3 = F.relu(torch.bmm(h2_flat, w3.transpose(1, 2)) + b3)  # (N, bs, 60)

        w4 = xi[:, 5620:6220].reshape(N, 10, 60)
        b4 = xi[:, 6220:6230].reshape(N, 1, 10)
        logits = torch.bmm(h3, w4.transpose(1, 2)) + b4  # (N, bs, 10)

        loss_sum = F.cross_entropy(logits.reshape(N * bs, 10), yb.reshape(N * bs), reduction="sum") / float(bs)
        (grad,) = torch.autograd.grad(loss_sum, xi)

        with torch.no_grad():
            xi = (xi - lr * grad).detach().requires_grad_(True)

    with torch.no_grad():
        return torch.sum(weights.unsqueeze(1) * xi.detach(), dim=0)
