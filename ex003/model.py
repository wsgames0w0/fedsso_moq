from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MCLR(nn.Module):
    """L2正則化付き多項ロジスティック回帰モデル (d = 7,850)。"""

    def __init__(self, input_dim: int = 784, num_classes: int = 10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.linear(x)


class CompactCNN(nn.Module):
    """論文 Table 1 準拠の CompactCNN モデル (d = 6,230)。"""

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
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset : offset + numel].view_as(p))
        offset += numel


def batched_mclr_local_sgd(
    x: torch.Tensor,
    packed_clients: dict,
    tau: int,
    batch_size: int,
    lr: float,
    l2_reg: float = 1e-4,
) -> torch.Tensor:
    """MCLR 全クライアント並列 tau ステップ SGD。"""
    N = int(packed_clients["num_clients"])
    X_pad = packed_clients["X_pad"]
    y_pad = packed_clients["y_pad"]
    n_vec = packed_clients["n_vec"]
    weights = packed_clients["weights"]

    xi = x.unsqueeze(0).expand(N, -1).clone()
    client_ids = torch.arange(N, device=x.device).unsqueeze(1)

    for _ in range(tau):
        idx = (torch.rand((N, batch_size), device=x.device) * n_vec.unsqueeze(1)).long()
        xb = X_pad[client_ids, idx]
        yb = y_pad[client_ids, idx]

        w = xi[:, :7840].view(N, 10, 784)
        b = xi[:, 7840:].view(N, 1, 10)
        logits = torch.bmm(xb, w.transpose(1, 2)) + b
        probs = torch.softmax(logits, dim=2)
        probs.scatter_add_(2, yb.unsqueeze(-1), torch.full((N, batch_size, 1), -1.0, device=x.device, dtype=probs.dtype))
        probs.div_(float(batch_size))

        grad_w = torch.bmm(probs.transpose(1, 2), xb) + l2_reg * w
        grad_b = probs.sum(dim=1) + l2_reg * b.squeeze(1)
        grad = torch.cat([grad_w.reshape(N, 7840), grad_b], dim=1)
        xi.sub_(lr * grad)

    return torch.sum(weights.unsqueeze(1) * xi, dim=0)


def batched_cnn_local_sgd(
    x: torch.Tensor,
    packed_clients: dict,
    tau: int,
    batch_size: int,
    lr: float,
) -> torch.Tensor:
    """CompactCNN 全クライアント並列 tau ステップ SGD。"""
    N = int(packed_clients["num_clients"])
    X_pad = packed_clients["X_pad"]
    y_pad = packed_clients["y_pad"]
    n_vec = packed_clients["n_vec"]
    weights = packed_clients["weights"]

    xi = x.detach().unsqueeze(0).expand(N, -1).clone().requires_grad_(True)
    client_ids = torch.arange(N, device=x.device).unsqueeze(1)

    for _ in range(tau):
        with torch.no_grad():
            idx = (torch.rand((N, batch_size), device=x.device) * n_vec.unsqueeze(1)).long()
            xb = X_pad[client_ids, idx]
            yb = y_pad[client_ids, idx]
            x_in = xb.permute(1, 0, 2, 3, 4).reshape(batch_size, N, 28, 28)

        w1 = xi[:, 0:125].reshape(N * 5, 1, 5, 5)
        b1 = xi[:, 125:130].reshape(N * 5)
        h1 = F.relu(F.max_pool2d(F.conv2d(x_in, w1, b1, groups=N), 2))

        w2 = xi[:, 130:755].reshape(N * 5, 5, 5, 5)
        b2 = xi[:, 755:760].reshape(N * 5)
        h2 = F.relu(F.max_pool2d(F.conv2d(h1, w2, b2, groups=N), 2))

        h2_flat = h2.view(batch_size, N, 80).permute(1, 0, 2)
        w3 = xi[:, 760:5560].reshape(N, 60, 80)
        b3 = xi[:, 5560:5620].reshape(N, 1, 60)
        h3 = F.relu(torch.bmm(h2_flat, w3.transpose(1, 2)) + b3)

        w4 = xi[:, 5620:6220].reshape(N, 10, 60)
        b4 = xi[:, 6220:6230].reshape(N, 1, 10)
        logits = torch.bmm(h3, w4.transpose(1, 2)) + b4

        loss_sum = F.cross_entropy(logits.reshape(N * batch_size, 10), yb.reshape(N * batch_size), reduction="sum") / float(batch_size)
        (grad,) = torch.autograd.grad(loss_sum, xi)

        with torch.no_grad():
            xi = (xi - lr * grad).detach().requires_grad_(True)

    with torch.no_grad():
        return torch.sum(weights.unsqueeze(1) * xi.detach(), dim=0)
