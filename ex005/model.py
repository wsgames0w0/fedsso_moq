from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompactCNN(nn.Module):
    """
    論文 Table 1 (EMNIST FOR CNN, 200ラウンド通信量 9,968,000 B -> パラメータ数 d = 6,230) 準拠モデル。
    Conv1(1->5, 5x5): 130, Conv2(5->5, 5x5): 630, FC1(80->60): 4,860, FC2(60->10): 610
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
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset : offset + numel].view_as(p))
        offset += numel


def get_flat_grads(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1) for p in model.parameters()])


def batched_cnn_local_sgd(
    x: torch.Tensor,
    packed_clients: dict,
    tau: int,
    batch_size: int,
    lr: float,
    fixed_batches: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """非パディング連結テンソルと F.conv2d(..., groups=N) による GPU 一括並列局所 SGD。"""
    N = int(packed_clients["num_clients"])
    X_cat = packed_clients["X_cat"]
    y_cat = packed_clients["y_cat"]
    offsets = packed_clients["offsets"]
    n_vec = packed_clients["n_vec"]
    weights = packed_clients["weights"]

    xi = x.detach().unsqueeze(0).expand(N, -1).clone().requires_grad_(True)

    for j in range(tau):
        with torch.no_grad():
            if fixed_batches is None:
                idx = offsets.unsqueeze(1) + (torch.rand((N, batch_size), device=x.device) * n_vec.unsqueeze(1)).long()
                xb = X_cat[idx]
                yb = y_cat[idx]
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
        h2 = F.relu(F.max_pool2d(F.conv2d(h1, w2, b2, groups=N), 2))

        h2_flat = h2.view(bs, N, 80).permute(1, 0, 2)
        w3 = xi[:, 760:5560].reshape(N, 60, 80)
        b3 = xi[:, 5560:5620].reshape(N, 1, 60)
        h3 = F.relu(torch.bmm(h2_flat, w3.transpose(1, 2)) + b3)

        w4 = xi[:, 5620:6220].reshape(N, 10, 60)
        b4 = xi[:, 6220:6230].reshape(N, 1, 10)
        logits = torch.bmm(h3, w4.transpose(1, 2)) + b4

        loss_sum = F.cross_entropy(logits.reshape(N * bs, 10), yb.reshape(N * bs), reduction="sum") / float(bs)
        (grad,) = torch.autograd.grad(loss_sum, xi)

        with torch.no_grad():
            xi = (xi - lr * grad).detach().requires_grad_(True)

    with torch.no_grad():
        return torch.sum(weights.unsqueeze(1) * xi.detach(), dim=0)


def verify_batched_cnn_equivalence(
    clients: list,
    packed_clients: dict,
    tau: int = 5,
    batch_size: int = 100,
    lr: float = 0.003,
    seed: int = 42,
    device: str = "cuda",
) -> float:
    """逐次 nn.Module ループ版と GPU バッチ並列版の最大絶対誤差を検証する。"""
    torch.manual_seed(seed)
    base_model = CompactCNN(10).to(device)
    x = get_flat_params(base_model).clone()
    N = len(clients)

    xb_steps = torch.zeros((tau, N, batch_size, 1, 28, 28), device=device, dtype=torch.float32)
    yb_steps = torch.zeros((tau, N, batch_size), device=device, dtype=torch.long)

    v_seq = torch.zeros_like(x)
    local_model = CompactCNN(10).to(device)
    for i, c in enumerate(clients):
        set_flat_params(local_model, x)
        n_i = c["n_samples"]
        for j in range(tau):
            idx = torch.randint(0, n_i, (batch_size,), device=device)
            xb = c["X_train"][idx]
            yb = c["y_train"][idx]
            xb_steps[j, i] = xb
            yb_steps[j, i] = yb
            local_model.zero_grad()
            loss = F.cross_entropy(local_model(xb), yb)
            loss.backward()
            with torch.no_grad():
                for p in local_model.parameters():
                    p.sub_(lr * p.grad)
        with torch.no_grad():
            v_seq.add_(c["weight"] * get_flat_params(local_model))

    v_batch = batched_cnn_local_sgd(
        x, packed_clients, tau=tau, batch_size=batch_size, lr=lr, fixed_batches=(xb_steps, yb_steps)
    )
    return float(torch.max(torch.abs(v_seq - v_batch)).item())
