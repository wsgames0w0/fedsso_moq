from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    論文 Figure 4(c) / Figure 5(c) / Figure 6(b) 用の LeNet-5 モデル (28x28入力, d = 44,426)。
    Conv1(1->6, 5x5): 156
    Conv2(6->16, 5x5): 2,416
    FC1(256->120): 30,840
    FC2(120->84): 10,164
    FC3(84->10): 850
    合計: 44,426 パラメータ
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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


def batched_lenet_local_sgd(
    x: torch.Tensor,
    packed_clients: dict,
    tau: int,
    batch_size: int,
    lr: float,
    fixed_batches: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    LeNet-5 (d=44,426) の全クライアント一括並列局所 SGD (F.conv2d groups=N + torch.bmm)。
    パラメータスライス境界:
      conv1.w [0:150], conv1.b [150:156]
      conv2.w [156:2556], conv2.b [2556:2572]
      fc1.w [2572:33292], fc1.b [33292:33412]
      fc2.w [33412:43492], fc2.b [43492:43576]
      fc3.w [43576:44416], fc3.b [44416:44426]
    """
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

        w1 = xi[:, 0:150].reshape(N * 6, 1, 5, 5)
        b1 = xi[:, 150:156].reshape(N * 6)
        h1 = F.relu(F.max_pool2d(F.conv2d(x_in, w1, b1, groups=N), 2))

        w2 = xi[:, 156:2556].reshape(N * 16, 6, 5, 5)
        b2 = xi[:, 2556:2572].reshape(N * 16)
        h2 = F.relu(F.max_pool2d(F.conv2d(h1, w2, b2, groups=N), 2))  # (bs, N*16, 4, 4)

        h2_flat = h2.view(bs, N, 256).permute(1, 0, 2)  # (N, bs, 256)
        w3 = xi[:, 2572:33292].reshape(N, 120, 256)
        b3 = xi[:, 33292:33412].reshape(N, 1, 120)
        h3 = F.relu(torch.bmm(h2_flat, w3.transpose(1, 2)) + b3)

        w4 = xi[:, 33412:43492].reshape(N, 84, 120)
        b4 = xi[:, 43492:43576].reshape(N, 1, 84)
        h4 = F.relu(torch.bmm(h3, w4.transpose(1, 2)) + b4)

        w5 = xi[:, 43576:44416].reshape(N, 10, 84)
        b5 = xi[:, 44416:44426].reshape(N, 1, 10)
        logits = torch.bmm(h4, w5.transpose(1, 2)) + b5

        loss_sum = F.cross_entropy(logits.reshape(N * bs, 10), yb.reshape(N * bs), reduction="sum") / float(bs)
        (grad,) = torch.autograd.grad(loss_sum, xi)

        with torch.no_grad():
            xi = (xi - lr * grad).detach().requires_grad_(True)

    with torch.no_grad():
        return torch.sum(weights.unsqueeze(1) * xi.detach(), dim=0)


def verify_batched_lenet_equivalence(
    clients: list,
    packed_clients: dict,
    tau: int = 5,
    batch_size: int = 100,
    lr: float = 0.01,
    seed: int = 42,
    device: str = "cuda",
) -> float:
    """逐次 nn.Module ループ版と GPU バッチ並列版の最大絶対誤差を検証する。"""
    torch.manual_seed(seed)
    base_model = LeNet5(10).to(device)
    x = get_flat_params(base_model).clone()
    N = len(clients)

    xb_steps = torch.zeros((tau, N, batch_size, 1, 28, 28), device=device, dtype=torch.float32)
    yb_steps = torch.zeros((tau, N, batch_size), device=device, dtype=torch.long)

    v_seq = torch.zeros_like(x)
    local_model = LeNet5(10).to(device)
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

    v_batch = batched_lenet_local_sgd(
        x, packed_clients, tau=tau, batch_size=batch_size, lr=lr, fixed_batches=(xb_steps, yb_steps)
    )
    return float(torch.max(torch.abs(v_seq - v_batch)).item())
