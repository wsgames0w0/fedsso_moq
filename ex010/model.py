from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sent140MLP(nn.Module):
    """
    論文 Figure 8(c,d,g,h) 用の Sent140 + GloVe 200d 感情分類 MLP (200 -> 64 -> 32 -> 2, d = 15,010)。
    FC1: 200*64 + 64 = 12,864
    FC2: 64*32 + 32 = 2,080
    FC3: 32*2 + 2 = 66
    合計: 15,010 パラメータ
    """

    def __init__(self, input_dim: int = 200, hidden1: int = 64, hidden2: int = 32, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


def batched_mlp_local_sgd(
    x: torch.Tensor,
    packed_clients: dict,
    tau: int,
    batch_size: int,
    lr: float,
    fixed_batches: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Sent140MLP (d=15,010) 用の GPU 一括並列局所 SGD (torch.bmm + autograd)。"""
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

        w1 = xi[:, 0:12800].reshape(N, 64, 200)
        b1 = xi[:, 12800:12864].reshape(N, 1, 64)
        h1 = F.relu(torch.bmm(xb, w1.transpose(1, 2)) + b1)

        w2 = xi[:, 12864:14912].reshape(N, 32, 64)
        b2 = xi[:, 14912:14944].reshape(N, 1, 32)
        h2 = F.relu(torch.bmm(h1, w2.transpose(1, 2)) + b2)

        w3 = xi[:, 14944:15008].reshape(N, 2, 32)
        b3 = xi[:, 15008:15010].reshape(N, 1, 2)
        logits = torch.bmm(h2, w3.transpose(1, 2)) + b3

        loss_sum = F.cross_entropy(logits.reshape(N * bs, 2), yb.reshape(N * bs), reduction="sum") / float(bs)
        (grad,) = torch.autograd.grad(loss_sum, xi)

        with torch.no_grad():
            xi = (xi - lr * grad).detach().requires_grad_(True)

    with torch.no_grad():
        return torch.sum(weights.unsqueeze(1) * xi.detach(), dim=0)


def verify_batched_mlp_equivalence(
    clients: list,
    packed_clients: dict,
    tau: int = 5,
    batch_size: int = 100,
    lr: float = 0.03,
    seed: int = 42,
    device: str = "cuda",
) -> float:
    torch.manual_seed(seed)
    base_model = Sent140MLP().to(device)
    x = get_flat_params(base_model).clone()
    N = len(clients)

    xb_steps = torch.zeros((tau, N, batch_size, 200), device=device, dtype=torch.float32)
    yb_steps = torch.zeros((tau, N, batch_size), device=device, dtype=torch.long)

    v_seq = torch.zeros_like(x)
    local_model = Sent140MLP().to(device)
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

    v_batch = batched_mlp_local_sgd(
        x, packed_clients, tau=tau, batch_size=batch_size, lr=lr, fixed_batches=(xb_steps, yb_steps)
    )
    return float(torch.max(torch.abs(v_seq - v_batch)).item())
