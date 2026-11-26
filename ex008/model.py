from typing import Optional, Tuple
import torch
import torch.nn as nn


class BinaryLR(nn.Module):
    """L2正則化付き二値ロジスティック回帰 (LIBSVM ijcnn1: input_dim=22, num_classes=2, d=46)。"""

    def __init__(self, input_dim: int = 22, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def get_flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def compute_lr_grad(
    x_vec: torch.Tensor,
    xb: torch.Tensor,
    yb: torch.Tensor,
    input_dim: int = 22,
    num_classes: int = 2,
    l2_reg: float = 1e-4,
) -> torch.Tensor:
    w_end = num_classes * input_dim
    w = x_vec[:w_end].view(num_classes, input_dim)
    b = x_vec[w_end:]
    bs = xb.size(0)
    logits = xb @ w.t() + b
    probs = torch.softmax(logits, dim=1)
    probs[torch.arange(bs, device=xb.device), yb] -= 1.0
    probs /= float(bs)
    grad_w = probs.t() @ xb + l2_reg * w
    grad_b = probs.sum(dim=0) + l2_reg * b
    return torch.cat([grad_w.reshape(-1), grad_b])


def batched_lr_local_sgd(
    x: torch.Tensor,
    packed_clients: dict,
    tau: int,
    batch_size: int,
    lr: float,
    l2_reg: float = 1e-4,
    fixed_batches: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    N = int(packed_clients["num_clients"])
    in_dim = int(packed_clients["input_dim"])
    n_cls = int(packed_clients["num_classes"])
    X_cat = packed_clients["X_cat"]
    y_cat = packed_clients["y_cat"]
    offsets = packed_clients["offsets"]
    n_vec = packed_clients["n_vec"]
    weights = packed_clients["weights"]

    w_end = n_cls * in_dim
    xi = x.unsqueeze(0).expand(N, -1).clone()

    for j in range(tau):
        if fixed_batches is None:
            idx = offsets.unsqueeze(1) + (torch.rand((N, batch_size), device=x.device) * n_vec.unsqueeze(1)).long()
            xb = X_cat[idx]
            yb = y_cat[idx]
        else:
            xb = fixed_batches[0][j]
            yb = fixed_batches[1][j]

        bs_cur = xb.size(1)
        w = xi[:, :w_end].view(N, n_cls, in_dim)
        b = xi[:, w_end:].view(N, 1, n_cls)
        logits = torch.bmm(xb, w.transpose(1, 2)) + b
        probs = torch.softmax(logits, dim=2)
        probs.scatter_add_(2, yb.unsqueeze(-1), torch.full((N, bs_cur, 1), -1.0, device=x.device, dtype=probs.dtype))
        probs.div_(float(bs_cur))

        grad_w = torch.bmm(probs.transpose(1, 2), xb) + l2_reg * w
        grad_b = probs.sum(dim=1) + l2_reg * b.squeeze(1)
        grad = torch.cat([grad_w.reshape(N, w_end), grad_b], dim=1)
        xi.sub_(lr * grad)

    return torch.sum(weights.unsqueeze(1) * xi, dim=0)


@torch.no_grad()
def verify_batched_lr_equivalence(
    clients: list,
    packed_clients: dict,
    tau: int = 5,
    batch_size: int = 100,
    lr: float = 0.001,
    l2_reg: float = 1e-4,
    seed: int = 42,
    device: str = "cuda",
) -> float:
    torch.manual_seed(seed)
    in_dim = int(packed_clients["input_dim"])
    n_cls = int(packed_clients["num_classes"])
    model = BinaryLR(in_dim, n_cls).to(device)
    x = get_flat_params(model).clone()
    N = len(clients)

    xb_steps = torch.zeros((tau, N, batch_size, in_dim), device=device, dtype=torch.float32)
    yb_steps = torch.zeros((tau, N, batch_size), device=device, dtype=torch.long)

    v_seq = torch.zeros_like(x)
    for i, c in enumerate(clients):
        xi = x.clone()
        n_i = c["n_samples"]
        for j in range(tau):
            idx = torch.randint(0, n_i, (batch_size,), device=device)
            xb = c["X_train"][idx]
            yb = c["y_train"][idx]
            xb_steps[j, i] = xb
            yb_steps[j, i] = yb
            grad = compute_lr_grad(xi, xb, yb, in_dim, n_cls, l2_reg)
            xi.sub_(lr * grad)
        v_seq.add_(c["weight"] * xi)

    v_batch = batched_lr_local_sgd(
        x, packed_clients, tau=tau, batch_size=batch_size, lr=lr, l2_reg=l2_reg,
        fixed_batches=(xb_steps, yb_steps),
    )
    return float(torch.max(torch.abs(v_seq - v_batch)).item())
