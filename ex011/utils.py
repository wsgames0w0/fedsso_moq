import math
import os
import tempfile
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = os.path.join(tempfile.gettempdir(), "fedsso_moq_mplconfig")
    os.makedirs(_mpl_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _mpl_dir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from model import MCLR, batched_mclr_local_sgd, get_flat_params

ALGO_ORDER: List[str] = [
    "FedAvg",
    "FedSSO",
    "FedSSO-NAQ",
    "FedSSO-MoQ",
    "FedSSO-AdaNAQ",
    "FedSSO-AdaMoQ",
    "FedSSO-LBFGS",
    "FedSSO-LNAQ",
    "FedSSO-LMoQ",
    "FedSSO-MLNAQ",
    "FedSSO-MLMoQ",
]

DEFAULT_SEEDS: List[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

DEFAULT_COLOR_MAP: Dict[str, str] = {
    "FedAvg": "#7f7f7f",
    "FedSSO": "#000000",
    "FedSSO-NAQ": "#1f77b4",
    "FedSSO-MoQ": "#d62728",
    "FedSSO-AdaNAQ": "#17becf",
    "FedSSO-AdaMoQ": "#ff7f0e",
    "FedSSO-LBFGS": "#8c564b",
    "FedSSO-LNAQ": "#9467bd",
    "FedSSO-LMoQ": "#2ca02c",
    "FedSSO-MLNAQ": "#bcbd22",
    "FedSSO-MLMoQ": "#e377c2",
}


def setup_japanese_matplotlib() -> None:
    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_partition_mnist(
    data_dir: str = "../datasets",
    num_clients: int = 20,
    beta: float = 0.1,
    iid: bool = False,
    train_ratio: float = 0.75,
    seed: int = 42,
    device: str = "cuda",
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MNIST を指定 Non-IID 度 beta または IID で N=20 分割し非パディング GPU テンソルを返す。"""
    set_seed(seed)
    train_ds = torchvision.datasets.MNIST(root=data_dir, train=True, download=False)
    test_ds = torchvision.datasets.MNIST(root=data_dir, train=False, download=False)
    X_all = torch.cat([train_ds.data.float(), test_ds.data.float()], dim=0)
    y_all = torch.cat([train_ds.targets, test_ds.targets], dim=0)
    X_all = (((X_all / 255.0) - 0.1307) / 0.3081).view(-1, 784)

    rng = np.random.default_rng(seed)
    y_np = y_all.numpy()
    if iid:
        all_idx = np.arange(len(X_all))
        rng.shuffle(all_idx)
        client_indices = [s.tolist() for s in np.array_split(all_idx, num_clients)]
    else:
        min_size = 0
        client_indices = [[] for _ in range(num_clients)]
        while min_size < 100:
            client_indices = [[] for _ in range(num_clients)]
            for c in range(10):
                idx_c = np.where(y_np == c)[0]
                rng.shuffle(idx_c)
                proportions = rng.dirichlet(np.repeat(beta, num_clients))
                proportions = np.clip(proportions, 0.004, None)
                proportions = proportions / proportions.sum()
                splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
                for i, split_idx in enumerate(np.split(idx_c, splits)):
                    client_indices[i].extend(split_idx.tolist())
            min_size = min(len(idx) for idx in client_indices)

    clients_data: List[Dict[str, Any]] = []
    X_train_list, y_train_list, X_test_list, y_test_list = [], [], [], []
    for i in range(num_clients):
        idx = np.array(client_indices[i])
        rng.shuffle(idx)
        n_train = int(len(idx) * train_ratio)
        tr_idx, te_idx = idx[:n_train], idx[n_train:]
        x_tr = X_all[tr_idx].to(device)
        y_tr = y_all[tr_idx].to(device)
        x_te = X_all[te_idx].to(device)
        y_te = y_all[te_idx].to(device)
        clients_data.append({"X_train": x_tr, "y_train": y_tr, "X_test": x_te, "y_test": y_te, "n_samples": len(tr_idx)})
        X_train_list.append(x_tr)
        y_train_list.append(y_tr)
        X_test_list.append(x_te)
        y_test_list.append(y_te)

    total_train = sum(c["n_samples"] for c in clients_data)
    weights_list = [c["n_samples"] / total_train for c in clients_data]
    for c, w_i in zip(clients_data, weights_list):
        c["weight"] = w_i

    offsets_list = []
    cur_off = 0
    for c in clients_data:
        offsets_list.append(cur_off)
        cur_off += c["n_samples"]

    X_cat = torch.cat(X_train_list, dim=0)
    y_cat = torch.cat(y_train_list, dim=0)
    packed_clients = {
        "num_clients": num_clients,
        "input_dim": 784,
        "num_classes": 10,
        "X_cat": X_cat,
        "y_cat": y_cat,
        "offsets": torch.tensor(offsets_list, device=device, dtype=torch.long),
        "n_vec": torch.tensor([c["n_samples"] for c in clients_data], device=device, dtype=torch.long),
        "weights": torch.tensor(weights_list, device=device, dtype=torch.float32),
    }
    return clients_data, packed_clients, X_cat, y_cat, torch.cat(X_test_list, dim=0), torch.cat(y_test_list, dim=0)


@torch.no_grad()
def evaluate_mclr(
    x_vec: torch.Tensor,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    l2_reg: float = 1e-4,
) -> Tuple[float, float]:
    if torch.isnan(x_vec).any() or torch.isinf(x_vec).any():
        return float("nan"), 0.098
    w = x_vec[:7840].view(10, 784)
    b = x_vec[7840:]
    ce_loss = float(nn.functional.cross_entropy(X_train @ w.t() + b, y_train).item())
    reg_loss = 0.5 * l2_reg * float(torch.dot(x_vec, x_vec).item())
    preds = (X_test @ w.t() + b).argmax(dim=1)
    test_acc = float((preds == y_test).float().mean().item())
    return ce_loss + reg_loss, test_acc


def get_initial_params(seed: int = 42, device: str = "cuda") -> torch.Tensor:
    set_seed(seed)
    return get_flat_params(MCLR(784, 10).to(device)).clone()


def adaptive_momentum_step_scalar(theta: float, gamma: float = 1e-5) -> Tuple[float, float]:
    t2 = theta * theta
    b = t2 - gamma
    theta_next = 0.5 * (-b + math.sqrt(b * b + 4.0 * t2))
    mu_k = (theta * (1.0 - theta)) / (t2 + theta_next)
    return mu_k, theta_next


def fedsso_curvature_clip(
    s: torch.Tensor, y: torch.Tensor, lam: float = 1e-4, Lam: float = 9999.0, enforce_pd: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Optional[float], bool]:
    y2 = float(torch.dot(y, y).item())
    s2 = float(torch.dot(s, s).item())
    if y2 <= 1e-16 or s2 <= 1e-16:
        return s, y, None, False
    ys = float(torch.dot(y, s).item())
    if not enforce_pd:
        return (s, y, ys, False) if abs(ys) > 1e-12 else (s, y, None, False)
    if ys > 0.0 and (lam < (y2 / ys) < Lam):
        return s, y, ys, False
    cur = (2.0 / (lam + Lam)) * y2
    return s * (cur / (math.sqrt(y2) * math.sqrt(s2))), y, cur, True


def _two_loop_direction(grad_base: torch.Tensor, p_hist: Deque[torch.Tensor], q_hist: Deque[torch.Tensor], pq_hist: Deque[float]) -> torch.Tensor:
    g = -grad_base.clone()
    n_hist = len(p_hist)
    if n_hist == 0:
        return g
    alphas = [0.0] * n_hist
    for i in range(n_hist - 1, -1, -1):
        phi_i = float(torch.dot(p_hist[i], g).item()) / pq_hist[i]
        alphas[i] = phi_i
        g.add_(q_hist[i], alpha=-phi_i)
    qq_last = float(torch.dot(q_hist[-1], q_hist[-1]).item())
    if qq_last > 1e-30:
        g.mul_(pq_hist[-1] / qq_last)
    for i in range(n_hist):
        tau_val = float(torch.dot(q_hist[i], g).item()) / pq_hist[i]
        g.add_(p_hist[i], alpha=(alphas[i] - tau_val))
    return g


def _memoryless_direction(grad_base: torch.Tensor, s_prev: torch.Tensor, y_prev: torch.Tensor, cur_prev: float, yy_prev: float) -> torch.Tensor:
    if cur_prev <= 1e-14 or yy_prev <= 1e-30:
        return -grad_base.clone()
    beta_k = float(torch.dot(s_prev, grad_base).item()) / cur_prev
    psi_k = float(torch.dot(y_prev, grad_base).item()) / cur_prev
    omega_k = yy_prev / cur_prev
    tau_k = cur_prev / yy_prev
    return -tau_k * (grad_base - (beta_k * y_prev + psi_k * s_prev)) - (1.0 + tau_k * omega_k) * beta_k * s_prev


@torch.no_grad()
def run_fedsso_moq_mnist(
    algo: str,
    packed_clients: Dict[str, torch.Tensor],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    rounds: int = 200,
    tau: int = 5,
    batch_size: int = 100,
    lr: float = 0.001,
    eta: float = 1.0,
    mu: float = 0.3,
    max_mu: float = 0.4,
    gamma: float = 1e-5,
    m_size: int = 10,
    lam: float = 1e-4,
    Lam: float = 9999.0,
    reset_period: int = 200,
    max_step_norm: float = 20.0,
    l2_reg: float = 1e-4,
    seed: int = 42,
    eval_rounds: Tuple[int, ...] = (20, 50, 100, 200),
) -> Dict[str, Any]:
    device = str(X_train.device)
    x = get_initial_params(seed, device)
    set_seed(seed + 100)
    d = x.numel()
    eval_set = set(eval_rounds)
    acc_at: Dict[int, float] = {}
    loss_at: Dict[int, float] = {}

    if algo == "FedAvg":
        for k in range(1, rounds + 1):
            x = batched_mclr_local_sgd(x, packed_clients, tau=tau, batch_size=batch_size, lr=lr, l2_reg=l2_reg)
            if k in eval_set:
                loss_k, acc_k = evaluate_mclr(x, X_train, y_train, X_test, y_test, l2_reg)
                loss_at[k] = loss_k
                acc_at[k] = acc_k
        return {"acc_at": acc_at, "loss_at": loss_at, "total_comm": rounds, "clip_count": 0, "reset_count": 0}

    is_full = algo in ("FedSSO", "FedSSO-NAQ", "FedSSO-MoQ", "FedSSO-AdaNAQ", "FedSSO-AdaMoQ")
    is_limited = algo in ("FedSSO-LBFGS", "FedSSO-LNAQ", "FedSSO-LMoQ")
    is_ml = algo in ("FedSSO-MLNAQ", "FedSSO-MLMoQ")
    is_naq = algo in ("FedSSO-NAQ", "FedSSO-AdaNAQ", "FedSSO-LNAQ", "FedSSO-MLNAQ")
    is_ada = algo in ("FedSSO-AdaNAQ", "FedSSO-AdaMoQ", "FedSSO-LNAQ", "FedSSO-LMoQ", "FedSSO-MLNAQ", "FedSSO-MLMoQ")
    has_momentum = algo not in ("FedSSO", "FedSSO-LBFGS")

    H = torch.eye(d, device=x.device, dtype=x.dtype) if is_full else None
    p_hist: Deque[torch.Tensor] = deque(maxlen=m_size)
    q_hist: Deque[torch.Tensor] = deque(maxlen=m_size)
    pq_hist: Deque[float] = deque(maxlen=m_size)
    ml_prev: Optional[Tuple[torch.Tensor, torch.Tensor, float, float]] = None

    vel = torch.zeros_like(x)
    theta = 1.0
    g_prev: Optional[torch.Tensor] = None
    g_base_prev: Optional[torch.Tensor] = None
    s_prev: Optional[torch.Tensor] = None
    cum_comm, clip_count, reset_count = 0, 0, 0

    def _reset_hessian() -> None:
        nonlocal ml_prev, reset_count
        if H is not None:
            H.zero_()
            H.diagonal().fill_(1.0)
        p_hist.clear()
        q_hist.clear()
        pq_hist.clear()
        ml_prev = None
        reset_count += 1

    for k in range(1, rounds + 1):
        v_agg = batched_mclr_local_sgd(x, packed_clients, tau=tau, batch_size=batch_size, lr=lr, l2_reg=l2_reg)
        cum_comm += 1
        g_k = (x - v_agg) / (lr * tau)

        if k % reset_period == 0:
            _reset_hessian()
        elif k > 1 and g_base_prev is not None and s_prev is not None:
            y_hat = g_k - g_base_prev
            s_c, y_c, cur, did_clip = fedsso_curvature_clip(s_prev, y_hat, lam=lam, Lam=Lam, enforce_pd=True)
            if did_clip:
                clip_count += 1
            if cur is not None and abs(cur) > 1e-14:
                if is_full and H is not None:
                    rho = 1.0 / cur
                    Hv = H @ y_c
                    c = rho * (1.0 + rho * float(torch.dot(y_c, Hv).item()))
                    U = torch.stack([c * s_c - rho * Hv, -rho * s_c], dim=1)
                    V = torch.stack([s_c, Hv], dim=0)
                    H.addmm_(U, V)
                elif is_limited:
                    p_hist.append(s_c.clone())
                    q_hist.append(y_c.clone())
                    pq_hist.append(cur)
                elif is_ml:
                    ml_prev = (s_c.clone(), y_c.clone(), cur, float(torch.dot(y_c, y_c).item()))

        if not has_momentum:
            mu_k = 0.0
        elif is_ada:
            mu_k, theta = adaptive_momentum_step_scalar(theta, gamma=gamma)
            mu_k = min(mu_k, max_mu)
        else:
            mu_k = mu

        if (not has_momentum) or k == 1 or mu_k == 0.0:
            g_base = g_k
        elif is_naq:
            x_look = x + mu_k * vel
            v_look = batched_mclr_local_sgd(x_look, packed_clients, tau=tau, batch_size=batch_size, lr=lr, l2_reg=l2_reg)
            cum_comm += 1
            g_base = (x_look - v_look) / (lr * tau)
        else:
            g_base = (1.0 + mu_k) * g_k - mu_k * g_prev

        if is_full and H is not None:
            d_k = -(H @ g_base)
        elif is_limited:
            d_k = _two_loop_direction(g_base, p_hist, q_hist, pq_hist)
        else:
            d_k = -g_base.clone() if ml_prev is None else _memoryless_direction(g_base, *ml_prev)

        slope = float(torch.dot(g_base, d_k).item())
        d_norm = float(torch.norm(d_k).item())
        if math.isnan(slope) or math.isnan(d_norm) or slope >= 0.0 or d_norm > max_step_norm:
            _reset_hessian()
            d_k = -g_base.clone()
            gn = float(torch.norm(d_k).item())
            if math.isfinite(gn) and gn > max_step_norm:
                d_k.mul_(max_step_norm / gn)

        vel = mu_k * vel + eta * d_k
        x = x + vel
        g_prev = g_k.clone()
        g_base_prev = g_base.clone()
        s_prev = (eta * d_k).clone()

        if k in eval_set:
            loss_k, acc_k = evaluate_mclr(x, X_train, y_train, X_test, y_test, l2_reg)
            loss_at[k] = loss_k
            acc_at[k] = acc_k

    if H is not None:
        del H
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"acc_at": acc_at, "loss_at": loss_at, "total_comm": cum_comm, "clip_count": clip_count, "reset_count": reset_count}


def run_11a_local_steps_study(
    data_dir: str = "../datasets",
    seeds: List[int] = DEFAULT_SEEDS,
    tau_values: Tuple[int, ...] = (1, 10, 20, 30, 40, 50),
    rounds: int = 200,
    lr: float = 0.001,
    eta: float = 1.0,
    mu: float = 0.3,
    max_mu: float = 0.4,
    m_size: int = 10,
    per_seed_csv_path: Optional[str] = "../datasets/ex011_local_steps_per_seed.csv",
    summary_csv_path: Optional[str] = "../datasets/ex011_local_steps_summary.csv",
    device: str = "cuda",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """実験 11A: 局所更新ステップ数 tau in {1, 10, 20, 30, 40, 50} の影響を全11手法 × 10シードで検証。"""
    per_seed_rows: List[Dict[str, Any]] = []
    pbar = tqdm(total=len(seeds) * len(tau_values) * len(ALGO_ORDER), desc="Ex011A Local Steps τ")

    for s in seeds:
        _, packed, X_tr, y_tr, X_te, y_te = load_and_partition_mnist(
            data_dir=data_dir, num_clients=20, beta=0.1, iid=False, seed=s, device=device
        )
        for tau in tau_values:
            for algo in ALGO_ORDER:
                res = run_fedsso_moq_mnist(
                    algo=algo, packed_clients=packed, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
                    rounds=rounds, tau=tau, batch_size=100, lr=lr, eta=eta, mu=mu, max_mu=max_mu,
                    m_size=m_size, seed=s, eval_rounds=(20, 50, 100, rounds),
                )
                per_seed_rows.append({
                    "Algorithm": algo,
                    "tau": tau,
                    "Seed": s,
                    "Acc_20r": res["acc_at"][20],
                    "Acc_50r": res["acc_at"][50],
                    "Acc_100r": res["acc_at"][100],
                    "Acc_200r": res["acc_at"][rounds],
                    "Loss_200r": res["loss_at"][rounds],
                    "Clip_Count": res["clip_count"],
                    "Reset_Count": res["reset_count"],
                    "Total_Comm": res["total_comm"],
                })
                pbar.update(1)
    pbar.close()

    per_seed_df = pd.DataFrame(per_seed_rows)
    summary_df = (
        per_seed_df.groupby(["Algorithm", "tau"], as_index=False)
        .agg(
            Seeds=("Seed", "count"),
            Acc20_Ave=("Acc_20r", "mean"),
            Acc50_Ave=("Acc_50r", "mean"),
            Acc_Ave=("Acc_200r", "mean"),
            Acc_Best=("Acc_200r", "max"),
            Acc_Worst=("Acc_200r", "min"),
            Acc_Std=("Acc_200r", "std"),
            Loss_Ave=("Loss_200r", "mean"),
            Clip_Ave=("Clip_Count", "mean"),
        )
    )
    summary_df["Algorithm"] = pd.Categorical(summary_df["Algorithm"], categories=ALGO_ORDER, ordered=True)
    summary_df = summary_df.sort_values(["Algorithm", "tau"]).reset_index(drop=True)
    summary_df["Algorithm"] = summary_df["Algorithm"].astype(str)

    if per_seed_csv_path:
        per_seed_df.to_csv(per_seed_csv_path, index=False)
    if summary_csv_path:
        summary_df.to_csv(summary_csv_path, index=False)
    return summary_df, per_seed_df


def run_11b_non_iid_study(
    data_dir: str = "../datasets",
    seeds: List[int] = DEFAULT_SEEDS,
    dist_configs: Tuple[Tuple[str, float, bool], ...] = (
        ("強Non-IID (β=0.05)", 0.05, False),
        ("中Non-IID (β=0.2)",  0.20, False),
        ("弱Non-IID (β=1.0)",  1.00, False),
        ("IID (均一分布)",      0.00, True),
    ),
    rounds: int = 200,
    lr: float = 0.001,
    eta: float = 1.0,
    mu: float = 0.3,
    max_mu: float = 0.4,
    m_size: int = 10,
    per_seed_csv_path: Optional[str] = "../datasets/ex011_non_iid_per_seed.csv",
    summary_csv_path: Optional[str] = "../datasets/ex011_non_iid_summary.csv",
    device: str = "cuda",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """実験 11B: Non-IID 度 beta in {0.05, 0.2, 1.0, IID} に対する耐性を全11手法 × 10シードで検証。"""
    per_seed_rows: List[Dict[str, Any]] = []
    pbar = tqdm(total=len(seeds) * len(dist_configs) * len(ALGO_ORDER), desc="Ex011B Non-IID β")

    for label, beta, is_iid in dist_configs:
        for s in seeds:
            _, packed, X_tr, y_tr, X_te, y_te = load_and_partition_mnist(
                data_dir=data_dir, num_clients=20, beta=beta if not is_iid else 0.1, iid=is_iid, seed=s, device=device
            )
            for algo in ALGO_ORDER:
                res = run_fedsso_moq_mnist(
                    algo=algo, packed_clients=packed, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
                    rounds=rounds, tau=5, batch_size=100, lr=lr, eta=eta, mu=mu, max_mu=max_mu,
                    m_size=m_size, seed=s, eval_rounds=(20, 50, 100, rounds),
                )
                per_seed_rows.append({
                    "Distribution": label,
                    "beta": beta if not is_iid else -1.0,
                    "Algorithm": algo,
                    "Seed": s,
                    "Acc_20r": res["acc_at"][20],
                    "Acc_50r": res["acc_at"][50],
                    "Acc_100r": res["acc_at"][100],
                    "Acc_200r": res["acc_at"][rounds],
                    "Loss_200r": res["loss_at"][rounds],
                    "Clip_Count": res["clip_count"],
                    "Reset_Count": res["reset_count"],
                })
                pbar.update(1)
    pbar.close()

    per_seed_df = pd.DataFrame(per_seed_rows)
    dist_order = [d[0] for d in dist_configs]
    summary_df = (
        per_seed_df.groupby(["Distribution", "Algorithm"], as_index=False)
        .agg(
            Seeds=("Seed", "count"),
            Acc20_Ave=("Acc_20r", "mean"),
            Acc100_Ave=("Acc_100r", "mean"),
            Acc_Ave=("Acc_200r", "mean"),
            Acc_Best=("Acc_200r", "max"),
            Acc_Worst=("Acc_200r", "min"),
            Acc_Std=("Acc_200r", "std"),
            Loss_Ave=("Loss_200r", "mean"),
            Clip_Ave=("Clip_Count", "mean"),
        )
    )
    summary_df["Distribution"] = pd.Categorical(summary_df["Distribution"], categories=dist_order, ordered=True)
    summary_df["Algorithm"] = pd.Categorical(summary_df["Algorithm"], categories=ALGO_ORDER, ordered=True)
    summary_df = summary_df.sort_values(["Distribution", "Algorithm"]).reset_index(drop=True)
    summary_df["Distribution"] = summary_df["Distribution"].astype(str)
    summary_df["Algorithm"] = summary_df["Algorithm"].astype(str)

    if per_seed_csv_path:
        per_seed_df.to_csv(per_seed_csv_path, index=False)
    if summary_csv_path:
        summary_df.to_csv(summary_csv_path, index=False)
    return summary_df, per_seed_df


def _resolve_mean_only_flag(flag: Union[bool, str], selected_seeds: Optional[List[int]], param_name: str) -> bool:
    if isinstance(flag, str):
        if flag == "auto":
            return selected_seeds is None
        raise ValueError(f"{param_name} は True, False, または 'auto' を指定してください: {flag!r}")
    if isinstance(flag, bool):
        return flag
    raise ValueError(f"{param_name} は True, False, または 'auto' を指定してください: {flag!r}")


def _format_seed_desc(seeds_present: List[int], selected_seeds: Optional[List[int]], use_plot_mean_only: bool) -> str:
    if len(seeds_present) == 1:
        return f"Seed={seeds_present[0]}"
    if use_plot_mean_only:
        return f"全{len(seeds_present)}シード平均のみ" if selected_seeds is None else f"{len(seeds_present)}シード平均のみ (Seeds={seeds_present})"
    return f"Seeds={seeds_present}"


def plot_filtered_11a_tau(
    selected_algos: Optional[List[str]] = None,
    selected_seeds: Optional[List[int]] = None,
    per_seed_csv_path: str = "../datasets/ex011_local_steps_per_seed.csv",
    show_mean: bool = True,
    plot_mean_only: Union[bool, str] = "auto",
    table_mean_only: Union[bool, str] = "auto",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    setup_japanese_matplotlib()
    use_plot_mean_only = _resolve_mean_only_flag(plot_mean_only, selected_seeds, "plot_mean_only")
    use_table_mean_only = _resolve_mean_only_flag(table_mean_only, selected_seeds, "table_mean_only")

    df = pd.read_csv(per_seed_csv_path)
    if selected_algos is not None:
        df = df[df["Algorithm"].isin(selected_algos)]
    if selected_seeds is not None:
        df = df[df["Seed"].isin([int(s) for s in selected_seeds])]
    if df.empty:
        raise ValueError("実験 11A: 一致するデータがありません。")

    algos_present = (
        [a for a in selected_algos if a in set(df["Algorithm"])]
        if selected_algos is not None
        else [a for a in ALGO_ORDER if a in set(df["Algorithm"])]
    )
    seeds_present = sorted(int(s) for s in df["Seed"].unique())
    taus = sorted(int(t) for t in df["tau"].unique())
    multi_seed = len(seeds_present) > 1
    draw_mean = multi_seed and (show_mean or use_plot_mean_only)
    draw_individuals = (not multi_seed) or (not use_plot_mean_only)
    seed_linestyles = ["-", "--", "-.", ":"]

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.5))
    for algo in algos_present:
        col = DEFAULT_COLOR_MAP.get(algo, "#333333")
        sub_a = df[df["Algorithm"] == algo].sort_values(["Seed", "tau"])
        if draw_individuals:
            for idx_s, s in enumerate(seeds_present):
                sub_s = sub_a[sub_a["Seed"] == s]
                if sub_s.empty:
                    continue
                ls = seed_linestyles[idx_s % len(seed_linestyles)] if (multi_seed and len(seeds_present) <= 4) else "-"
                lw = 1.0 if draw_mean else 1.9
                alpha = 0.45 if draw_mean else 0.90
                lbl = f"{algo} (seed={s})" if (not multi_seed or len(seeds_present) <= 2 or not draw_mean) else None
                axes[0].plot(sub_s["tau"], sub_s["Acc_20r"], color=col, linestyle=ls, linewidth=lw, alpha=alpha)
                axes[1].plot(sub_s["tau"], sub_s["Acc_200r"], color=col, linestyle=ls, linewidth=lw, alpha=alpha, label=lbl)
        if draw_mean:
            m_df = sub_a.groupby("tau", as_index=False)[["Acc_20r", "Acc_200r"]].mean().sort_values("tau")
            axes[0].plot(m_df["tau"], m_df["Acc_20r"], color=col, linestyle="-", linewidth=2.2, marker="o", label=f"{algo} (Ave)")
            axes[1].plot(m_df["tau"], m_df["Acc_200r"], color=col, linestyle="-", linewidth=2.2, marker="o", label=f"{algo} (Ave, {len(seeds_present)}s)")

    seed_desc = _format_seed_desc(seeds_present, selected_seeds, use_plot_mean_only)
    axes[0].set_title(f"(a) 20ラウンド時点精度 vs 局所ステップ数 τ ({seed_desc})", fontsize=11.5)
    axes[1].set_title(f"(b) 200ラウンド最終精度 vs 局所ステップ数 τ ({seed_desc})", fontsize=11.5)
    for ax in axes:
        ax.set_xticks(taus)
        ax.set_xlabel("局所更新ステップ数 τ", fontsize=10.5)
        ax.set_ylabel("テスト精度 (Test Accuracy)", fontsize=10.5)
        ax.grid(True, linestyle="--", alpha=0.4)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(loc="lower right", fontsize=8.2, ncol=1 if len(labels) <= 8 else 2, framealpha=0.92)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    df_out = df.copy()
    df_out["Algorithm"] = pd.Categorical(df_out["Algorithm"], categories=algos_present, ordered=True)
    if use_table_mean_only:
        summary = (
            df_out.groupby(["Algorithm", "tau"], as_index=False, observed=True)
            .agg(Seeds=("Seed", "count"), Acc20_Ave=("Acc_20r", "mean"), Acc_Ave=("Acc_200r", "mean"), Loss_Ave=("Loss_200r", "mean"))
            .sort_values(["Algorithm", "tau"])
            .reset_index(drop=True)
        )
        summary["Algorithm"] = summary["Algorithm"].astype(str)
        return summary
    out_df = df_out[["Algorithm", "tau", "Seed", "Acc_20r", "Acc_200r", "Loss_200r"]].sort_values(["Algorithm", "tau", "Seed"]).reset_index(drop=True)
    out_df["Algorithm"] = out_df["Algorithm"].astype(str)
    return out_df


def plot_filtered_11b_non_iid(
    selected_algos: Optional[List[str]] = None,
    selected_seeds: Optional[List[int]] = None,
    per_seed_csv_path: str = "../datasets/ex011_non_iid_per_seed.csv",
    show_mean: bool = True,
    plot_mean_only: Union[bool, str] = "auto",
    table_mean_only: Union[bool, str] = "auto",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    setup_japanese_matplotlib()
    use_plot_mean_only = _resolve_mean_only_flag(plot_mean_only, selected_seeds, "plot_mean_only")
    use_table_mean_only = _resolve_mean_only_flag(table_mean_only, selected_seeds, "table_mean_only")

    df = pd.read_csv(per_seed_csv_path)
    if selected_algos is not None:
        df = df[df["Algorithm"].isin(selected_algos)]
    if selected_seeds is not None:
        df = df[df["Seed"].isin([int(s) for s in selected_seeds])]
    if df.empty:
        raise ValueError("実験 11B: 一致するデータがありません。")

    algos_present = (
        [a for a in selected_algos if a in set(df["Algorithm"])]
        if selected_algos is not None
        else [a for a in ALGO_ORDER if a in set(df["Algorithm"])]
    )
    dist_order = ["強Non-IID (β=0.05)", "中Non-IID (β=0.2)", "弱Non-IID (β=1.0)", "IID (均一分布)"]
    dists_present = [d for d in dist_order if d in set(df["Distribution"])]
    dist_x = {d: idx for idx, d in enumerate(dists_present)}
    df = df.copy()
    df["dist_idx"] = df["Distribution"].map(dist_x)

    seeds_present = sorted(int(s) for s in df["Seed"].unique())
    multi_seed = len(seeds_present) > 1
    draw_mean = multi_seed and (show_mean or use_plot_mean_only)
    draw_individuals = (not multi_seed) or (not use_plot_mean_only)
    seed_linestyles = ["-", "--", "-.", ":"]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6))
    for algo in algos_present:
        col = DEFAULT_COLOR_MAP.get(algo, "#333333")
        sub_a = df[df["Algorithm"] == algo].sort_values(["Seed", "dist_idx"])
        if draw_individuals:
            for idx_s, s in enumerate(seeds_present):
                sub_s = sub_a[sub_a["Seed"] == s]
                if sub_s.empty:
                    continue
                ls = seed_linestyles[idx_s % len(seed_linestyles)] if (multi_seed and len(seeds_present) <= 4) else "-"
                lw = 1.0 if draw_mean else 1.9
                alpha = 0.45 if draw_mean else 0.90
                lbl = f"{algo} (seed={s})" if (not multi_seed or len(seeds_present) <= 2 or not draw_mean) else None
                axes[0].plot(sub_s["dist_idx"], sub_s["Acc_20r"], color=col, linestyle=ls, linewidth=lw, alpha=alpha)
                axes[1].plot(sub_s["dist_idx"], sub_s["Acc_200r"], color=col, linestyle=ls, linewidth=lw, alpha=alpha, label=lbl)
        if draw_mean:
            m_df = sub_a.groupby("dist_idx", as_index=False)[["Acc_20r", "Acc_200r"]].mean().sort_values("dist_idx")
            axes[0].plot(m_df["dist_idx"], m_df["Acc_20r"], color=col, linestyle="-", linewidth=2.2, marker="o", label=f"{algo} (Ave)")
            axes[1].plot(m_df["dist_idx"], m_df["Acc_200r"], color=col, linestyle="-", linewidth=2.2, marker="o", label=f"{algo} (Ave, {len(seeds_present)}s)")

    seed_desc = _format_seed_desc(seeds_present, selected_seeds, use_plot_mean_only)
    axes[0].set_title(f"(a) Non-IID度別 20ラウンドテスト精度 ({seed_desc})", fontsize=11.5)
    axes[1].set_title(f"(b) Non-IID度別 200ラウンド最終テスト精度 ({seed_desc})", fontsize=11.5)
    for ax in axes:
        ax.set_xticks(range(len(dists_present)))
        ax.set_xticklabels(dists_present, rotation=12)
        ax.set_ylabel("テスト精度 (Test Accuracy)", fontsize=10.5)
        ax.grid(True, linestyle="--", alpha=0.4)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(loc="lower right", fontsize=8.2, ncol=1 if len(labels) <= 8 else 2, framealpha=0.92)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    df_out = df.copy()
    df_out["Distribution"] = pd.Categorical(df_out["Distribution"], categories=dists_present, ordered=True)
    df_out["Algorithm"] = pd.Categorical(df_out["Algorithm"], categories=algos_present, ordered=True)
    if use_table_mean_only:
        summary = (
            df_out.groupby(["Distribution", "Algorithm"], as_index=False, observed=True)
            .agg(Seeds=("Seed", "count"), Acc20_Ave=("Acc_20r", "mean"), Acc_Ave=("Acc_200r", "mean"), Loss_Ave=("Loss_200r", "mean"))
            .sort_values(["Distribution", "Algorithm"])
            .reset_index(drop=True)
        )
        summary["Distribution"] = summary["Distribution"].astype(str)
        summary["Algorithm"] = summary["Algorithm"].astype(str)
        return summary
    out_df = df_out[["Distribution", "Algorithm", "Seed", "Acc_20r", "Acc_200r", "Loss_200r"]].sort_values(["Distribution", "Algorithm", "Seed"]).reset_index(drop=True)
    out_df["Distribution"] = out_df["Distribution"].astype(str)
    out_df["Algorithm"] = out_df["Algorithm"].astype(str)
    return out_df


def plot_11a_local_steps(summary_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """実験 11A の全11手法 × τ 平均結果を 2 パネルで描画する。"""
    setup_japanese_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4))
    taus = sorted(int(t) for t in summary_df["tau"].unique())
    for algo in ALGO_ORDER:
        sub = summary_df[summary_df["Algorithm"] == algo].sort_values("tau")
        if sub.empty:
            continue
        col = DEFAULT_COLOR_MAP.get(algo, "#333333")
        lw = 2.3 if algo in ("FedSSO", "FedSSO-NAQ", "FedSSO-MoQ") else 1.5
        ls = "--" if algo in ("FedAvg", "FedSSO-LNAQ", "FedSSO-LMoQ") else "-"
        axes[0].plot(sub["tau"], sub["Acc20_Ave"], color=col, linestyle=ls, linewidth=lw, marker="o", markersize=4.5, label=algo)
        axes[1].plot(sub["tau"], sub["Acc_Ave"], color=col, linestyle=ls, linewidth=lw, marker="s", markersize=4.5, label=algo)
    axes[0].set_title("(a) 20ラウンド時点精度 vs 局所ステップ数 τ (10シード平均)", fontsize=11.5)
    axes[1].set_title("(b) 200ラウンド最終精度 vs 局所ステップ数 τ (10シード平均)", fontsize=11.5)
    for ax in axes:
        ax.set_xticks(taus)
        ax.set_xlabel("局所更新ステップ数 τ")
        ax.set_ylabel("テスト精度 (Test Accuracy)")
        ax.grid(True, linestyle="--", alpha=0.4)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=6, fontsize=8.8, framealpha=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_11b_non_iid(summary_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """実験 11B の全11手法 × Non-IID度 β 平均結果を描画する。"""
    setup_japanese_matplotlib()
    dist_order = ["強Non-IID (β=0.05)", "中Non-IID (β=0.2)", "弱Non-IID (β=1.0)", "IID (均一分布)"]
    dists = [d for d in dist_order if d in set(summary_df["Distribution"])]
    d_idx = {d: i for i, d in enumerate(dists)}
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4))
    for algo in ALGO_ORDER:
        sub = summary_df[summary_df["Algorithm"] == algo].copy()
        if sub.empty:
            continue
        sub["d_idx"] = sub["Distribution"].map(d_idx)
        sub = sub.sort_values("d_idx")
        col = DEFAULT_COLOR_MAP.get(algo, "#333333")
        lw = 2.3 if algo in ("FedSSO", "FedSSO-NAQ", "FedSSO-MoQ") else 1.5
        ls = "--" if algo in ("FedAvg", "FedSSO-LNAQ", "FedSSO-LMoQ") else "-"
        axes[0].plot(sub["d_idx"], sub["Acc20_Ave"], color=col, linestyle=ls, linewidth=lw, marker="o", markersize=4.5, label=algo)
        axes[1].plot(sub["d_idx"], sub["Acc_Ave"], color=col, linestyle=ls, linewidth=lw, marker="s", markersize=4.5, label=algo)
    axes[0].set_title("(a) 20ラウンドテスト精度 vs Non-IID度 (10シード平均)", fontsize=11.5)
    axes[1].set_title("(b) 200ラウンド最終テスト精度 vs Non-IID度 (10シード平均)", fontsize=11.5)
    for ax in axes:
        ax.set_xticks(range(len(dists)))
        ax.set_xticklabels(dists, rotation=12)
        ax.set_ylabel("テスト精度 (Test Accuracy)")
        ax.grid(True, linestyle="--", alpha=0.4)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=6, fontsize=8.8, framealpha=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

