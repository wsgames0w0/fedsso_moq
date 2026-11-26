import glob
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
from tqdm import tqdm

from model import BinaryLR, batched_lr_local_sgd, get_flat_params

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


def load_and_partition_ijcnn1(
    data_dir: str = "../datasets",
    num_clients: int = 3,
    beta: float = 0.1,
    train_ratio: float = 0.75,
    seed: int = 42,
    device: str = "cuda",
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    LIBSVM ijcnn1 (22特徴量, 二値分類) を読み込み、論文 Table 3 / Figure 7 に準拠して
    N=3 クライアントへ Dirichlet (beta=0.1) Non-IID 分割を行う。
    """
    set_seed(seed)
    cache = torch.load(os.path.join(data_dir, "LIBSVM", "ijcnn1.pt"), weights_only=False)
    X_all = cache["X"].float()
    y_all = cache["y"].long()

    mean_v = X_all.mean(dim=0, keepdim=True)
    std_v = X_all.std(dim=0, keepdim=True).clamp(min=1e-6)
    X_all = (X_all - mean_v) / std_v

    rng = np.random.default_rng(seed)
    y_arr = y_all.numpy()
    min_size = 0
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    while min_size < 1000:
        client_indices = [[] for _ in range(num_clients)]
        for c in (0, 1):
            idx_c = np.where(y_arr == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(np.repeat(beta, num_clients))
            proportions = np.clip(proportions, 0.02, None)
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
        "input_dim": 22,
        "num_classes": 2,
        "X_cat": X_cat,
        "y_cat": y_cat,
        "offsets": torch.tensor(offsets_list, device=device, dtype=torch.long),
        "n_vec": torch.tensor([c["n_samples"] for c in clients_data], device=device, dtype=torch.long),
        "weights": torch.tensor(weights_list, device=device, dtype=torch.float32),
    }
    return clients_data, packed_clients, X_cat, y_cat, torch.cat(X_test_list, dim=0), torch.cat(y_test_list, dim=0)


@torch.no_grad()
def evaluate_lr(
    x_vec: torch.Tensor,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    l2_reg: float = 1e-4,
) -> Tuple[float, float]:
    if torch.isnan(x_vec).any() or torch.isinf(x_vec).any():
        return float("nan"), 0.5
    w = x_vec[:44].view(2, 22)
    b = x_vec[44:]
    ce = float(nn.functional.cross_entropy(X_train @ w.t() + b, y_train).item())
    reg = 0.5 * l2_reg * float(torch.dot(x_vec, x_vec).item())
    preds = (X_test @ w.t() + b).argmax(dim=1)
    acc = float((preds == y_test).float().mean().item())
    return ce + reg, acc


def get_initial_params(seed: int = 42, device: str = "cuda") -> torch.Tensor:
    set_seed(seed)
    return get_flat_params(BinaryLR(22, 2).to(device)).clone()


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


def estimate_server_memory_mb(algo: str, d: int, m_size: int = 10) -> float:
    if algo == "FedAvg":
        floats = d
    elif algo == "FedSSO":
        floats = d * d + 4 * d
    elif algo in ("FedSSO-NAQ", "FedSSO-AdaNAQ"):
        floats = d * d + 6 * d
    elif algo in ("FedSSO-MoQ", "FedSSO-AdaMoQ"):
        floats = d * d + 7 * d
    elif algo == "FedSSO-LBFGS":
        floats = (2 * m_size + 4) * d
    elif algo == "FedSSO-LNAQ":
        floats = (2 * m_size + 6) * d
    elif algo == "FedSSO-LMoQ":
        floats = (2 * m_size + 7) * d
    elif algo == "FedSSO-MLNAQ":
        floats = 8 * d
    elif algo == "FedSSO-MLMoQ":
        floats = 9 * d
    else:
        floats = d
    return floats * 4.0 / (1024.0 ** 2)


@torch.no_grad()
def run_fedsso_moq_lr(
    algo: str,
    packed_clients: Dict[str, torch.Tensor],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    rounds: int = 50,
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
    enforce_pd: bool = True,
    max_step_norm: float = 20.0,
    l2_reg: float = 1e-4,
    seed: int = 42,
    measure_server_time: bool = True,
) -> Dict[str, Any]:
    device = str(X_train.device)
    x = get_initial_params(seed, device)
    set_seed(seed + 100)
    d = x.numel()

    if algo == "FedAvg":
        losses, accs, comm_rounds = [], [], []
        server_time_ms = 0.0
        t0 = time.perf_counter()
        for k in range(1, rounds + 1):
            v = batched_lr_local_sgd(x, packed_clients, tau=tau, batch_size=batch_size, lr=lr, l2_reg=l2_reg)
            if measure_server_time and X_train.device.type == "cuda":
                torch.cuda.synchronize()
            ts = time.perf_counter()
            x = v
            if measure_server_time and X_train.device.type == "cuda":
                torch.cuda.synchronize()
            server_time_ms += (time.perf_counter() - ts) * 1000.0
            loss, acc = evaluate_lr(x, X_train, y_train, X_test, y_test, l2_reg)
            losses.append(loss)
            accs.append(acc)
            comm_rounds.append(k)
        return {
            "train_loss": losses,
            "test_acc": accs,
            "comm_rounds": comm_rounds,
            "clip_count": 0,
            "reset_count": 0,
            "avg_server_time_ms": server_time_ms / rounds,
            "server_mem_mb": estimate_server_memory_mb("FedAvg", d),
            "total_time_s": time.perf_counter() - t0,
        }

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

    losses, accs, comm_rounds = [], [], []
    cum_comm, clip_count, reset_count = 0, 0, 0
    server_time_ms = 0.0
    t_total_start = time.perf_counter()

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
        v_agg = batched_lr_local_sgd(x, packed_clients, tau=tau, batch_size=batch_size, lr=lr, l2_reg=l2_reg)
        cum_comm += 1

        if measure_server_time and X_train.device.type == "cuda":
            torch.cuda.synchronize()
        ts = time.perf_counter()

        g_k = (x - v_agg) / (lr * tau)
        if k % reset_period == 0:
            _reset_hessian()
        elif k > 1 and g_base_prev is not None and s_prev is not None:
            y_hat = g_k - g_base_prev
            s_c, y_c, cur, did_clip = fedsso_curvature_clip(s_prev, y_hat, lam=lam, Lam=Lam, enforce_pd=enforce_pd)
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

        if measure_server_time and X_train.device.type == "cuda":
            torch.cuda.synchronize()
        server_time_ms += (time.perf_counter() - ts) * 1000.0

        if (not has_momentum) or k == 1 or mu_k == 0.0:
            g_base = g_k
        elif is_naq:
            x_look = x + mu_k * vel
            v_look = batched_lr_local_sgd(x_look, packed_clients, tau=tau, batch_size=batch_size, lr=lr, l2_reg=l2_reg)
            cum_comm += 1
            if measure_server_time and X_train.device.type == "cuda":
                torch.cuda.synchronize()
            ts2 = time.perf_counter()
            g_base = (x_look - v_look) / (lr * tau)
            if measure_server_time and X_train.device.type == "cuda":
                torch.cuda.synchronize()
            server_time_ms += (time.perf_counter() - ts2) * 1000.0
        else:
            if measure_server_time and X_train.device.type == "cuda":
                torch.cuda.synchronize()
            ts2 = time.perf_counter()
            g_base = (1.0 + mu_k) * g_k - mu_k * g_prev
            if measure_server_time and X_train.device.type == "cuda":
                torch.cuda.synchronize()
            server_time_ms += (time.perf_counter() - ts2) * 1000.0

        if measure_server_time and X_train.device.type == "cuda":
            torch.cuda.synchronize()
        ts3 = time.perf_counter()

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

        if measure_server_time and X_train.device.type == "cuda":
            torch.cuda.synchronize()
        server_time_ms += (time.perf_counter() - ts3) * 1000.0

        loss, acc = evaluate_lr(x, X_train, y_train, X_test, y_test, l2_reg)
        losses.append(loss)
        accs.append(acc)
        comm_rounds.append(cum_comm)

    if H is not None:
        del H
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "train_loss": losses,
        "test_acc": accs,
        "comm_rounds": comm_rounds,
        "clip_count": clip_count,
        "reset_count": reset_count,
        "avg_server_time_ms": server_time_ms / rounds,
        "server_mem_mb": estimate_server_memory_mb(algo, d, m_size=m_size),
        "total_time_s": time.perf_counter() - t_total_start,
    }


def first_round_reaching_target(values: List[float], target: float, comm_rounds: Optional[List[int]] = None) -> Tuple[Optional[int], Optional[int]]:
    for idx, val in enumerate(values, start=1):
        if val >= target:
            cr = comm_rounds[idx - 1] if comm_rounds is not None else idx
            return idx, cr
    return None, None


def run_multiseed_ijcnn1_lr(
    algo_configs: List[Dict[str, Any]],
    packed_clients: Dict[str, torch.Tensor],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    setting_tag: str = "same",
    seeds: List[int] = DEFAULT_SEEDS,
    rounds: int = 50,
    tau: int = 5,
    batch_size: int = 100,
    l2_reg: float = 1e-4,
    targets: Tuple[float, ...] = (0.85, 0.90, 0.91, 0.92),
    param_dim: int = 46,
    save_log_dir: Optional[str] = None,
    history_csv_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    bytes_per_comm = param_dim * 8
    if save_log_dir:
        os.makedirs(save_log_dir, exist_ok=True)
    if history_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(history_csv_path)), exist_ok=True)

    per_seed_rows, summary_rows, history_frames = [], [], []
    curves_dict: Dict[str, Dict[str, Any]] = {}
    pbar = tqdm(total=len(algo_configs) * len(seeds), desc=f"ijcnn1 LR ({setting_tag})")

    for cfg in algo_configs:
        name = cfg["name"]
        algo = cfg["algo"]
        lr = cfg.get("lr", 0.001)
        eta = cfg.get("eta", 1.0)
        mu = cfg.get("mu", 0.3)
        max_mu = cfg.get("max_mu", 0.4)
        m_size = cfg.get("m_size", 10)
        lam = cfg.get("lam", 1e-4)
        Lam = cfg.get("Lam", 9999.0)
        reset_period = cfg.get("reset_period", 200)
        enforce_pd = cfg.get("enforce_pd", True)

        seed_losses, seed_accs, seed_comms = [], [], []
        final_losses, final_accs, max_accs = [], [], []
        clips_list, resets_list, srv_ms_list, total_s_list = [], [], [], []
        target_iters: Dict[float, List[int]] = {t: [] for t in targets}
        target_comms: Dict[float, List[int]] = {t: [] for t in targets}
        mem_mb = estimate_server_memory_mb(algo, param_dim, m_size=m_size)

        for s in seeds:
            res = run_fedsso_moq_lr(
                algo=algo, packed_clients=packed_clients,
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                rounds=rounds, tau=tau, batch_size=batch_size, lr=lr, eta=eta,
                mu=mu, max_mu=max_mu, m_size=m_size, lam=lam, Lam=Lam,
                reset_period=reset_period, enforce_pd=enforce_pd, l2_reg=l2_reg, seed=s,
            )
            seed_losses.append(res["train_loss"])
            seed_accs.append(res["test_acc"])
            seed_comms.append(res["comm_rounds"])

            f_loss = res["train_loss"][-1]
            f_acc = res["test_acc"][-1]
            m_acc = float(np.nanmax(res["test_acc"]))
            final_losses.append(f_loss)
            final_accs.append(f_acc)
            max_accs.append(m_acc)
            clips_list.append(res["clip_count"])
            resets_list.append(res["reset_count"])
            srv_ms_list.append(res["avg_server_time_ms"])
            total_s_list.append(res["total_time_s"])

            row_seed = {
                "Setting": setting_tag,
                "Algorithm": name,
                "Seed": s,
                "Final_Loss": f_loss,
                "Final_Acc": f_acc,
                "Max_Acc": m_acc,
                "Clip_Count": res["clip_count"],
                "Reset_Count": res["reset_count"],
                "Server_Time_ms": res["avg_server_time_ms"],
                "Server_Mem_MB": mem_mb,
                "Total_Comm_Rounds": res["comm_rounds"][-1],
                "Total_Time_s": res["total_time_s"],
            }
            for t in targets:
                it, cr = first_round_reaching_target(res["test_acc"], t, res["comm_rounds"])
                row_seed[f"Iter@{t:.2f}"] = it if it is not None else np.nan
                row_seed[f"Comm@{t:.2f}"] = cr if cr is not None else np.nan
                if it is not None:
                    target_iters[t].append(it)
                    target_comms[t].append(cr)
            per_seed_rows.append(row_seed)

            hist_df = pd.DataFrame({
                "Setting": setting_tag,
                "Algorithm": name,
                "Seed": s,
                "Round": np.arange(1, len(res["train_loss"]) + 1, dtype=int),
                "Comm_Round": res["comm_rounds"],
                "Train_Loss": res["train_loss"],
                "Test_Acc": res["test_acc"],
            })
            if save_log_dir:
                safe_name = name.replace("/", "_").replace(" ", "_")
                hist_df.to_csv(os.path.join(save_log_dir, f"{setting_tag}_{safe_name}_seed{s}.csv"), index=False)
            if history_csv_path:
                history_frames.append(hist_df)
            pbar.update(1)

        loss_arr = np.array(seed_losses, dtype=float)
        acc_arr = np.array(seed_accs, dtype=float)
        comm_arr = np.mean(np.array(seed_comms, dtype=float), axis=0)
        curves_dict[name] = {
            "rounds": np.arange(1, rounds + 1),
            "comm_rounds": comm_arr,
            "loss_mean": np.nanmean(loss_arr, axis=0),
            "loss_std": np.nanstd(loss_arr, axis=0),
            "acc_mean": np.nanmean(acc_arr, axis=0),
            "acc_std": np.nanstd(acc_arr, axis=0),
            "acc_min": np.nanmin(acc_arr, axis=0),
            "acc_max": np.nanmax(acc_arr, axis=0),
        }

        main_target = targets[-2] if len(targets) >= 2 else targets[-1]
        mean_comm_main = float(np.mean(target_comms[main_target])) if target_comms[main_target] else float(comm_arr[-1])
        row_sum = {
            "Setting": setting_tag,
            "Algorithm": name,
            "Comm_Per_Iter": round(float(comm_arr[-1]) / rounds, 3),
            "Loss_Ave": float(np.nanmean(final_losses)),
            "Acc_Ave": float(np.mean(final_accs)),
            "Acc_Med": float(np.median(final_accs)),
            "Acc_Best": float(np.max(final_accs)),
            "Acc_Worst": float(np.min(final_accs)),
            "Acc_Std": float(np.std(final_accs)),
            "MaxAcc_Ave": float(np.mean(max_accs)),
            "Clip_Ave": float(np.mean(clips_list)),
            "Reset_Ave": float(np.mean(resets_list)),
            "Server_Time_ms": float(np.mean(srv_ms_list)),
            "Server_Mem_MB": mem_mb,
            f"Bytes@{main_target:.2f}": int(round(mean_comm_main * bytes_per_comm)),
        }
        for t in targets:
            row_sum[f"Reach@{t:.2f}"] = f"{len(target_iters[t])}/{len(seeds)}"
            row_sum[f"Iter@{t:.2f}"] = round(float(np.mean(target_iters[t])), 1) if target_iters[t] else None
            row_sum[f"Comm@{t:.2f}"] = round(float(np.mean(target_comms[t])), 1) if target_comms[t] else None
        summary_rows.append(row_sum)

    if history_csv_path and history_frames:
        pd.concat(history_frames, ignore_index=True).to_csv(history_csv_path, index=False)
    pbar.close()
    return pd.DataFrame(summary_rows), pd.DataFrame(per_seed_rows), curves_dict


def plot_ijcnn1_curves(curves_same: Dict[str, Dict[str, Any]], curves_opt: Dict[str, Dict[str, Any]], save_path: Optional[str] = None) -> None:
    setup_japanese_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0))
    styles = [
        ("FedAvg",        "#7f7f7f", "--", "o"),
        ("FedSSO",        "#000000", "-",  "s"),
        ("FedSSO-NAQ",    "#1f77b4", "-",  "^"),
        ("FedSSO-MoQ",    "#d62728", "-",  "v"),
        ("FedSSO-AdaNAQ", "#17becf", "-.", "D"),
        ("FedSSO-AdaMoQ", "#ff7f0e", "-.", "p"),
        ("FedSSO-LBFGS",  "#8c564b", ":",  "X"),
        ("FedSSO-LNAQ",   "#9467bd", "--", "<"),
        ("FedSSO-LMoQ",   "#2ca02c", "--", ">"),
        ("FedSSO-MLNAQ",  "#bcbd22", ":",  "*"),
        ("FedSSO-MLMoQ",  "#e377c2", ":",  "h"),
    ]
    for row_idx, (cdict, title_tag) in enumerate([(curves_same, "同一設定 (α=0.001)"), (curves_opt, "最適設定")]):
        for name, col, ls, mk in styles:
            if name not in cdict:
                continue
            c = cdict[name]
            r = c["rounds"]
            axes[row_idx, 0].plot(r, c["loss_mean"], color=col, linestyle=ls, marker=mk, markevery=5, markersize=4.5, linewidth=1.7, label=name)
            axes[row_idx, 1].plot(r, c["acc_mean"], color=col, linestyle=ls, marker=mk, markevery=5, markersize=4.5, linewidth=1.7, label=name)
            axes[row_idx, 1].fill_between(r, c["acc_mean"] - c["acc_std"], c["acc_mean"] + c["acc_std"], color=col, alpha=0.07)

        axes[row_idx, 0].set_title(f"LIBSVM ijcnn1 訓練損失 ({title_tag}, 10シード平均)", fontsize=11)
        axes[row_idx, 0].set_xlabel("グローバル反復数 (k)")
        axes[row_idx, 0].set_ylabel("訓練損失 (Train Loss)")
        axes[row_idx, 0].grid(True, linestyle="--", alpha=0.4)

        axes[row_idx, 1].set_title(f"LIBSVM ijcnn1 テスト精度 ({title_tag}, 10シード平均)", fontsize=11)
        axes[row_idx, 1].set_xlabel("グローバル反復数 (k)")
        axes[row_idx, 1].set_ylabel("テスト精度 (Test Accuracy)")
        axes[row_idx, 1].grid(True, linestyle="--", alpha=0.4)

    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=6, fontsize=9, framealpha=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_ijcnn1_boxplot(per_seed_opt_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    setup_japanese_matplotlib()
    order = [a for a in ALGO_ORDER if a in set(per_seed_opt_df["Algorithm"])]
    data = [per_seed_opt_df[per_seed_opt_df["Algorithm"] == a]["Final_Acc"].values for a in order]
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showmeans=True)
    for patch, a in zip(bp["boxes"], order):
        patch.set_facecolor(DEFAULT_COLOR_MAP.get(a, "#cccccc"))
        patch.set_alpha(0.55)
    ax.set_title("LIBSVM ijcnn1 (最適設定) 最終テスト精度のシード間分布 (10シード, 50ラウンド)", fontsize=12)
    ax.set_ylabel("最終テスト精度 (Test Accuracy)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def _resolve_mean_only_flag(flag: Union[bool, str], selected_seeds: Optional[List[int]], param_name: str) -> bool:
    if isinstance(flag, str):
        if flag == "auto":
            return selected_seeds is None
        raise ValueError(f"{param_name} は True, False, または 'auto' を指定してください: {flag!r}")
    if isinstance(flag, bool):
        return flag
    raise ValueError(f"{param_name} は True, False, または 'auto' を指定してください: {flag!r}")


def load_history_csv(
    history_csv_path: str = "../datasets/ex008_ijcnn1_lr_opt_history.csv",
    log_dir: str = "../datasets/ex008_ijcnn1_lr_logs",
    setting_tag: str = "opt",
) -> pd.DataFrame:
    if history_csv_path and os.path.isfile(history_csv_path):
        df = pd.read_csv(history_csv_path)
        return df[df["Setting"] == setting_tag] if "Setting" in df.columns else df
    if log_dir and os.path.isdir(log_dir):
        files = sorted(glob.glob(os.path.join(log_dir, f"{setting_tag}_*.csv")))
        if files:
            return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raise FileNotFoundError(f"履歴CSVが見つかりません: {history_csv_path} / {log_dir}")


def plot_filtered_results(
    selected_algos: Optional[List[str]] = None,
    selected_seeds: Optional[List[int]] = None,
    setting_tag: str = "opt",
    history_csv_path: str = "../datasets/ex008_ijcnn1_lr_opt_history.csv",
    log_dir: str = "../datasets/ex008_ijcnn1_lr_logs",
    x_axis: str = "round",
    show_mean: bool = True,
    plot_mean_only: Union[bool, str] = "auto",
    table_mean_only: Union[bool, str] = "auto",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    setup_japanese_matplotlib()
    use_plot_mean_only = _resolve_mean_only_flag(plot_mean_only, selected_seeds, "plot_mean_only")
    use_table_mean_only = _resolve_mean_only_flag(table_mean_only, selected_seeds, "table_mean_only")
    df = load_history_csv(history_csv_path=history_csv_path, log_dir=log_dir, setting_tag=setting_tag)

    if selected_algos is not None:
        df = df[df["Algorithm"].isin(selected_algos)]
    if selected_seeds is not None:
        df = df[df["Seed"].isin([int(s) for s in selected_seeds])]
    if df.empty:
        raise ValueError("指定されたアルゴリズム・シード条件に一致するデータがありません。")

    algos_present = (
        [a for a in selected_algos if a in set(df["Algorithm"])]
        if selected_algos is not None
        else [a for a in ALGO_ORDER if a in set(df["Algorithm"])] + [a for a in sorted(df["Algorithm"].unique()) if a not in ALGO_ORDER]
    )
    seeds_present = sorted(int(s) for s in df["Seed"].unique())
    x_col = "Comm_Round" if x_axis == "comm" else "Round"
    x_label = "累積通信ラウンド数" if x_axis == "comm" else "グローバル反復数 (k)"

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    multi_seed = len(seeds_present) > 1
    draw_mean = multi_seed and (show_mean or use_plot_mean_only)
    draw_individuals = (not multi_seed) or (not use_plot_mean_only)
    seed_linestyles = ["-", "--", "-.", ":"]

    for algo in algos_present:
        col = DEFAULT_COLOR_MAP.get(algo, "#333333")
        sub_a = df[df["Algorithm"] == algo].sort_values(["Seed", "Round"])
        if draw_individuals:
            for idx_s, s in enumerate(seeds_present):
                sub_s = sub_a[sub_a["Seed"] == s]
                if sub_s.empty:
                    continue
                ls = seed_linestyles[idx_s % len(seed_linestyles)] if (multi_seed and len(seeds_present) <= 4) else "-"
                lw = 1.1 if draw_mean else 2.0
                alpha = 0.55 if draw_mean else 0.90
                lbl = f"{algo} (seed={s})" if (not multi_seed or len(seeds_present) <= 3 or not draw_mean) else None
                axes[0].plot(sub_s[x_col], sub_s["Train_Loss"], color=col, linestyle=ls, linewidth=lw, alpha=alpha)
                axes[1].plot(sub_s[x_col], sub_s["Test_Acc"], color=col, linestyle=ls, linewidth=lw, alpha=alpha, label=lbl)
        if draw_mean:
            mean_df = sub_a.groupby("Round", as_index=False)[["Comm_Round", "Train_Loss", "Test_Acc"]].mean()
            axes[0].plot(mean_df[x_col], mean_df["Train_Loss"], color=col, linestyle="-", linewidth=2.3, alpha=0.95)
            axes[1].plot(mean_df[x_col], mean_df["Test_Acc"], color=col, linestyle="-", linewidth=2.3, alpha=0.95, label=f"{algo} (Ave, {len(seeds_present)} seeds)")

    if len(seeds_present) == 1:
        seed_desc = f"Seed={seeds_present[0]}"
    elif use_plot_mean_only:
        seed_desc = f"全{len(seeds_present)}シード平均のみ" if selected_seeds is None else f"{len(seeds_present)}シード平均のみ (Seeds={seeds_present})"
    else:
        seed_desc = f"Seeds={seeds_present}"

    axes[0].set_title(f"(a) LIBSVM ijcnn1 ({setting_tag}) 訓練損失 ({seed_desc})", fontsize=11.5)
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("訓練損失 (Train Loss)")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].set_title(f"(b) LIBSVM ijcnn1 ({setting_tag}) テスト精度 ({seed_desc})", fontsize=11.5)
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("テスト精度 (Test Accuracy)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(loc="lower right", fontsize=8.5, ncol=1 if len(labels) <= 8 else 2, framealpha=0.92)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    max_round = df.groupby(["Algorithm", "Seed"])["Round"].transform("max")
    final_df = df[df["Round"] == max_round][["Algorithm", "Seed", "Round", "Comm_Round", "Train_Loss", "Test_Acc"]].copy()
    final_df["Algorithm"] = pd.Categorical(final_df["Algorithm"], categories=algos_present, ordered=True)
    if use_table_mean_only:
        mean_tbl = (
            final_df.groupby("Algorithm", as_index=False, observed=True)
            .agg(
                Seeds=("Seed", "count"),
                Round=("Round", "max"),
                Comm_Round=("Comm_Round", "mean"),
                Train_Loss_Ave=("Train_Loss", "mean"),
                Test_Acc_Ave=("Test_Acc", "mean"),
            )
            .sort_values("Algorithm")
            .reset_index(drop=True)
        )
        mean_tbl["Algorithm"] = mean_tbl["Algorithm"].astype(str)
        return mean_tbl
    final_df = final_df.sort_values(["Algorithm", "Seed"]).reset_index(drop=True)
    final_df["Algorithm"] = final_df["Algorithm"].astype(str)
    return final_df
