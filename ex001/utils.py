import glob
import math
import os
import tempfile
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

# Matplotlib 一時キャッシュの設定（ワークスペース外の一時領域を使用）
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

from model import MCLR, get_flat_params


# 実験1で比較する全11手法の標準表示順序
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


def setup_japanese_matplotlib() -> None:
    """Matplotlib で日本語フォントとマイナス記号が正しく表示されるように設定する。"""
    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed: int = 42) -> None:
    """乱数シードを固定して実験の再現性を確保する。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_partition_mnist(
    data_dir: str = "../datasets",
    num_clients: int = 20,
    beta: float = 0.1,
    train_ratio: float = 0.75,
    seed: int = 42,
    device: str = "cuda",
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    MNISTデータセットを読み込み、PFL-Non-IIDに準拠したDirichlet分布によるNon-IID分割を行う。
    クライアント数は20、各クライアント内で75%:25%に訓練・テスト分割し、全テンソルをGPU上へ配置する。
    さらに全クライアント一括並列SGD計算用のパック済みテンソル辞書も生成して返す。
    """
    set_seed(seed)
    train_ds = torchvision.datasets.MNIST(root=data_dir, train=True, download=False)
    test_ds = torchvision.datasets.MNIST(root=data_dir, train=False, download=False)

    X_all = torch.cat([train_ds.data.float(), test_ds.data.float()], dim=0)
    y_all = torch.cat([train_ds.targets, test_ds.targets], dim=0)

    # MNIST標準化 (平均 0.1307, 標準偏差 0.3081)
    X_all = ((X_all / 255.0) - 0.1307) / 0.3081
    X_all = X_all.view(-1, 784)

    num_classes = 10
    rng = np.random.default_rng(seed)
    min_size = 0
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    while min_size < 120:
        client_indices = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx_c = np.where(y_all.numpy() == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(np.repeat(beta, num_clients))
            proportions = np.clip(proportions, 0.005, None)
            proportions = proportions / proportions.sum()
            splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            for i, split_idx in enumerate(np.split(idx_c, splits)):
                client_indices[i].extend(split_idx.tolist())
        min_size = min(len(idx) for idx in client_indices)

    clients_data: List[Dict[str, Any]] = []
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    class_dist_matrix = np.zeros((num_clients, num_classes), dtype=int)

    for i in range(num_clients):
        idx = np.array(client_indices[i])
        rng.shuffle(idx)
        n_total = len(idx)
        n_train = int(n_total * train_ratio)

        train_idx = idx[:n_train]
        test_idx = idx[n_train:]

        x_tr = X_all[train_idx].to(device)
        y_tr = y_all[train_idx].to(device)
        x_te = X_all[test_idx].to(device)
        y_te = y_all[test_idx].to(device)

        for c in range(num_classes):
            class_dist_matrix[i, c] = int((y_all[idx] == c).sum().item())

        clients_data.append({
            "X_train": x_tr,
            "y_train": y_tr,
            "X_test": x_te,
            "y_test": y_te,
            "n_samples": len(train_idx),
        })
        X_train_list.append(x_tr)
        y_train_list.append(y_tr)
        X_test_list.append(x_te)
        y_test_list.append(y_te)

    total_train = sum(c["n_samples"] for c in clients_data)
    for c in clients_data:
        c["weight"] = c["n_samples"] / total_train

    # 全クライアント並列ミニバッチ処理用のパディング済みテンソルを作成
    max_n = max(c["n_samples"] for c in clients_data)
    X_pad = torch.zeros((num_clients, max_n, 784), device=device, dtype=torch.float32)
    y_pad = torch.zeros((num_clients, max_n), device=device, dtype=torch.long)
    n_vec = torch.zeros(num_clients, device=device, dtype=torch.long)
    weights = torch.zeros(num_clients, device=device, dtype=torch.float32)

    for i, c in enumerate(clients_data):
        n_i = c["n_samples"]
        X_pad[i, :n_i] = c["X_train"]
        y_pad[i, :n_i] = c["y_train"]
        n_vec[i] = n_i
        weights[i] = c["weight"]

    packed_clients = {
        "X_pad": X_pad,
        "y_pad": y_pad,
        "n_vec": n_vec,
        "weights": weights,
        "num_clients": num_clients,
    }

    return (
        clients_data,
        packed_clients,
        torch.cat(X_train_list, dim=0),
        torch.cat(y_train_list, dim=0),
        torch.cat(X_test_list, dim=0),
        torch.cat(y_test_list, dim=0),
        class_dist_matrix,
    )


def compute_mclr_grad(
    x_vec: torch.Tensor,
    xb: torch.Tensor,
    yb: torch.Tensor,
    l2_reg: float = 1e-4,
) -> torch.Tensor:
    """逐次検証用の単一クライアント MCLR ミニバッチ勾配計算。"""
    w = x_vec[:7840].view(10, 784)
    b = x_vec[7840:]
    bs = xb.size(0)
    logits = xb @ w.t() + b
    probs = torch.softmax(logits, dim=1)
    probs[torch.arange(bs, device=xb.device), yb] -= 1.0
    probs /= bs
    grad_w = probs.t() @ xb + l2_reg * w
    grad_b = probs.sum(dim=0) + l2_reg * b
    return torch.cat([grad_w.reshape(-1), grad_b])


def batched_mclr_local_sgd(
    x: torch.Tensor,
    packed_clients: Dict[str, torch.Tensor],
    tau: int,
    batch_size: int,
    lr: float,
    l2_reg: float = 1e-4,
    fixed_batches: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    全クライアント (N=20) の tau ステップ局所SGDを GPU 上で torch.bmm により一括並列計算し、
    サーバ集約モデル v_k = sum_i p_i * x_{k, tau}^{(i)} を返す。
    fixed_batches が渡された場合は (tau, N, bs, 784), (tau, N, bs) をそのまま使用する（等価性検証用）。
    """
    N = int(packed_clients["num_clients"])
    X_pad = packed_clients["X_pad"]
    y_pad = packed_clients["y_pad"]
    n_vec = packed_clients["n_vec"]
    weights = packed_clients["weights"]

    xi = x.unsqueeze(0).expand(N, -1).clone()  # (N, 7850)
    client_ids = torch.arange(N, device=x.device).unsqueeze(1)

    for j in range(tau):
        if fixed_batches is None:
            idx = (torch.rand((N, batch_size), device=x.device) * n_vec.unsqueeze(1)).long()
            xb = X_pad[client_ids, idx]  # (N, bs, 784)
            yb = y_pad[client_ids, idx]  # (N, bs)
        else:
            xb = fixed_batches[0][j]
            yb = fixed_batches[1][j]

        w = xi[:, :7840].view(N, 10, 784)
        b = xi[:, 7840:].view(N, 1, 10)
        logits = torch.bmm(xb, w.transpose(1, 2)) + b  # (N, bs, 10)
        probs = torch.softmax(logits, dim=2)
        probs.scatter_add_(2, yb.unsqueeze(-1), torch.full((N, xb.size(1), 1), -1.0, device=x.device, dtype=probs.dtype))
        probs.div_(float(xb.size(1)))

        grad_w = torch.bmm(probs.transpose(1, 2), xb) + l2_reg * w
        grad_b = probs.sum(dim=1) + l2_reg * b.squeeze(1)
        grad = torch.cat([grad_w.reshape(N, 7840), grad_b], dim=1)
        xi.sub_(lr * grad)

    return torch.sum(weights.unsqueeze(1) * xi, dim=0)


@torch.no_grad()
def verify_batched_mclr_equivalence(
    clients: List[Dict[str, Any]],
    packed_clients: Dict[str, torch.Tensor],
    tau: int = 5,
    batch_size: int = 100,
    lr: float = 0.001,
    l2_reg: float = 1e-4,
    seed: int = 42,
    device: str = "cuda",
) -> float:
    """
    同一ミニバッチ系列を用いて、逐次クライアントループ版と GPU バッチ並列版 (batched_mclr_local_sgd)
    の集約ベクトル v_k の最大絶対誤差 ||v_seq - v_batch||_inf を検証する。
    """
    set_seed(seed)
    model = MCLR(784, 10).to(device)
    x = get_flat_params(model).clone()
    N = len(clients)

    xb_steps = torch.zeros((tau, N, batch_size, 784), device=device, dtype=torch.float32)
    yb_steps = torch.zeros((tau, N, batch_size), device=device, dtype=torch.long)

    # 逐次版を実行しつつミニバッチを記録
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
            grad = compute_mclr_grad(xi, xb, yb, l2_reg)
            xi.sub_(lr * grad)
        v_seq.add_(c["weight"] * xi)

    v_batch = batched_mclr_local_sgd(
        x, packed_clients, tau=tau, batch_size=batch_size, lr=lr, l2_reg=l2_reg,
        fixed_batches=(xb_steps, yb_steps),
    )
    max_diff = float(torch.max(torch.abs(v_seq - v_batch)).item())
    return max_diff


@torch.no_grad()
def evaluate_mclr(
    x_vec: torch.Tensor,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    l2_reg: float = 1e-4,
) -> Tuple[float, float]:
    """グローバルモデルの訓練損失 (L2正則化付き交差エントロピー) とテスト精度を計算する。"""
    if torch.isnan(x_vec).any() or torch.isinf(x_vec).any():
        return float("nan"), 0.098
    w = x_vec[:7840].view(10, 784)
    b = x_vec[7840:]
    logits_tr = X_train @ w.t() + b
    ce_loss = nn.functional.cross_entropy(logits_tr, y_train).item()
    reg_loss = 0.5 * l2_reg * float(torch.dot(x_vec, x_vec).item())
    train_loss = ce_loss + reg_loss

    logits_te = X_test @ w.t() + b
    preds = logits_te.argmax(dim=1)
    test_acc = float((preds == y_test).float().mean().item())
    return train_loss, test_acc


def get_initial_params(seed: int = 42, device: str = "cuda") -> torch.Tensor:
    """初期モデルパラメータ x_1 を生成する。"""
    set_seed(seed)
    model = MCLR(784, 10).to(device)
    return get_flat_params(model).clone()


def adaptive_momentum_step_scalar(theta: float, gamma: float = 1e-5) -> Tuple[float, float]:
    """
    MoQ論文 式(64), (65) に基づき適応的慣性係数 mu_k と次ステップの theta_{k+1} を求める。
    theta_{k+1}^2 = (1 - theta_{k+1}) * theta_k^2 + gamma * theta_{k+1}
    """
    t2 = theta * theta
    b = t2 - gamma
    theta_next = 0.5 * (-b + math.sqrt(b * b + 4.0 * t2))
    mu_k = (theta * (1.0 - theta)) / (t2 + theta_next)
    return mu_k, theta_next


def fedsso_curvature_clip(
    s: torch.Tensor,
    y: torch.Tensor,
    lam: float = 1e-4,
    Lam: float = 9999.0,
    enforce_pd: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[float], bool]:
    """
    FedSSO論文 Algorithm 1 (BFGS Updating process) に準拠した正定値性保証と曲率クリッピング。
    条件: lam < ||y||^2 / cur < Lam を満たさない場合、cur = (2 / (lam + Lam)) * ||y||^2 に設定し、
    逆行列形式 (H_k / Two-Loop / MemoryLess) での正定値性と双対有界性を保つため
    s_tilde = s * (cur / (||y|| * ||s||)) へリスケーリングする。
    """
    y2 = float(torch.dot(y, y).item())
    s2 = float(torch.dot(s, s).item())
    if y2 <= 1e-16 or s2 <= 1e-16:
        return s, y, None, False

    ys = float(torch.dot(y, s).item())
    if not enforce_pd:
        if abs(ys) <= 1e-12:
            return s, y, None, False
        return s, y, ys, False

    if ys > 0.0 and (lam < (y2 / ys) < Lam):
        return s, y, ys, False

    cur = (2.0 / (lam + Lam)) * y2
    scale = cur / (math.sqrt(y2) * math.sqrt(s2))
    return s * scale, y, cur, True


def _two_loop_direction(
    grad_base: torch.Tensor,
    p_hist: Deque[torch.Tensor],
    q_hist: Deque[torch.Tensor],
    pq_hist: Deque[float],
) -> torch.Tensor:
    """
    LQN / LNAQ / LMoQ 用の Two-Loop Recursion (moq/ex002/utils.py 準拠)。
    探索方向 -H_k * grad_base を O(m * d) で算出する。
    """
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
        gamma_k = pq_hist[-1] / qq_last
        g.mul_(gamma_k)

    for i in range(n_hist):
        tau_val = float(torch.dot(q_hist[i], g).item()) / pq_hist[i]
        g.add_(p_hist[i], alpha=(alphas[i] - tau_val))
    return g


def _memoryless_direction(
    grad_base: torch.Tensor,
    s_prev: torch.Tensor,
    y_prev: torch.Tensor,
    cur_prev: float,
    yy_prev: float,
) -> torch.Tensor:
    """
    MLNAQ / MLMoQ 用の O(d) 閉形式探索方向 (moq/ex002/utils.py 準拠)。
    """
    if cur_prev <= 1e-14 or yy_prev <= 1e-30:
        return -grad_base.clone()
    beta_k = float(torch.dot(s_prev, grad_base).item()) / cur_prev
    psi_k = float(torch.dot(y_prev, grad_base).item()) / cur_prev
    omega_k = yy_prev / cur_prev
    tau_k = cur_prev / yy_prev
    return (
        -tau_k * (grad_base - (beta_k * y_prev + psi_k * s_prev))
        - (1.0 + tau_k * omega_k) * beta_k * s_prev
    )


def estimate_server_memory_mb(algo: str, d: int, m_size: int = 10) -> float:
    """サーバ側理論メモリ使用量 (MB, float32=4 Bytes 基準) を算出する。"""
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
def run_fedavg_mclr(
    packed_clients: Dict[str, torch.Tensor],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    rounds: int = 200,
    tau: int = 5,
    batch_size: int = 100,
    lr: float = 0.03,
    l2_reg: float = 1e-4,
    seed: int = 42,
    eval_final_only: bool = False,
) -> Dict[str, Any]:
    """1次最適化ベースライン FedAvg をMCLR上で実行する。"""
    device = str(X_train.device)
    x = get_initial_params(seed, device)
    set_seed(seed + 100)
    d = x.numel()

    losses, accs, comm_rounds = [], [], []
    cum_comm = 0
    server_time_ms = 0.0
    t_total_start = time.perf_counter()

    for k in range(1, rounds + 1):
        v = batched_mclr_local_sgd(x, packed_clients, tau=tau, batch_size=batch_size, lr=lr, l2_reg=l2_reg)
        cum_comm += 1
        if X_train.device.type == "cuda":
            torch.cuda.synchronize()
        ts = time.perf_counter()
        x = v
        if X_train.device.type == "cuda":
            torch.cuda.synchronize()
        server_time_ms += (time.perf_counter() - ts) * 1000.0

        if (not eval_final_only) or (k == rounds):
            loss, acc = evaluate_mclr(x, X_train, y_train, X_test, y_test, l2_reg)
        else:
            loss, acc = float("nan"), float("nan")
        losses.append(loss)
        accs.append(acc)
        comm_rounds.append(cum_comm)

    total_time_s = time.perf_counter() - t_total_start
    return {
        "train_loss": losses,
        "test_acc": accs,
        "comm_rounds": comm_rounds,
        "clip_count": 0,
        "reset_count": 0,
        "avg_server_time_ms": server_time_ms / rounds,
        "server_mem_mb": estimate_server_memory_mb("FedAvg", d),
        "total_time_s": total_time_s,
    }


@torch.no_grad()
def run_fedsso_moq_mclr(
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
    enforce_pd: bool = True,
    max_step_norm: float = 20.0,
    l2_reg: float = 1e-4,
    seed: int = 42,
    eval_final_only: bool = False,
    measure_server_time: bool = True,
) -> Dict[str, Any]:
    """
    FedSSO および FedSSO + MoQファミリー（NAQ, MoQ, AdaNAQ, AdaMoQ, LNAQ, LMoQ, MLNAQ, MLMoQ, LBFGS）
    を MCLR モデル上で統一実行する。
    """
    if algo == "FedAvg":
        return run_fedavg_mclr(
            packed_clients, X_train, y_train, X_test, y_test,
            rounds=rounds, tau=tau, batch_size=batch_size, lr=lr, l2_reg=l2_reg, seed=seed,
        )

    device = str(X_train.device)
    x = get_initial_params(seed, device)
    set_seed(seed + 100)
    d = x.numel()

    is_full = algo in ("FedSSO", "FedSSO-NAQ", "FedSSO-MoQ", "FedSSO-AdaNAQ", "FedSSO-AdaMoQ")
    is_limited = algo in ("FedSSO-LBFGS", "FedSSO-LNAQ", "FedSSO-LMoQ")
    is_ml = algo in ("FedSSO-MLNAQ", "FedSSO-MLMoQ")
    is_naq = algo in ("FedSSO-NAQ", "FedSSO-AdaNAQ", "FedSSO-LNAQ", "FedSSO-MLNAQ")
    is_ada = algo in (
        "FedSSO-AdaNAQ", "FedSSO-AdaMoQ",
        "FedSSO-LNAQ", "FedSSO-LMoQ",
        "FedSSO-MLNAQ", "FedSSO-MLMoQ",
    )
    has_momentum = algo not in ("FedSSO", "FedSSO-LBFGS")

    eye = torch.eye(d, device=x.device, dtype=x.dtype) if is_full else None
    H = eye.clone() if eye is not None else None
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
    cum_comm = 0
    clip_count = 0
    reset_count = 0
    server_time_ms = 0.0
    t_total_start = time.perf_counter()

    def _reset_hessian() -> None:
        nonlocal ml_prev, reset_count
        if H is not None and eye is not None:
            H.copy_(eye)
        p_hist.clear()
        q_hist.clear()
        pq_hist.clear()
        ml_prev = None
        reset_count += 1

    for k in range(1, rounds + 1):
        # 1. 現在点 x_k でFL通信を実行し Lighthouse 平均勾配 g_k を取得 (通信 +1)
        v_agg = batched_mclr_local_sgd(x, packed_clients, tau=tau, batch_size=batch_size, lr=lr, l2_reg=l2_reg)
        cum_comm += 1

        if measure_server_time and X_train.device.type == "cuda":
            torch.cuda.synchronize()
        ts = time.perf_counter()

        g_k = (x - v_agg) / (lr * tau)

        # 2. 曲率更新 (k > 1)
        if k % reset_period == 0:
            _reset_hessian()
        elif k > 1 and g_base_prev is not None and s_prev is not None:
            y_hat = g_k - g_base_prev
            s_c, y_c, cur, did_clip = fedsso_curvature_clip(
                s_prev, y_hat, lam=lam, Lam=Lam, enforce_pd=enforce_pd
            )
            if did_clip:
                clip_count += 1
            if cur is not None and abs(cur) > 1e-14:
                if is_full and H is not None:
                    rho = 1.0 / cur
                    Hy = H @ y_c
                    yHy = float(torch.dot(y_c, Hy).item())
                    H.add_(rho * (1.0 + rho * yHy) * torch.outer(s_c, s_c))
                    H.sub_(rho * (torch.outer(Hy, s_c) + torch.outer(s_c, Hy)))
                elif is_limited:
                    p_hist.append(s_c.clone())
                    q_hist.append(y_c.clone())
                    pq_hist.append(cur)
                elif is_ml:
                    ml_prev = (s_c.clone(), y_c.clone(), cur, float(torch.dot(y_c, y_c).item()))

        # 3. 慣性係数 mu_k の決定
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

        # 4. 先読み点 x_k + mu_k * vel における勾配 g_base の取得
        if (not has_momentum) or k == 1 or mu_k == 0.0:
            g_base = g_k
        elif is_naq:
            # NAQ系は先読み点をクライアントへ送信して真の Lighthouse 勾配を取得 (追加通信 +1)
            x_look = x + mu_k * vel
            v_look = batched_mclr_local_sgd(x_look, packed_clients, tau=tau, batch_size=batch_size, lr=lr, l2_reg=l2_reg)
            cum_comm += 1
            if measure_server_time and X_train.device.type == "cuda":
                torch.cuda.synchronize()
            ts2 = time.perf_counter()
            g_base = (x_look - v_look) / (lr * tau)
            if measure_server_time and X_train.device.type == "cuda":
                torch.cuda.synchronize()
            server_time_ms += (time.perf_counter() - ts2) * 1000.0
        else:
            # MoQ系はサーバ側のみで線形外挿 (追加通信ゼロ)
            if measure_server_time and X_train.device.type == "cuda":
                torch.cuda.synchronize()
            ts2 = time.perf_counter()
            g_base = (1.0 + mu_k) * g_k - mu_k * g_prev
            if measure_server_time and X_train.device.type == "cuda":
                torch.cuda.synchronize()
            server_time_ms += (time.perf_counter() - ts2) * 1000.0

        # 5. 探索方向 d_k の計算とモデル更新
        if measure_server_time and X_train.device.type == "cuda":
            torch.cuda.synchronize()
        ts3 = time.perf_counter()

        if is_full and H is not None:
            d_k = -(H @ g_base)
        elif is_limited:
            d_k = _two_loop_direction(g_base, p_hist, q_hist, pq_hist)
        else:
            if ml_prev is None:
                d_k = -g_base.clone()
            else:
                d_k = _memoryless_direction(g_base, *ml_prev)

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

        if (not eval_final_only) or (k == rounds):
            loss, acc = evaluate_mclr(x, X_train, y_train, X_test, y_test, l2_reg)
        else:
            loss, acc = float("nan"), float("nan")
        losses.append(loss)
        accs.append(acc)
        comm_rounds.append(cum_comm)

    total_time_s = time.perf_counter() - t_total_start
    return {
        "train_loss": losses,
        "test_acc": accs,
        "comm_rounds": comm_rounds,
        "clip_count": clip_count,
        "reset_count": reset_count,
        "avg_server_time_ms": server_time_ms / rounds,
        "server_mem_mb": estimate_server_memory_mb(algo, d, m_size=m_size),
        "total_time_s": total_time_s,
    }


def first_round_reaching_target(values: List[float], target: float, comm_rounds: Optional[List[int]] = None) -> Tuple[Optional[int], Optional[int]]:
    """目標テスト精度に初めて到達した反復番号と累積通信ラウンド数を返す。"""
    for idx, val in enumerate(values, start=1):
        if val >= target:
            cr = comm_rounds[idx - 1] if comm_rounds is not None else idx
            return idx, cr
    return None, None


def run_multiseed_mclr(
    algo_configs: List[Dict[str, Any]],
    packed_clients: Dict[str, torch.Tensor],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    seeds: List[int] = DEFAULT_SEEDS,
    rounds: int = 200,
    tau: int = 5,
    batch_size: int = 100,
    l2_reg: float = 1e-4,
    targets: Tuple[float, ...] = (0.80, 0.88, 0.90, 0.91),
    param_dim: int = 7850,
    save_log_dir: Optional[str] = None,
    history_csv_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    指定された手法設定リストに対して複数シード (10シード以上) の実験を実行し、
    手法別サマリー (Ave/Med/Best/Worst/Std)、シード別詳細、および学習曲線集計を返す。
    save_log_dir / history_csv_path を指定すると、シード別・アルゴリズム別の全ラウンド推移CSVを保存する。
    """
    bytes_per_comm = param_dim * 8  # FedSSO論文 Table 1 準拠 (8 Bytes / param)
    if save_log_dir:
        os.makedirs(save_log_dir, exist_ok=True)
    if history_csv_path:
        parent_dir = os.path.dirname(os.path.abspath(history_csv_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    per_seed_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    history_frames: List[pd.DataFrame] = []
    curves_dict: Dict[str, Dict[str, Any]] = {}

    total_jobs = len(algo_configs) * len(seeds)
    pbar = tqdm(total=total_jobs, desc="MCLR Multi-Seed Experiment")

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

        seed_losses = []
        seed_accs = []
        seed_comms = []
        final_losses, final_accs, max_accs = [], [], []
        clips_list, resets_list, srv_ms_list, total_s_list = [], [], [], []
        target_iters: Dict[float, List[int]] = {t: [] for t in targets}
        target_comms: Dict[float, List[int]] = {t: [] for t in targets}
        mem_mb = estimate_server_memory_mb(algo, param_dim, m_size=m_size)

        for s in seeds:
            res = run_fedsso_moq_mclr(
                algo=algo,
                packed_clients=packed_clients,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                rounds=rounds,
                tau=tau,
                batch_size=batch_size,
                lr=lr,
                eta=eta,
                mu=mu,
                max_mu=max_mu,
                m_size=m_size,
                lam=lam,
                Lam=Lam,
                reset_period=reset_period,
                enforce_pd=enforce_pd,
                l2_reg=l2_reg,
                seed=s,
            )
            seed_losses.append(res["train_loss"])
            seed_accs.append(res["test_acc"])
            seed_comms.append(res["comm_rounds"])

            if save_log_dir or history_csv_path:
                df_hist_seed = pd.DataFrame({
                    "Algorithm": name,
                    "Seed": int(s),
                    "Round": np.arange(1, len(res["test_acc"]) + 1, dtype=int),
                    "Comm_Round": [int(c) for c in res["comm_rounds"]],
                    "Train_Loss": [float(v) for v in res["train_loss"]],
                    "Test_Acc": [float(v) for v in res["test_acc"]],
                })
                if save_log_dir:
                    df_hist_seed.to_csv(os.path.join(save_log_dir, f"{name}_seed{s}.csv"), index=False)
                if history_csv_path:
                    history_frames.append(df_hist_seed)

            f_loss = float(res["train_loss"][-1])
            f_acc = float(res["test_acc"][-1])
            m_acc = float(np.nanmax(res["test_acc"]))
            final_losses.append(f_loss)
            final_accs.append(f_acc)
            max_accs.append(m_acc)
            clips_list.append(int(res["clip_count"]))
            resets_list.append(int(res["reset_count"]))
            srv_ms_list.append(float(res["avg_server_time_ms"]))
            total_s_list.append(float(res["total_time_s"]))

            row_seed: Dict[str, Any] = {
                "Algorithm": name,
                "Seed": s,
                "Final_Loss": f_loss,
                "Final_Acc": f_acc,
                "Max_Acc": m_acc,
                "Comm_Rounds_Total": int(res["comm_rounds"][-1]),
                "Clip_Count": int(res["clip_count"]),
                "Reset_Count": int(res["reset_count"]),
                "Avg_Server_ms": float(res["avg_server_time_ms"]),
                "Total_Time_s": float(res["total_time_s"]),
            }
            for t in targets:
                it_r, cm_r = first_round_reaching_target(res["test_acc"], t, res["comm_rounds"])
                row_seed[f"Iter@{t:.2f}"] = it_r
                row_seed[f"Comm@{t:.2f}"] = cm_r
                if it_r is not None:
                    target_iters[t].append(it_r)
                    target_comms[t].append(cm_r)
            per_seed_rows.append(row_seed)
            pbar.update(1)
            pbar.set_postfix(algo=name, seed=s, acc=f"{f_acc:.4f}")

        loss_arr = np.array(seed_losses, dtype=np.float64)
        acc_arr = np.array(seed_accs, dtype=np.float64)
        comm_arr = np.array(seed_comms, dtype=np.float64)

        curves_dict[name] = {
            "loss_mean": np.nanmean(loss_arr, axis=0),
            "loss_std": np.nanstd(loss_arr, axis=0),
            "loss_min": np.nanmin(loss_arr, axis=0),
            "loss_max": np.nanmax(loss_arr, axis=0),
            "acc_mean": np.nanmean(acc_arr, axis=0),
            "acc_std": np.nanstd(acc_arr, axis=0),
            "acc_min": np.nanmin(acc_arr, axis=0),
            "acc_max": np.nanmax(acc_arr, axis=0),
            "comm_mean": np.mean(comm_arr, axis=0),
        }

        summary_row: Dict[str, Any] = {
            "Algorithm": name,
            "Seeds": len(seeds),
            "Acc_Ave": float(np.mean(final_accs)),
            "Acc_Med": float(np.median(final_accs)),
            "Acc_Best": float(np.max(final_accs)),
            "Acc_Worst": float(np.min(final_accs)),
            "Acc_Std": float(np.std(final_accs)),
            "MaxAcc_Ave": float(np.mean(max_accs)),
            "MaxAcc_Best": float(np.max(max_accs)),
            "MaxAcc_Worst": float(np.min(max_accs)),
            "Loss_Ave": float(np.mean(final_losses)),
            "Loss_Med": float(np.median(final_losses)),
            "Loss_Best": float(np.min(final_losses)),
            "Loss_Worst": float(np.max(final_losses)),
            "Loss_Std": float(np.std(final_losses)),
            "Comm_Per_Iter": float(comm_arr[:, -1].mean() / rounds),
            "Total_Comm_200r": float(comm_arr[:, -1].mean()),
            "Clip_Ave": float(np.mean(clips_list)),
            "Reset_Ave": float(np.mean(resets_list)),
            "Server_Time_ms": float(np.mean(srv_ms_list)),
            "Server_Mem_MB": mem_mb,
            "Total_Time_s": float(np.mean(total_s_list)),
        }
        for t in targets:
            it_vals = target_iters[t]
            cm_vals = target_comms[t]
            summary_row[f"Reach_{t:.2f}_Rate"] = len(it_vals) / len(seeds)
            summary_row[f"Iter@{t:.2f}_Ave"] = float(np.mean(it_vals)) if it_vals else np.nan
            summary_row[f"Comm@{t:.2f}_Ave"] = float(np.mean(cm_vals)) if cm_vals else np.nan
            summary_row[f"Bytes@{t:.2f}_Ave"] = float(np.mean(cm_vals) * bytes_per_comm) if cm_vals else np.nan
        summary_rows.append(summary_row)

    if history_csv_path and history_frames:
        pd.concat(history_frames, ignore_index=True).to_csv(history_csv_path, index=False)

    pbar.close()
    return pd.DataFrame(summary_rows), pd.DataFrame(per_seed_rows), curves_dict


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


def load_mclr_history_csv(
    history_csv_path: str = "../datasets/ex001_mclr_history.csv",
    log_dir: str = "../datasets/ex001_mclr_logs",
) -> pd.DataFrame:
    """保存済みの MCLR 全履歴CSV（無ければシード別CSV群）を読み込む。"""
    if history_csv_path and os.path.isfile(history_csv_path):
        return pd.read_csv(history_csv_path)
    if log_dir and os.path.isdir(log_dir):
        files = sorted(glob.glob(os.path.join(log_dir, "*.csv")))
        if files:
            return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raise FileNotFoundError(
        f"履歴CSVが見つかりません: history_csv_path={history_csv_path}, log_dir={log_dir}"
    )


def _resolve_mean_only_flag(
    flag: Union[bool, str],
    selected_seeds: Optional[List[int]],
    param_name: str,
) -> bool:
    """'auto' / True / False を判定して平均のみモードの真偽値を返す。"""
    if isinstance(flag, str):
        if flag == "auto":
            return selected_seeds is None
        raise ValueError(f"{param_name} は True, False, または 'auto' を指定してください: {flag!r}")
    if isinstance(flag, bool):
        return flag
    raise ValueError(f"{param_name} は True, False, または 'auto' を指定してください: {flag!r}")


def plot_filtered_mclr_results(
    selected_algos: Optional[List[str]] = None,
    selected_seeds: Optional[List[int]] = None,
    history_csv_path: str = "../datasets/ex001_mclr_history.csv",
    log_dir: str = "../datasets/ex001_mclr_logs",
    x_axis: str = "round",
    show_mean: bool = True,
    plot_mean_only: Union[bool, str] = "auto",
    table_mean_only: Union[bool, str] = "auto",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    保存済み CSV から特定のアルゴリズム・シード値のみに絞って MCLR の学習曲線（損失・精度）を描画する。

    Parameters
    ----------
    selected_algos  : 表示対象アルゴリズム名のリスト（None なら全手法）
    selected_seeds  : 表示対象シード値のリスト（None なら全シード）
    x_axis          : "round" (グローバル反復数) または "comm" (累積通信ラウンド数)
    show_mean       : 複数シード選択時に平均太線を表示するかどうか
    plot_mean_only  : True=平均曲線のみ描画（複数シード時）, False=個別シードも描画, "auto"=selected_seedsがNoneのとき平均のみ
    table_mean_only : True=返却表をアルゴリズム別平均のみに集約, False=シード別の行を出力, "auto"=selected_seedsがNoneのとき平均のみ
    save_path       : 画像保存先パス（None なら画面表示のみ）
    """
    setup_japanese_matplotlib()
    use_plot_mean_only = _resolve_mean_only_flag(plot_mean_only, selected_seeds, "plot_mean_only")
    use_table_mean_only = _resolve_mean_only_flag(table_mean_only, selected_seeds, "table_mean_only")
    df = load_mclr_history_csv(history_csv_path=history_csv_path, log_dir=log_dir)

    if selected_algos is not None:
        df = df[df["Algorithm"].isin(selected_algos)]
    if selected_seeds is not None:
        seed_ints = [int(s) for s in selected_seeds]
        df = df[df["Seed"].isin(seed_ints)]
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
            lbl_m = f"{algo} (Ave, {len(seeds_present)} seeds)"
            axes[0].plot(mean_df[x_col], mean_df["Train_Loss"], color=col, linestyle="-", linewidth=2.3, alpha=0.95)
            axes[1].plot(mean_df[x_col], mean_df["Test_Acc"], color=col, linestyle="-", linewidth=2.3, alpha=0.95, label=lbl_m)

    if len(seeds_present) == 1:
        seed_desc = f"Seed={seeds_present[0]}"
    elif use_plot_mean_only:
        seed_desc = f"全{len(seeds_present)}シード平均のみ" if selected_seeds is None else f"{len(seeds_present)}シード平均のみ (Seeds={seeds_present})"
    else:
        seed_desc = f"Seeds={seeds_present}"
    axes[0].set_title(f"(a) MCLR 訓練損失 絞り込み比較 ({seed_desc})", fontsize=11.5)
    axes[0].set_xlabel(x_label, fontsize=10.5)
    axes[0].set_ylabel("訓練損失 (Train Loss)", fontsize=10.5)
    axes[0].set_yscale("log")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].set_title(f"(b) MCLR テスト精度 絞り込み比較 ({seed_desc})", fontsize=11.5)
    axes[1].set_xlabel(x_label, fontsize=10.5)
    axes[1].set_ylabel("テスト精度 (Test Accuracy)", fontsize=10.5)
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
