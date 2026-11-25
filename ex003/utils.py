import math
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = os.path.join(tempfile.gettempdir(), "fedsso_moq_mplconfig")
    os.makedirs(_mpl_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _mpl_dir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ex001 / ex002 の実績ある実装を再利用しつつ独立実行も可能にする
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EX001_DIR = os.path.join(_ROOT_DIR, "ex001")
_EX002_DIR = os.path.join(_ROOT_DIR, "ex002")

import importlib.util

def _load_module(mod_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


DEFAULT_SEEDS: List[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]


def setup_japanese_matplotlib() -> None:
    """Matplotlib で日本語フォントとマイナス記号が正しく表示されるように設定する。"""
    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def get_runners():
    """ex001 (MCLR) と ex002 (CompactCNN) のデータ分割・実験実行関数群を読み込む。"""
    if _EX001_DIR not in sys.path:
        sys.path.append(_EX001_DIR)
    ex1_utils = _load_module("ex001_utils_ref", os.path.join(_EX001_DIR, "utils.py"))

    # ex002のmodel.pyを一時的に参照するためパスを調整
    orig_sys_path = list(sys.path)
    sys.path.insert(0, _EX002_DIR)
    ex2_model = _load_module("model", os.path.join(_EX002_DIR, "model.py"))
    sys.modules["model"] = ex2_model
    ex2_utils = _load_module("ex002_utils_ref", os.path.join(_EX002_DIR, "utils.py"))
    sys.path = orig_sys_path
    return ex1_utils, ex2_model, ex2_utils


def run_momentum_sensitivity_study(
    ex1_utils,
    ex2_model,
    ex2_utils,
    packed_mclr, X_tr_m, y_tr_m, X_te_m, y_te_m,
    packed_cnn, X_tr_c, y_tr_c, X_te_c, y_te_c,
    seeds: List[int] = DEFAULT_SEEDS,
    rounds_mclr: int = 200,
    rounds_cnn: int = 200,
) -> pd.DataFrame:
    """
    実験 3A: 慣性係数 mu (MoQ/NAQ) および上限キャップ max_mu (AdaMoQ/AdaNAQ) の感度分析。
    mu, max_mu in {0.1, 0.2, ..., 0.9} に加え、基準線 FedSSO (mu=0.0) を
    MCLR (200r, 10シード) および CompactCNN (200r, 10シード) で比較する。
    """
    rows: List[Dict[str, Any]] = []
    mus = [round(0.1 * i, 1) for i in range(1, 10)]

    configs: List[Tuple[str, float, float]] = [("FedSSO", 0.0, 0.0)]
    for m in mus:
        configs.append(("FedSSO-MoQ", m, m))
    for mm in mus:
        configs.append(("FedSSO-AdaMoQ", 0.0, mm))
    for m in mus:
        configs.append(("FedSSO-NAQ", m, m))
    for mm in mus:
        configs.append(("FedSSO-AdaNAQ", 0.0, mm))

    pbar = tqdm(total=len(configs) * 2 * len(seeds), desc="Exp 3A: Momentum Sensitivity (MoQ & NAQ, 200r)")

    def _run_model(model_label: str, rounds_n: int, lr_val: float, eta_val: float) -> None:
        for algo, mu_val, max_mu_val in configs:
            accs, losses, clips, resets = [], [], [], []
            diverged_cnt = 0
            for s in seeds:
                if model_label == "MCLR":
                    r = ex1_utils.run_fedsso_moq_mclr(
                        algo=algo, packed_clients=packed_mclr,
                        X_train=X_tr_m, y_train=y_tr_m, X_test=X_te_m, y_test=y_te_m,
                        rounds=rounds_n, lr=lr_val, eta=eta_val, mu=mu_val, max_mu=max_mu_val,
                        seed=s, eval_final_only=True, measure_server_time=False,
                    )
                else:
                    r = ex2_utils.run_fedsso_moq_nn(
                        algo=algo, model_cls=ex2_model.CompactCNN, packed_clients=packed_cnn,
                        X_train=X_tr_c, y_train=y_tr_c, X_test=X_te_c, y_test=y_te_c,
                        rounds=rounds_n, lr=lr_val, eta=eta_val, mu=mu_val, max_mu=max_mu_val,
                        seed=s, eval_final_only=True, measure_server_time=False,
                    )
                f_acc = float(r["test_acc"][-1])
                f_loss = float(r["train_loss"][-1])
                if (not np.isfinite(f_loss)) or f_acc < 0.20:
                    diverged_cnt += 1
                accs.append(f_acc)
                losses.append(f_loss)
                clips.append(r["clip_count"])
                resets.append(r["reset_count"])
                pbar.update(1)

            is_fixed = algo in ("FedSSO", "FedSSO-MoQ", "FedSSO-NAQ")
            param_val = mu_val if is_fixed else max_mu_val
            finite_losses = [v for v in losses if np.isfinite(v)]
            rows.append({
                "Model": model_label,
                "Algorithm": algo,
                "Param_Name": "mu" if is_fixed else "max_mu",
                "Param_Val": param_val,
                "Acc_Ave": float(np.mean(accs)),
                "Acc_Med": float(np.median(accs)),
                "Acc_Best": float(np.max(accs)),
                "Acc_Worst": float(np.min(accs)),
                "Acc_Std": float(np.std(accs)),
                "Loss_Ave": float(np.mean(finite_losses)) if finite_losses else float("nan"),
                "Diverged_Seeds": diverged_cnt,
                "Clip_Ave": float(np.mean(clips)),
                "Reset_Ave": float(np.mean(resets)),
            })

    _run_model("MCLR", rounds_mclr, lr_val=0.001, eta_val=1.0)
    _run_model("CompactCNN", rounds_cnn, lr_val=0.07, eta_val=0.3)

    pbar.close()
    return pd.DataFrame(rows)


def run_memory_size_study(
    ex1_utils,
    packed_mclr, X_tr_m, y_tr_m, X_te_m, y_te_m,
    seeds: List[int] = DEFAULT_SEEDS,
    rounds: int = 200,
    m_values: Tuple[int, ...] = (1, 2, 4, 8, 16, 32),
) -> pd.DataFrame:
    """
    実験 3B: 記憶制限準ニュートン法 (FedSSO-LBFGS, FedSSO-LNAQ, FedSSO-LMoQ) の
    履歴数 m in {1, 2, 4, 8, 16, 32} の影響分析 (MCLR, 200ラウンド, 10シード)。
    """
    algos = ["FedSSO-LBFGS", "FedSSO-LNAQ", "FedSSO-LMoQ"]
    rows: List[Dict[str, Any]] = []
    pbar = tqdm(total=len(algos) * len(m_values) * len(seeds), desc="Exp 3B: Memory Size m (2^0..2^5, 200r)")

    for algo in algos:
        for m in m_values:
            accs, losses, comms, srv_ms, clips = [], [], [], [], []
            for s in seeds:
                r = ex1_utils.run_fedsso_moq_mclr(
                    algo=algo, packed_clients=packed_mclr,
                    X_train=X_tr_m, y_train=y_tr_m, X_test=X_te_m, y_test=y_te_m,
                    rounds=rounds, lr=0.001, eta=1.0, max_mu=0.4, m_size=m,
                    seed=s, eval_final_only=True, measure_server_time=True,
                )
                accs.append(r["test_acc"][-1])
                losses.append(r["train_loss"][-1])
                comms.append(r["comm_rounds"][-1])
                srv_ms.append(r["avg_server_time_ms"])
                clips.append(r["clip_count"])
                pbar.update(1)
            rows.append({
                "Algorithm": algo,
                "m": m,
                "Acc_Ave": float(np.mean(accs)),
                "Acc_Med": float(np.median(accs)),
                "Acc_Best": float(np.max(accs)),
                "Acc_Worst": float(np.min(accs)),
                "Acc_Std": float(np.std(accs)),
                "Loss_Ave": float(np.mean(losses)),
                "Total_Comm_Ave": float(np.mean(comms)),
                "Server_Time_ms": float(np.mean(srv_ms)),
                "Server_Mem_MB": ex1_utils.estimate_server_memory_mb(algo, 7850, m_size=m),
                "Clip_Ave": float(np.mean(clips)),
            })

    pbar.close()
    return pd.DataFrame(rows)


def run_curvature_ablation_study(
    ex2_model,
    ex2_utils,
    packed_cnn, X_tr_c, y_tr_c, X_te_c, y_te_c,
    seeds: List[int] = DEFAULT_SEEDS,
    rounds: int = 200,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    実験 3C: 正定値性保証（曲率クリッピング lam, Lam）のアブレーション検証。
    非凸 CompactCNN (200r, 10シード) において MoQ系 + NAQ系 全7手法で 4 条件を比較する。
    """
    algos = [
        "FedSSO",
        "FedSSO-MoQ", "FedSSO-LMoQ", "FedSSO-MLMoQ",
        "FedSSO-NAQ", "FedSSO-LNAQ", "FedSSO-MLNAQ",
    ]
    conditions = [
        {"cond": "標準 (1e-4, 9999)",    "lam": 1e-4, "Lam": 9999.0, "enforce_pd": True},
        {"cond": "狭帯域 (1e-2, 500)",   "lam": 1e-2, "Lam": 500.0,  "enforce_pd": True},
        {"cond": "過大上限 (1e-4, 1e7)", "lam": 1e-4, "Lam": 1e7,    "enforce_pd": True},
        {"cond": "正定値保証なし",        "lam": 1e-4, "Lam": 9999.0, "enforce_pd": False},
    ]

    rows: List[Dict[str, Any]] = []
    curves: Dict[str, np.ndarray] = {}
    pbar = tqdm(total=len(algos) * len(conditions) * len(seeds), desc="Exp 3C: Curvature Ablation (200r)")

    for algo in algos:
        for c in conditions:
            accs, max_accs, losses, clips, resets = [], [], [], [], []
            acc_curves = []
            diverged_cnt = 0
            for s in seeds:
                r = ex2_utils.run_fedsso_moq_nn(
                    algo=algo, model_cls=ex2_model.CompactCNN, packed_clients=packed_cnn,
                    X_train=X_tr_c, y_train=y_tr_c, X_test=X_te_c, y_test=y_te_c,
                    rounds=rounds, lr=0.07, eta=0.3, mu=0.25, max_mu=0.30,
                    lam=c["lam"], Lam=c["Lam"], enforce_pd=c["enforce_pd"], seed=s,
                    eval_final_only=False, eval_train_final_only=True, measure_server_time=False,
                )
                f_acc = r["test_acc"][-1]
                f_loss = r["train_loss"][-1]
                if not np.isfinite(f_loss) or f_acc < 0.20:
                    diverged_cnt += 1
                accs.append(f_acc)
                max_accs.append(float(np.nanmax(r["test_acc"])))
                losses.append(f_loss)
                clips.append(r["clip_count"])
                resets.append(r["reset_count"])
                acc_curves.append(r["test_acc"])
                pbar.update(1)

            key = f"{algo} | {c['cond']}"
            curves[key] = np.mean(np.array(acc_curves, dtype=np.float64), axis=0)

            rows.append({
                "Algorithm": algo,
                "Condition": c["cond"],
                "Acc_Ave": float(np.mean(accs)),
                "Acc_Med": float(np.median(accs)),
                "Acc_Best": float(np.max(accs)),
                "Acc_Worst": float(np.min(accs)),
                "Acc_Std": float(np.std(accs)),
                "MaxAcc_Ave": float(np.mean(max_accs)),
                "Loss_Ave": float(np.nanmean(losses)) if np.isfinite(losses).any() else float("nan"),
                "Diverged_Seeds": diverged_cnt,
                "Clip_Ave": float(np.mean(clips)),
                "Reset_Ave": float(np.mean(resets)),
            })

    pbar.close()
    return pd.DataFrame(rows), curves


def compute_theoretical_scaling(
    d_values: Tuple[int, ...] = (1000, 5000, 10000, 50000, 100000, 500000, 1000000),
    m_size: int = 10,
) -> pd.DataFrame:
    """
    実験 3D: モデルパラメータ次元 d に対するサーバ側メモリ量 (MB, float32) と
    1ラウンドあたりサーバ演算量 (FLOPs近似) の理論スケーリングを算出する。
    """
    rows: List[Dict[str, Any]] = []
    for d in d_values:
        # サーバメモリ (MB, float32 = 4 Bytes)
        mem_full_sso = (d * d + 4 * d) * 4.0 / (1024.0 ** 2)
        mem_full_moq = (d * d + 7 * d) * 4.0 / (1024.0 ** 2)
        mem_lbfgs = ((2 * m_size + 4) * d) * 4.0 / (1024.0 ** 2)
        mem_lmoq = ((2 * m_size + 7) * d) * 4.0 / (1024.0 ** 2)
        mem_mlmoq = (9 * d) * 4.0 / (1024.0 ** 2)

        # サーバ1反復計算量 (乗加算要素数: Full=O(d^2), L=O(m d), ML=O(d))
        ops_full_sso = 6 * d * d + 4 * d
        ops_full_moq = 6 * d * d + 8 * d
        ops_lmoq = (8 * m_size + 8) * d
        ops_mlmoq = 14 * d

        rows.append({
            "d": d,
            "Mem_FedSSO_MB": mem_full_sso,
            "Mem_AdaMoQ_MB": mem_full_moq,
            "Mem_LBFGS_MB": mem_lbfgs,
            "Mem_LMoQ_MB": mem_lmoq,
            "Mem_MLMoQ_MB": mem_mlmoq,
            "Ops_FedSSO": ops_full_sso,
            "Ops_AdaMoQ": ops_full_moq,
            "Ops_LMoQ": ops_lmoq,
            "Ops_MLMoQ": ops_mlmoq,
        })
    return pd.DataFrame(rows)
