import glob
import math
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union

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
    per_seed_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    実験 3A: 慣性係数 mu (MoQ/NAQ) および上限キャップ max_mu (AdaMoQ/AdaNAQ) の感度分析。
    mu, max_mu in {0.1, 0.2, ..., 0.9} に加え、基準線 FedSSO (mu=0.0) を
    MCLR (200r, 10シード) および CompactCNN (200r, 10シード) で比較する。
    per_seed_csv_path を指定するとシード別結果CSVを保存する。
    """
    rows: List[Dict[str, Any]] = []
    per_seed_rows: List[Dict[str, Any]] = []
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
            is_fixed = algo in ("FedSSO", "FedSSO-MoQ", "FedSSO-NAQ")
            param_name = "mu" if is_fixed else "max_mu"
            param_val = mu_val if is_fixed else max_mu_val

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
                is_div = int((not np.isfinite(f_loss)) or f_acc < 0.20)
                if is_div:
                    diverged_cnt += 1
                accs.append(f_acc)
                losses.append(f_loss)
                clips.append(r["clip_count"])
                resets.append(r["reset_count"])
                per_seed_rows.append({
                    "Model": model_label,
                    "Algorithm": algo,
                    "Param_Name": param_name,
                    "Param_Val": param_val,
                    "Seed": int(s),
                    "Final_Acc": f_acc,
                    "Final_Loss": f_loss,
                    "Diverged": is_div,
                    "Clip_Count": int(r["clip_count"]),
                    "Reset_Count": int(r["reset_count"]),
                })
                pbar.update(1)

            finite_losses = [v for v in losses if np.isfinite(v)]
            rows.append({
                "Model": model_label,
                "Algorithm": algo,
                "Param_Name": param_name,
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

    if per_seed_csv_path and per_seed_rows:
        parent_dir = os.path.dirname(os.path.abspath(per_seed_csv_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        pd.DataFrame(per_seed_rows).to_csv(per_seed_csv_path, index=False)

    pbar.close()
    return pd.DataFrame(rows)


def run_memory_size_study(
    ex1_utils,
    packed_mclr, X_tr_m, y_tr_m, X_te_m, y_te_m,
    seeds: List[int] = DEFAULT_SEEDS,
    rounds: int = 200,
    m_values: Tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    per_seed_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    実験 3B: 記憶制限準ニュートン法 (FedSSO-LBFGS, FedSSO-LNAQ, FedSSO-LMoQ) の
    履歴数 m in {1, 2, 4, 8, 16, 32} の影響分析 (MCLR, 200ラウンド, 10シード)。
    per_seed_csv_path を指定するとシード別結果CSVを保存する。
    """
    algos = ["FedSSO-LBFGS", "FedSSO-LNAQ", "FedSSO-LMoQ"]
    rows: List[Dict[str, Any]] = []
    per_seed_rows: List[Dict[str, Any]] = []
    pbar = tqdm(total=len(algos) * len(m_values) * len(seeds), desc="Exp 3B: Memory Size m (2^0..2^5, 200r)")

    for algo in algos:
        for m in m_values:
            accs, losses, comms, srv_ms, clips = [], [], [], [], []
            mem_mb = ex1_utils.estimate_server_memory_mb(algo, 7850, m_size=m)
            for s in seeds:
                r = ex1_utils.run_fedsso_moq_mclr(
                    algo=algo, packed_clients=packed_mclr,
                    X_train=X_tr_m, y_train=y_tr_m, X_test=X_te_m, y_test=y_te_m,
                    rounds=rounds, lr=0.001, eta=1.0, max_mu=0.4, m_size=m,
                    seed=s, eval_final_only=True, measure_server_time=True,
                )
                f_acc = float(r["test_acc"][-1])
                f_loss = float(r["train_loss"][-1])
                t_comm = int(r["comm_rounds"][-1])
                srv_t = float(r["avg_server_time_ms"])
                c_cnt = int(r["clip_count"])
                accs.append(f_acc)
                losses.append(f_loss)
                comms.append(t_comm)
                srv_ms.append(srv_t)
                clips.append(c_cnt)
                per_seed_rows.append({
                    "Algorithm": algo,
                    "m": int(m),
                    "Seed": int(s),
                    "Final_Acc": f_acc,
                    "Final_Loss": f_loss,
                    "Total_Comm": t_comm,
                    "Server_Time_ms": srv_t,
                    "Server_Mem_MB": mem_mb,
                    "Clip_Count": c_cnt,
                })
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
                "Server_Mem_MB": mem_mb,
                "Clip_Ave": float(np.mean(clips)),
            })

    if per_seed_csv_path and per_seed_rows:
        parent_dir = os.path.dirname(os.path.abspath(per_seed_csv_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        pd.DataFrame(per_seed_rows).to_csv(per_seed_csv_path, index=False)

    pbar.close()
    return pd.DataFrame(rows)


def run_curvature_ablation_study(
    ex2_model,
    ex2_utils,
    packed_cnn, X_tr_c, y_tr_c, X_te_c, y_te_c,
    seeds: List[int] = DEFAULT_SEEDS,
    rounds: int = 200,
    per_seed_csv_path: Optional[str] = None,
    history_csv_path: Optional[str] = None,
    save_log_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    実験 3C: 正定値性保証（曲率クリッピング lam, Lam）のアブレーション検証。
    非凸 CompactCNN (200r, 10シード) において MoQ系 + NAQ系 全7手法で 4 条件を比較する。
    per_seed_csv_path / history_csv_path / save_log_dir を指定するとシード別CSVおよび学習曲線履歴CSVを保存する。
    """
    algos = [
        "FedSSO",
        "FedSSO-MoQ", "FedSSO-LMoQ", "FedSSO-MLMoQ",
        "FedSSO-NAQ", "FedSSO-LNAQ", "FedSSO-MLNAQ",
    ]
    conditions = [
        {"cond": "標準 (1e-4, 9999)",    "tag": "standard", "lam": 1e-4, "Lam": 9999.0, "enforce_pd": True},
        {"cond": "狭帯域 (1e-2, 500)",   "tag": "narrow",   "lam": 1e-2, "Lam": 500.0,  "enforce_pd": True},
        {"cond": "過大上限 (1e-4, 1e7)", "tag": "wide",     "lam": 1e-4, "Lam": 1e7,    "enforce_pd": True},
        {"cond": "正定値保証なし",        "tag": "no_pd",    "lam": 1e-4, "Lam": 9999.0, "enforce_pd": False},
    ]

    if save_log_dir:
        os.makedirs(save_log_dir, exist_ok=True)
    for p in (per_seed_csv_path, history_csv_path):
        if p:
            parent_dir = os.path.dirname(os.path.abspath(p))
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    per_seed_rows: List[Dict[str, Any]] = []
    history_frames: List[pd.DataFrame] = []
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
                f_acc = float(r["test_acc"][-1])
                m_acc = float(np.nanmax(r["test_acc"]))
                f_loss = float(r["train_loss"][-1])
                is_div = int((not np.isfinite(f_loss)) or f_acc < 0.20)
                if is_div:
                    diverged_cnt += 1
                accs.append(f_acc)
                max_accs.append(m_acc)
                losses.append(f_loss)
                clips.append(r["clip_count"])
                resets.append(r["reset_count"])
                acc_curves.append(r["test_acc"])

                per_seed_rows.append({
                    "Algorithm": algo,
                    "Condition": c["cond"],
                    "Seed": int(s),
                    "Final_Acc": f_acc,
                    "Max_Acc": m_acc,
                    "Final_Loss": f_loss,
                    "Diverged": is_div,
                    "Clip_Count": int(r["clip_count"]),
                    "Reset_Count": int(r["reset_count"]),
                })

                if save_log_dir or history_csv_path:
                    df_hist = pd.DataFrame({
                        "Algorithm": algo,
                        "Condition": c["cond"],
                        "Seed": int(s),
                        "Round": np.arange(1, len(r["test_acc"]) + 1, dtype=int),
                        "Test_Acc": [float(v) for v in r["test_acc"]],
                    })
                    if save_log_dir:
                        df_hist.to_csv(os.path.join(save_log_dir, f"{algo}_{c['tag']}_seed{s}.csv"), index=False)
                    if history_csv_path:
                        history_frames.append(df_hist)
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

    if per_seed_csv_path and per_seed_rows:
        pd.DataFrame(per_seed_rows).to_csv(per_seed_csv_path, index=False)
    if history_csv_path and history_frames:
        pd.concat(history_frames, ignore_index=True).to_csv(history_csv_path, index=False)

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


DEFAULT_COLOR_MAP_3: Dict[str, str] = {
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


def _format_seed_desc(seeds_present: List[int], selected_seeds: Optional[List[int]], use_plot_mean_only: bool) -> str:
    if len(seeds_present) == 1:
        return f"Seed={seeds_present[0]}"
    if use_plot_mean_only:
        return f"全{len(seeds_present)}シード平均のみ" if selected_seeds is None else f"{len(seeds_present)}シード平均のみ (Seeds={seeds_present})"
    return f"Seeds={seeds_present}"


def plot_filtered_3a_momentum(
    selected_model: str = "CompactCNN",
    selected_algos: Optional[List[str]] = None,
    selected_seeds: Optional[List[int]] = None,
    per_seed_csv_path: str = "../datasets/ex003_momentum_sensitivity_per_seed.csv",
    show_mean: bool = True,
    plot_mean_only: Union[bool, str] = "auto",
    table_mean_only: Union[bool, str] = "auto",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    実験 3A のシード別CSVから特定モデル・アルゴリズム・シード値に絞って慣性感度グラフを描画する。
    """
    setup_japanese_matplotlib()
    use_plot_mean_only = _resolve_mean_only_flag(plot_mean_only, selected_seeds, "plot_mean_only")
    use_table_mean_only = _resolve_mean_only_flag(table_mean_only, selected_seeds, "table_mean_only")

    df = pd.read_csv(per_seed_csv_path)
    df = df[df["Model"] == selected_model]
    if selected_algos is not None:
        df = df[df["Algorithm"].isin(selected_algos)]
    if selected_seeds is not None:
        df = df[df["Seed"].isin([int(s) for s in selected_seeds])]
    if df.empty:
        raise ValueError("実験 3A: 指定された条件に一致するデータがありません。")

    seeds_present = sorted(int(s) for s in df["Seed"].unique())
    algos_order = ["FedSSO", "FedSSO-MoQ", "FedSSO-AdaMoQ", "FedSSO-NAQ", "FedSSO-AdaNAQ"]
    algos_present = (
        [a for a in selected_algos if a in set(df["Algorithm"])]
        if selected_algos is not None
        else [a for a in algos_order if a in set(df["Algorithm"])] + [a for a in sorted(df["Algorithm"].unique()) if a not in algos_order]
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    multi_seed = len(seeds_present) > 1
    draw_mean = multi_seed and (show_mean or use_plot_mean_only)
    draw_individuals = (not multi_seed) or (not use_plot_mean_only)
    seed_linestyles = ["-", "--", "-.", ":"]

    if "FedSSO" in algos_present:
        base_df = df[df["Algorithm"] == "FedSSO"]
        base_mean = float(base_df["Final_Acc"].mean())
        base_lbl = (
            f"FedSSO (μ=0, Seed={seeds_present[0]}={base_mean:.4f})"
            if len(seeds_present) == 1
            else f"FedSSO (μ=0 Ave={base_mean:.4f})"
        )
        ax.axhline(base_mean, color="#333333", linestyle=":", linewidth=1.8, label=base_lbl)

    for algo in algos_present:
        if algo == "FedSSO":
            continue
        col = DEFAULT_COLOR_MAP_3.get(algo, "#333333")
        sub_a = df[df["Algorithm"] == algo].sort_values(["Seed", "Param_Val"])
        if draw_individuals:
            for idx_s, s in enumerate(seeds_present):
                sub_s = sub_a[sub_a["Seed"] == s]
                if sub_s.empty:
                    continue
                ls = seed_linestyles[idx_s % len(seed_linestyles)] if (multi_seed and len(seeds_present) <= 4) else "-"
                lw = 1.1 if draw_mean else 2.0
                alpha = 0.50 if draw_mean else 0.90
                marker = "o" if not draw_mean else None
                lbl = f"{algo} (seed={s})" if (not multi_seed or len(seeds_present) <= 3 or not draw_mean) else None
                ax.plot(sub_s["Param_Val"], sub_s["Final_Acc"], color=col, linestyle=ls, linewidth=lw, alpha=alpha, marker=marker, label=lbl)

        if draw_mean:
            m_df = sub_a.groupby("Param_Val", as_index=False)["Final_Acc"].mean().sort_values("Param_Val")
            ax.plot(m_df["Param_Val"], m_df["Final_Acc"], color=col, linestyle="-", linewidth=2.4, marker="o", label=f"{algo} (Ave, {len(seeds_present)} seeds)")

    seed_desc = _format_seed_desc(seeds_present, selected_seeds, use_plot_mean_only)
    ax.set_title(f"実験 3A 絞り込み ({selected_model}, 200r, {seed_desc})", fontsize=11.5)
    ax.set_xlabel("慣性パラメータ (μ または μ_max)", fontsize=10.5)
    ax.set_ylabel("最終テスト精度", fontsize=10.5)
    ax.set_xticks([round(0.1 * i, 1) for i in range(1, 10)])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", fontsize=8.5, ncol=1 if len(algos_present) <= 3 else 2, framealpha=0.92)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    df_out = df.copy()
    df_out["Algorithm"] = pd.Categorical(df_out["Algorithm"], categories=algos_present, ordered=True)
    if use_table_mean_only:
        summary = (
            df_out.groupby(["Model", "Algorithm", "Param_Name", "Param_Val"], as_index=False, observed=True)
            .agg(
                Seeds=("Seed", "count"),
                Acc_Ave=("Final_Acc", "mean"),
                Loss_Ave=("Final_Loss", "mean"),
                Acc_Best=("Final_Acc", "max"),
                Acc_Worst=("Final_Acc", "min"),
            )
            .sort_values(["Algorithm", "Param_Val"])
            .reset_index(drop=True)
        )
        summary["Algorithm"] = summary["Algorithm"].astype(str)
        return summary
    cols = ["Model", "Algorithm", "Param_Name", "Param_Val", "Seed", "Final_Acc", "Final_Loss"]
    if "Diverged" in df_out.columns:
        cols.append("Diverged")
    out_df = df_out[cols].sort_values(["Algorithm", "Param_Val", "Seed"]).reset_index(drop=True)
    out_df["Algorithm"] = out_df["Algorithm"].astype(str)
    return out_df


def plot_filtered_3b_memory(
    selected_algos: Optional[List[str]] = None,
    selected_seeds: Optional[List[int]] = None,
    per_seed_csv_path: str = "../datasets/ex003_memory_size_per_seed.csv",
    show_mean: bool = True,
    plot_mean_only: Union[bool, str] = "auto",
    table_mean_only: Union[bool, str] = "auto",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    実験 3B のシード別CSVから特定のアルゴリズム・シード値に絞って履歴数 m の影響グラフを描画する。
    """
    setup_japanese_matplotlib()
    use_plot_mean_only = _resolve_mean_only_flag(plot_mean_only, selected_seeds, "plot_mean_only")
    use_table_mean_only = _resolve_mean_only_flag(table_mean_only, selected_seeds, "table_mean_only")

    df = pd.read_csv(per_seed_csv_path)
    if selected_algos is not None:
        df = df[df["Algorithm"].isin(selected_algos)]
    if selected_seeds is not None:
        df = df[df["Seed"].isin([int(s) for s in selected_seeds])]
    if df.empty:
        raise ValueError("実験 3B: 指定された条件に一致するデータがありません。")

    seeds_present = sorted(int(s) for s in df["Seed"].unique())
    algos_order = ["FedSSO-LBFGS", "FedSSO-LNAQ", "FedSSO-LMoQ"]
    algos_present = (
        [a for a in selected_algos if a in set(df["Algorithm"])]
        if selected_algos is not None
        else [a for a in algos_order if a in set(df["Algorithm"])] + [a for a in sorted(df["Algorithm"].unique()) if a not in algos_order]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    multi_seed = len(seeds_present) > 1
    draw_mean = multi_seed and (show_mean or use_plot_mean_only)
    draw_individuals = (not multi_seed) or (not use_plot_mean_only)
    seed_linestyles = ["-", "--", "-.", ":"]
    m_ticks = sorted(int(m) for m in df["m"].unique())
    m_labels = [str(m) for m in m_ticks]

    for algo in algos_present:
        col = DEFAULT_COLOR_MAP_3.get(algo, "#333333")
        sub_a = df[df["Algorithm"] == algo].sort_values(["Seed", "m"])
        if draw_individuals:
            for idx_s, s in enumerate(seeds_present):
                sub_s = sub_a[sub_a["Seed"] == s]
                if sub_s.empty:
                    continue
                ls = seed_linestyles[idx_s % len(seed_linestyles)] if (multi_seed and len(seeds_present) <= 4) else "-"
                lw = 1.1 if draw_mean else 2.0
                alpha = 0.50 if draw_mean else 0.90
                marker = "o" if not draw_mean else None
                lbl = f"{algo} (seed={s})" if (not multi_seed or len(seeds_present) <= 3 or not draw_mean) else None
                axes[0].plot(sub_s["m"], sub_s["Final_Acc"], color=col, linestyle=ls, linewidth=lw, alpha=alpha, marker=marker, label=lbl)
                axes[1].plot(sub_s["m"], sub_s["Server_Time_ms"], color=col, linestyle=ls, linewidth=lw, alpha=alpha, marker="s" if marker else None)

        if draw_mean:
            m_df = sub_a.groupby("m", as_index=False)[["Final_Acc", "Server_Time_ms"]].mean().sort_values("m")
            axes[0].plot(m_df["m"], m_df["Final_Acc"], color=col, linestyle="-", linewidth=2.4, marker="o", label=f"{algo} (Ave, {len(seeds_present)} seeds)")
            axes[1].plot(m_df["m"], m_df["Server_Time_ms"], color=col, linestyle="-", linewidth=2.4, marker="s", label=f"{algo} (Ave)")

    seed_desc = _format_seed_desc(seeds_present, selected_seeds, use_plot_mean_only)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(m_ticks)
        ax.set_xticklabels(m_labels)
        ax.set_xlim(min(m_ticks) * 0.82, max(m_ticks) * 1.18)
        ax.set_xlabel("履歴数 m (log2 スケール)", fontsize=10.5)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_title(f"(a) 実験 3B テスト精度 絞り込み ({seed_desc})", fontsize=11.5)
    axes[0].set_ylabel("最終テスト精度", fontsize=10.5)
    axes[0].legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    axes[1].set_title(f"(b) 実験 3B サーバ計算時間 絞り込み ({seed_desc})", fontsize=11.5)
    axes[1].set_ylabel("平均サーバ計算時間 (ms/iter)", fontsize=10.5)
    if draw_mean:
        axes[1].legend(loc="upper left", fontsize=8.5, framealpha=0.92)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    df_out = df.copy()
    df_out["Algorithm"] = pd.Categorical(df_out["Algorithm"], categories=algos_present, ordered=True)
    if use_table_mean_only:
        summary = (
            df_out.groupby(["Algorithm", "m"], as_index=False, observed=True)
            .agg(
                Seeds=("Seed", "count"),
                Acc_Ave=("Final_Acc", "mean"),
                Loss_Ave=("Final_Loss", "mean"),
                Acc_Best=("Final_Acc", "max"),
                Acc_Worst=("Final_Acc", "min"),
                Server_Time_ms=("Server_Time_ms", "mean"),
            )
            .sort_values(["Algorithm", "m"])
            .reset_index(drop=True)
        )
        summary["Algorithm"] = summary["Algorithm"].astype(str)
        return summary
    cols = ["Algorithm", "m", "Seed", "Final_Acc", "Final_Loss", "Server_Time_ms"]
    out_df = df_out[cols].sort_values(["Algorithm", "m", "Seed"]).reset_index(drop=True)
    out_df["Algorithm"] = out_df["Algorithm"].astype(str)
    return out_df


def plot_filtered_3c_ablation(
    selected_algos: Optional[List[str]] = None,
    selected_conditions: Optional[List[str]] = None,
    selected_seeds: Optional[List[int]] = None,
    history_csv_path: str = "../datasets/ex003_curvature_ablation_history.csv",
    log_dir: str = "../datasets/ex003_curvature_logs",
    show_mean: bool = True,
    plot_mean_only: Union[bool, str] = "auto",
    table_mean_only: Union[bool, str] = "auto",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    実験 3C の学習曲線履歴CSV（無ければシード別CSV群）から特定のアルゴリズム・条件・シード値に絞って学習曲線を描画する。
    """
    setup_japanese_matplotlib()
    use_plot_mean_only = _resolve_mean_only_flag(plot_mean_only, selected_seeds, "plot_mean_only")
    use_table_mean_only = _resolve_mean_only_flag(table_mean_only, selected_seeds, "table_mean_only")

    if history_csv_path and os.path.isfile(history_csv_path):
        df = pd.read_csv(history_csv_path)
    elif log_dir and os.path.isdir(log_dir):
        files = sorted(glob.glob(os.path.join(log_dir, "*.csv")))
        if not files:
            raise FileNotFoundError(f"3C 履歴CSVが見つかりません: {history_csv_path} / {log_dir}")
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    else:
        raise FileNotFoundError(f"3C 履歴CSVが見つかりません: {history_csv_path} / {log_dir}")

    if selected_algos is not None:
        df = df[df["Algorithm"].isin(selected_algos)]
    if selected_conditions is not None:
        df = df[df["Condition"].isin(selected_conditions)]
    if selected_seeds is not None:
        df = df[df["Seed"].isin([int(s) for s in selected_seeds])]
    if df.empty:
        raise ValueError("実験 3C: 指定された条件に一致するデータがありません。")

    algos_order = [
        "FedSSO",
        "FedSSO-MoQ", "FedSSO-LMoQ", "FedSSO-MLMoQ",
        "FedSSO-NAQ", "FedSSO-LNAQ", "FedSSO-MLNAQ",
    ]
    algos_present = (
        [a for a in selected_algos if a in set(df["Algorithm"])]
        if selected_algos is not None
        else [a for a in algos_order if a in set(df["Algorithm"])] + [a for a in sorted(df["Algorithm"].unique()) if a not in algos_order]
    )
    cond_styles = [
        ("標準 (1e-4, 9999)",    "#1f77b4", "-"),
        ("狭帯域 (1e-2, 500)",   "#2ca02c", "--"),
        ("過大上限 (1e-4, 1e7)", "#ff7f0e", "-."),
        ("正定値保証なし",        "#d62728", ":"),
    ]
    conds_present = [c for c in cond_styles if c[0] in set(df["Condition"])]
    seeds_present = sorted(int(s) for s in df["Seed"].unique())
    multi_seed = len(seeds_present) > 1
    draw_mean = multi_seed and (show_mean or use_plot_mean_only)
    draw_individuals = (not multi_seed) or (not use_plot_mean_only)

    n_alg = len(algos_present)
    ncols = min(4, n_alg)
    nrows = (n_alg + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.4 * nrows), squeeze=False)

    seed_desc = _format_seed_desc(seeds_present, selected_seeds, use_plot_mean_only)
    for idx_a, algo in enumerate(algos_present):
        ax = axes[idx_a // ncols][idx_a % ncols]
        sub_a = df[df["Algorithm"] == algo]
        for cond_name, col, ls in conds_present:
            sub_c = sub_a[sub_a["Condition"] == cond_name].sort_values(["Seed", "Round"])
            if sub_c.empty:
                continue
            if draw_individuals:
                for s in seeds_present:
                    sub_s = sub_c[sub_c["Seed"] == s]
                    if sub_s.empty:
                        continue
                    lw = 0.8 if draw_mean else 1.8
                    alpha = 0.35 if draw_mean else 0.90
                    lbl = f"{cond_name} (s={s})" if (not multi_seed or len(seeds_present) <= 2 or not draw_mean) else None
                    ax.plot(sub_s["Round"], sub_s["Test_Acc"], color=col, linestyle=ls, linewidth=lw, alpha=alpha, label=lbl)
            if draw_mean:
                m_df = sub_c.groupby("Round", as_index=False)["Test_Acc"].mean().sort_values("Round")
                ax.plot(m_df["Round"], m_df["Test_Acc"], color=col, linestyle=ls, linewidth=2.1, alpha=0.95, label=f"{cond_name} (Ave)")

        ax.set_title(f"{algo} ({seed_desc})", fontsize=10.5)
        ax.set_xlabel("グローバル反復数 (k)", fontsize=9.5)
        ax.set_ylabel("テスト精度", fontsize=9.5)
        ax.set_ylim(0.08, 1.02)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", fontsize=7.8, framealpha=0.92)

    for idx_empty in range(n_alg, nrows * ncols):
        axes[idx_empty // ncols][idx_empty % ncols].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    max_round = df.groupby(["Algorithm", "Condition", "Seed"])["Round"].transform("max")
    final_df = df[df["Round"] == max_round].copy()
    final_df["Algorithm"] = pd.Categorical(final_df["Algorithm"], categories=algos_present, ordered=True)
    if use_table_mean_only:
        summary = (
            final_df.groupby(["Algorithm", "Condition"], as_index=False, observed=True)
            .agg(
                Seeds=("Seed", "count"),
                Round=("Round", "max"),
                Test_Acc_Ave=("Test_Acc", "mean"),
                Acc_Best=("Test_Acc", "max"),
                Acc_Worst=("Test_Acc", "min"),
            )
            .sort_values(["Algorithm", "Condition"])
            .reset_index(drop=True)
        )
        summary["Algorithm"] = summary["Algorithm"].astype(str)
        return summary
    out_df = (
        final_df[["Algorithm", "Condition", "Seed", "Round", "Test_Acc"]]
        .sort_values(["Algorithm", "Condition", "Seed"])
        .reset_index(drop=True)
    )
    out_df["Algorithm"] = out_df["Algorithm"].astype(str)
    return out_df
