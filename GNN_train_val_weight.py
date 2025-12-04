#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_gnn_auto_trainval_pde_weighted.py

- DATA_DIR 内から自動的に pEqn_*_rank{RANK_STR}.dat を走査し、
  TIME_LIST を最大 MAX_NUM_CASES 件まで自動生成。
- その TIME_LIST を train/val に分割して学習。
- 損失は data loss (相対二乗誤差) + mesh-quality-weighted PDE loss。
- 学習中に、loss / data_loss / PDE_loss / rel_err_train / rel_err_val を
  リアルタイムにポップアップ表示。

"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from datetime import datetime
# 日本語フォントを指定（インストール済みのものから選ぶ）
plt.rcParams['font.family'] = 'IPAexGothic'    # or 'Noto Sans CJK JP' など
# マイナス記号が文字化けする場合の対策
plt.rcParams['axes.unicode_minus'] = False

try:
    from torch_geometric.nn import SAGEConv
except ImportError:
    raise RuntimeError(
        "torch_geometric がインストールされていません。"
        "pip install torch-geometric などでインストールしてください。"
    )

# ------------------------------------------------------------
# 設定
# ------------------------------------------------------------

DATA_DIR       = "./gnn"
OUTPUT_DIR     = "./"

RANK_STR       = "7"
NUM_EPOCHS     = 1000
LR             = 1e-3
WEIGHT_DECAY   = 1e-5
MAX_NUM_CASES  = 100   # 自動検出した time のうち先頭 MAX_NUM_CASES 件を使用
TRAIN_FRACTION = 0.8   # 全ケースのうち train に使う割合

LAMBDA_DATA = 0.1
LAMBDA_PDE  = 0.0001

W_PDE_MAX = 10.0  # w_pde の最大値

EPS_DATA = 1e-12  # データ損失用 eps
EPS_RES  = 1e-12  # 残差正規化用 eps
EPS_PLOT = 1e-12  # ★ログプロット用の下限値

RANDOM_SEED = 42  # train/val をランダム分割するためのシード

# 可視化の更新間隔（エポック）
PLOT_INTERVAL = 10

# ログファイル用
LOGGER_FILE = None

def log_print(msg: str):
    """標準出力とログファイル（あれば）の両方に同じメッセージを出力する。"""
    print(msg)
    global LOGGER_FILE
    if LOGGER_FILE is not None:
        print(msg, file=LOGGER_FILE)
        LOGGER_FILE.flush()

# ------------------------------------------------------------
# ユーティリティ: time list 自動検出
# ------------------------------------------------------------

def find_time_list(data_dir: str, rank_str: str):
    times = []
    for fn in os.listdir(data_dir):
        if not fn.startswith("pEqn_"):
            continue
        if not fn.endswith(f"_rank{rank_str}.dat"):
            continue

        core = fn[len("pEqn_") : -len(f"_rank{rank_str}.dat")]
        time_str = core

        x_path   = os.path.join(data_dir, f"x_{time_str}_rank{rank_str}.dat")
        csr_path = os.path.join(data_dir, f"A_csr_{time_str}.dat")
        if os.path.exists(x_path) and os.path.exists(csr_path):
            times.append(time_str)

    times = sorted(set(times), key=lambda s: float(s))
    return times

# ------------------------------------------------------------
# pEqn + CSR + x_true 読み込み
# ------------------------------------------------------------

def load_case_with_csr(data_dir: str, time_str: str, rank_str: str):
    p_path   = os.path.join(data_dir, f"pEqn_{time_str}_rank{rank_str}.dat")
    x_path   = os.path.join(data_dir, f"x_{time_str}_rank{rank_str}.dat")
    csr_path = os.path.join(data_dir, f"A_csr_{time_str}.dat")

    if not os.path.exists(p_path):
        raise FileNotFoundError(p_path)
    if not os.path.exists(x_path):
        raise FileNotFoundError(x_path)
    if not os.path.exists(csr_path):
        raise FileNotFoundError(csr_path)

    with open(p_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    try:
        header_nc = lines[0].split()
        header_nf = lines[1].split()
        assert header_nc[0] == "nCells"
        assert header_nf[0] == "nFaces"
        nCells = int(header_nc[1])
    except Exception as e:
        raise RuntimeError(f"nCells/nFaces ヘッダの解釈に失敗しました: {p_path}\n{e}")

    try:
        idx_cells = next(i for i, ln in enumerate(lines) if ln.startswith("CELLS"))
        idx_edges = next(i for i, ln in enumerate(lines) if ln.startswith("EDGES"))
    except StopIteration:
        raise RuntimeError(f"CELLS/EDGES セクションが見つかりません: {p_path}")

    idx_wall = None
    for i, ln in enumerate(lines):
        if ln.startswith("WALL_FACES"):
            idx_wall = i
            break
    if idx_wall is None:
        idx_wall = len(lines)

    cell_lines = lines[idx_cells + 1: idx_edges]
    edge_lines = lines[idx_edges + 1: idx_wall]

    if len(cell_lines) != nCells:
        log_print(f"[WARN] nCells={nCells} と CELLS 行数={len(cell_lines)} が異なります (time={time_str}).")

    feats_np = np.zeros((len(cell_lines), 13), dtype=np.float32)
    b_np     = np.zeros(len(cell_lines), dtype=np.float32)

    for ln in cell_lines:
        parts = ln.split()
        if len(parts) < 14:
            raise RuntimeError(f"CELLS 行の列数が足りません: {ln}")
        cell_id = int(parts[0])
        xcoord  = float(parts[1])
        ycoord  = float(parts[2])
        zcoord  = float(parts[3])
        diag    = float(parts[4])
        b_val   = float(parts[5])
        skew    = float(parts[6])
        non_ortho  = float(parts[7])
        aspect     = float(parts[8])
        diag_con   = float(parts[9])
        V          = float(parts[10])
        h          = float(parts[11])
        size_jump  = float(parts[12])
        Co         = float(parts[13])

        if not (0 <= cell_id < len(cell_lines)):
            raise RuntimeError(f"cell_id の範囲がおかしいです: {cell_id}")

        feats_np[cell_id, :] = np.array(
            [
                xcoord, ycoord, zcoord,
                diag, b_val, skew, non_ortho, aspect,
                diag_con, V, h, size_jump, Co
            ],
            dtype=np.float32
        )
        b_np[cell_id] = b_val

    e_src = []
    e_dst = []
    for ln in edge_lines:
        parts = ln.split()
        if parts[0] == "WALL_FACES":
            break
        if len(parts) != 5:
            raise RuntimeError(f"EDGES 行の列数が 5 ではありません: {ln}")
        lower = int(parts[1])
        upper = int(parts[2])
        if not (0 <= lower < len(cell_lines) and 0 <= upper < len(cell_lines)):
            raise RuntimeError(f"lower/upper の cell index が範囲外です: {ln}")

        e_src.append(lower)
        e_dst.append(upper)
        e_src.append(upper)
        e_dst.append(lower)

    edge_index_np = np.vstack([
        np.array(e_src, dtype=np.int64),
        np.array(e_dst, dtype=np.int64)
    ])

    x_true_np = np.zeros(len(cell_lines), dtype=np.float32)
    with open(x_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) != 2:
                raise RuntimeError(f"x_*.dat の行形式が想定外です: {ln}")
            cid = int(parts[0])
            val = float(parts[1])
            if not (0 <= cid < len(cell_lines)):
                raise RuntimeError(f"x_*.dat の cell id が範囲外です: {cid}")
            x_true_np[cid] = val

    with open(csr_path, "r") as f:
        csr_lines = [ln.strip() for ln in f if ln.strip()]

    try:
        h0 = csr_lines[0].split()
        h1 = csr_lines[1].split()
        h2 = csr_lines[2].split()
        assert h0[0] == "nRows"
        assert h1[0] == "nCols"
        assert h2[0] == "nnz"
        nRows = int(h0[1])
        nCols = int(h1[1])
        nnz   = int(h2[1])
    except Exception as e:
        raise RuntimeError(f"A_csr_{time_str}.dat のヘッダ解釈に失敗しました: {csr_path}\n{e}")

    if nRows != nCells:
        log_print(f"[WARN] CSR nRows={nRows} と pEqn nCells={nCells} が異なります (time={time_str}).")

    try:
        idx_rowptr = next(i for i, ln in enumerate(csr_lines) if ln.startswith("ROW_PTR"))
        idx_colind = next(i for i, ln in enumerate(csr_lines) if ln.startswith("COL_IND"))
        idx_vals   = next(i for i, ln in enumerate(csr_lines) if ln.startswith("VALUES"))
    except StopIteration:
        raise RuntimeError(f"ROW_PTR/COL_IND/VALUES が見つかりません: {csr_path}")

    row_ptr_str = csr_lines[idx_rowptr + 1].split()
    col_ind_str = csr_lines[idx_colind + 1].split()
    vals_str    = csr_lines[idx_vals + 1].split()

    if len(row_ptr_str) != nRows + 1:
        raise RuntimeError(
            f"ROW_PTR の長さが nRows+1 と一致しません: len={len(row_ptr_str)}, nRows={nRows}"
        )
    if len(col_ind_str) != nnz:
        raise RuntimeError(
            f"COL_IND の長さが nnz と一致しません: len={len(col_ind_str)}, nnz={nnz}"
        )
    if len(vals_str) != nnz:
        raise RuntimeError(
            f"VALUES の長さが nnz と一致しません: len={len(vals_str)}, nnz={nnz}"
        )

    row_ptr_np = np.array(row_ptr_str, dtype=np.int64)
    col_ind_np = np.array(col_ind_str, dtype=np.int64)
    vals_np    = np.array(vals_str,    dtype=np.float32)

    row_idx_np = np.empty(nnz, dtype=np.int64)
    for i in range(nRows):
        start = row_ptr_np[i]
        end   = row_ptr_np[i+1]
        row_idx_np[start:end] = i

    return {
        "time": time_str,
        "feats_np": feats_np,
        "edge_index_np": edge_index_np,
        "x_true_np": x_true_np,
        "b_np": b_np,
        "row_ptr_np": row_ptr_np,
        "col_ind_np": col_ind_np,
        "vals_np": vals_np,
        "row_idx_np": row_idx_np,
    }

# ------------------------------------------------------------
# GNN
# ------------------------------------------------------------

class SimpleSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, 1))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return x.view(-1)

# ------------------------------------------------------------
# CSR Ax
# ------------------------------------------------------------

def matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x):
    y = torch.zeros_like(x)
    y.index_add_(0, row_idx, vals * x[col_ind])
    return y

# ------------------------------------------------------------
# メッシュ品質 w_pde
# ------------------------------------------------------------

def build_w_pde_from_feats(feats_np: np.ndarray,
                           w_pde_max: float = W_PDE_MAX) -> np.ndarray:
    """
    メッシュ品質に基づくPDE損失の重みを計算
    """
    # メトリクス抽出
    skew      = feats_np[:, 5]
    non_ortho = feats_np[:, 6]
    aspect    = feats_np[:, 7]
    size_jump = feats_np[:, 11]

    # 基準値
    SKEW_REF      = 0.2
    NONORTH_REF   = 10.0
    ASPECT_REF    = 5.0
    SIZEJUMP_REF  = 1.5

    # 正規化（0.0〜5.0にクリップ）
    q_skew      = np.clip(skew      / (SKEW_REF + 1e-12),     0.0, 5.0)
    q_non_ortho = np.clip(non_ortho / (NONORTH_REF + 1e-12),  0.0, 5.0)
    q_aspect    = np.clip(aspect    / (ASPECT_REF + 1e-12),   0.0, 5.0)
    q_sizeJump  = np.clip(size_jump / (SIZEJUMP_REF + 1e-12), 0.0, 5.0)

    # 線形結合
    w_raw = (
        1.0
        + 1.0 * (q_skew      - 1.0)
        + 1.0 * (q_non_ortho - 1.0)
        + 1.0 * (q_aspect    - 1.0)
        + 1.0 * (q_sizeJump  - 1.0)
    )

    # クリップ
    w_clipped = np.clip(w_raw, 1.0, w_pde_max)

    return w_clipped.astype(np.float32)

# ------------------------------------------------------------
# raw_case → torch case への変換ヘルパ
# ------------------------------------------------------------

def convert_raw_case_to_torch_case(rc, feat_mean, feat_std, x_mean, x_std, device):
    feats_np  = rc["feats_np"]
    x_true_np = rc["x_true_np"]

    feats_norm     = (feats_np  - feat_mean) / feat_std
    x_true_norm_np = (x_true_np - x_mean)   / x_std

    # ★ ここで w_pde_np を計算
    w_pde_np = build_w_pde_from_feats(feats_np)

    feats       = torch.from_numpy(feats_norm).float().to(device)
    edge_index  = torch.from_numpy(rc["edge_index_np"]).long().to(device)
    x_true      = torch.from_numpy(x_true_np).float().to(device)
    x_true_norm = torch.from_numpy(x_true_norm_np).float().to(device)

    b       = torch.from_numpy(rc["b_np"]).float().to(device)
    row_ptr = torch.from_numpy(rc["row_ptr_np"]).long().to(device)
    col_ind = torch.from_numpy(rc["col_ind_np"]).long().to(device)
    vals    = torch.from_numpy(rc["vals_np"]).float().to(device)
    row_idx = torch.from_numpy(rc["row_idx_np"]).long().to(device)

    w_pde = torch.from_numpy(w_pde_np).float().to(device)

    return {
        "time": rc["time"],
        "feats": feats,
        "edge_index": edge_index,
        "x_true": x_true,
        "x_true_norm": x_true_norm,
        "b": b,
        "row_ptr": row_ptr,
        "col_ind": col_ind,
        "vals": vals,
        "row_idx": row_idx,
        "w_pde": w_pde,
        "w_pde_np": w_pde_np,  # ★ 分布ログ用に numpy を保持しておく
    }

# ------------------------------------------------------------
# 可視化ユーティリティ
# ------------------------------------------------------------

EPS_PLOT = 1e-12  # まだ無ければ定数として追加

def init_plot():
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))

    # タイトルに係数を表示
    fig.suptitle(
#        f"Training Progress "
        f"データ損失係数: {LAMBDA_DATA:g}, PDE損失係数: {LAMBDA_PDE:g}",
        fontsize=12
    )

    # ここでは subplots_adjust は一旦使わない（tight_layout で調整）
    return fig, ax



def update_plot(fig, ax, history):
    ax.clear()

    epochs = np.array(history["epoch"], dtype=np.int32)
    if len(epochs) == 0:
        return

    loss      = np.array(history["loss"], dtype=np.float64)
    data_loss = np.array(history["data_loss"], dtype=np.float64)
    pde_loss  = np.array(history["pde_loss"], dtype=np.float64)
    rel_tr    = np.array(history["rel_err_train"], dtype=np.float64)

    rel_val = np.array(
        [np.nan if v is None else float(v) for v in history["rel_err_val"]],
        dtype=np.float64
    )

    # ★ 追加: 圧力真値・予測値の RMS（訓練データ平均）
    x_true_rms = np.array(history["x_true_rms"], dtype=np.float64)
    x_pred_rms = np.array(history["x_pred_rms"], dtype=np.float64)

    loss_safe      = np.clip(loss,      EPS_PLOT, None)
    data_loss_safe = np.clip(data_loss, EPS_PLOT, None)
    pde_loss_safe  = np.clip(pde_loss,  EPS_PLOT, None)
    rel_tr_safe    = np.clip(rel_tr,    EPS_PLOT, None)

    rel_val_safe = rel_val.copy()
    mask = np.isfinite(rel_val_safe)
    rel_val_safe[mask] = np.clip(rel_val_safe[mask], EPS_PLOT, None)

    # ★ RMS もログスケールで表示できるように下限をクリップ
    x_true_rms_safe = np.clip(x_true_rms, EPS_PLOT, None)
    x_pred_rms_safe = np.clip(x_pred_rms, EPS_PLOT, None)

    # --- プロット ---
    ax.plot(epochs, loss_safe,      label="総損失",                   linewidth=2)
    ax.plot(epochs, data_loss_safe, label="データ損失",              linewidth=1.5, linestyle="--")
    ax.plot(epochs, pde_loss_safe,  label="PDE 損失",               linewidth=1.5, linestyle="--")
    ax.plot(epochs, rel_tr_safe,    label="相対誤差（訓練データ）",    linewidth=1.5)
    ax.plot(epochs, rel_val_safe,   label="相対誤差（テストデータ）",  linewidth=1.5)

    # ★ 追加: 圧力の真値・予測の RMS（訓練データ平均）
    ax.plot(epochs, x_true_rms_safe, label="圧力真値 RMS（訓練）",  linewidth=1.0, linestyle=":")
    ax.plot(epochs, x_pred_rms_safe, label="圧力予測 RMS（訓練）",  linewidth=1.0, linestyle=":")

    ax.set_xlabel("エポック数")
    ax.set_ylabel("損失 / 相対誤差 / 圧力RMS")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    plt.pause(0.01)




# ------------------------------------------------------------
# メイン: train/val 分離版
# ------------------------------------------------------------

def train_gnn_auto_trainval_pde_weighted(data_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global LOGGER_FILE

    # --- ログファイルと実行時間計測のセットアップ ---
    os.makedirs(data_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 係数をファイル名用のタグに変換（例: 0.1 → "0p1", 1e-4 → "0p0001" など）
    lambda_data_tag = str(LAMBDA_DATA).replace('.', 'p')
    lambda_pde_tag  = str(LAMBDA_PDE).replace('.', 'p')

    log_filename = (
#        f"gnn_train_log_"
        f"log_"
        f"DATA{lambda_data_tag}_"
#        f"LP{lambda_pde_tag}_"
        f"PDE{lambda_pde_tag}.txt"
#        f"{timestamp}.txt"
    )
    log_path = os.path.join(OUTPUT_DIR, log_filename)

    LOGGER_FILE = open(log_path, "w", buffering=1)  # 行バッファ

    start_time = time.time()

    log_print(f"[INFO] Logging to {log_path}")
    log_print(f"[INFO] device = {device}")

    # --- time list 検出 & 分割 ---
    all_times = find_time_list(data_dir, RANK_STR)
    if not all_times:
        raise RuntimeError(
            f"{data_dir} 内に pEqn_*_rank{RANK_STR}.dat / x_* / A_csr_* が見つかりませんでした。"
        )

    # 以降の print(...) はすべて log_print(...) に置き換え
    random.seed(RANDOM_SEED)
    random.shuffle(all_times)

    all_times = all_times[:MAX_NUM_CASES]
    n_total = len(all_times)
    n_train = max(1, int(n_total * TRAIN_FRACTION))
    n_val   = n_total - n_train

    time_train = all_times[:n_train]
    time_val   = all_times[n_train:]

    log_print(f"[INFO] 検出された time 数 (使用分) = {n_total}")
    log_print(f"[INFO] train: {n_train} cases, val: {n_val} cases (TRAIN_FRACTION={TRAIN_FRACTION})")
    log_print("=== 使用する train ケース (time, rank) ===")
    for t in time_train:
        log_print(f"  time={t}, rank={RANK_STR}")
    log_print("=== 使用する val ケース (time, rank) ===")
    if time_val:
        for t in time_val:
            log_print(f"  time={t}, rank={RANK_STR}")
    else:
        log_print("  (val ケースなし)")
    log_print("===========================================")


    # --- raw ケース読み込み（train + val 両方） ---
    raw_cases_train = []
    raw_cases_val   = []

    train_set = set(time_train)
    for t in all_times:
        log_print(f"[LOAD] time={t}, rank={RANK_STR} のグラフ+PDE情報を読み込み中...")
        rc = load_case_with_csr(data_dir, t, RANK_STR)
        if t in train_set:
            raw_cases_train.append(rc)
        else:
            raw_cases_val.append(rc)

    # 一貫性チェック
    nCells0 = raw_cases_train[0]["feats_np"].shape[0]
    nFeat   = raw_cases_train[0]["feats_np"].shape[1]
    for rc in raw_cases_train + raw_cases_val:
        if rc["feats_np"].shape[0] != nCells0 or rc["feats_np"].shape[1] != nFeat:
            raise RuntimeError("全ケースで nCells/nFeatures が一致していません。")

    log_print(f"[INFO] nCells (1 ケース目) = {nCells0}, nFeatures = {nFeat}")

    # --- グローバル正規化: train+val 全体で統計を取る ---
    all_feats = np.concatenate(
        [rc["feats_np"] for rc in (raw_cases_train + raw_cases_val)], axis=0
    )
    all_xtrue = np.concatenate(
        [rc["x_true_np"] for rc in (raw_cases_train + raw_cases_val)], axis=0
    )

    feat_mean = all_feats.mean(axis=0, keepdims=True)
    feat_std  = all_feats.std(axis=0, keepdims=True) + 1e-12

    x_mean = all_xtrue.mean()
    x_std  = all_xtrue.std() + 1e-12

    log_print(
        f"[INFO] x_true (all train+val cases): "
        f"min={all_xtrue.min():.3e}, max={all_xtrue.max():.3e}, mean={x_mean:.3e}"
    )

    x_mean_t = torch.tensor(x_mean, dtype=torch.float32, device=device)
    x_std_t  = torch.tensor(x_std,  dtype=torch.float32, device=device)

    # --- torch ケース化 & w_pde 統計 ---
    cases_train = []
    cases_val   = []
    w_all_list  = []

    for rc in raw_cases_train:
        cs = convert_raw_case_to_torch_case(rc, feat_mean, feat_std, x_mean, x_std, device)
        cases_train.append(cs)
        w_all_list.append(cs["w_pde_np"].reshape(-1))

    for rc in raw_cases_val:
        cs = convert_raw_case_to_torch_case(rc, feat_mean, feat_std, x_mean, x_std, device)
        cases_val.append(cs)
        w_all_list.append(cs["w_pde_np"].reshape(-1))

    # --- w_pde の分布ログ（全 train+val ケースまとめ） ---
    if w_all_list:
        w_all = np.concatenate(w_all_list, axis=0)

        w_min  = float(w_all.min())
        w_max  = float(w_all.max())
        w_mean = float(w_all.mean())
        p50    = float(np.percentile(w_all, 50))
        p90    = float(np.percentile(w_all, 90))
        p99    = float(np.percentile(w_all, 99))

        log_print("=== w_pde (mesh-quality weights) statistics over all train+val cases ===")
        log_print(f"  count = {w_all.size}")
        log_print(f"  min   = {w_min:.3e}")
        log_print(f"  mean  = {w_mean:.3e}")
        log_print(f"  max   = {w_max:.3e}")
        log_print(f"  p50   = {p50:.3e}")
        log_print(f"  p90   = {p90:.3e}")
        log_print(f"  p99   = {p99:.3e}")
        log_print("==========================================================================")

    num_train = len(cases_train)
    num_val   = len(cases_val)

    # --- モデル定義 ---
    model = SimpleSAGE(in_channels=nFeat, hidden_channels=64, num_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    log_print("=== Training start (relative data loss + weighted PDE loss, train/val split) ===")

    # --- 可視化用の準備 ---
    fig, ax = init_plot()
    history = {
        "epoch": [],
        "loss": [],
        "data_loss": [],
        "pde_loss": [],
        "rel_err_train": [],
        "rel_err_val": [],  # val が無いときは None
        # ★ 追加: 圧力の真値・予測値の RMS（訓練データ平均）
        "x_true_rms": [],
        "x_pred_rms": [],
    }


    # --- 学習ループ ---
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        total_data_loss = 0.0
        total_pde_loss  = 0.0
        sum_rel_err_tr  = 0.0
        sum_R_pred_tr   = 0.0
        sum_rmse_tr     = 0.0

        # ★ 追加: 真値・予測の RMS の合計（後で平均をとる）
        sum_x_true_rms_tr = 0.0
        sum_x_pred_rms_tr = 0.0

        # -------- train で勾配計算 --------
        for cs in cases_train:
            feats       = cs["feats"]
            edge_index  = cs["edge_index"]
            x_true      = cs["x_true"]
            b           = cs["b"]
            row_ptr     = cs["row_ptr"]
            col_ind     = cs["col_ind"]
            vals        = cs["vals"]
            row_idx     = cs["row_idx"]
            w_pde       = cs["w_pde"]

            # モデルは正規化スケールで出力
            x_pred_norm = model(feats, edge_index)
            # 非正規化スケールに戻す
            x_pred = x_pred_norm * x_std_t + x_mean_t

            # データ損失: 相対二乗誤差
            diff = x_pred - x_true
            data_loss_case = torch.sum(diff * diff) / (torch.sum(x_true * x_true) + EPS_DATA)

            # PDE 損失: w_pde 付き相対残差²
            Ax = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_pred)
            r  = Ax - b

            sqrt_w = torch.sqrt(w_pde)
            wr = sqrt_w * r
            wb = sqrt_w * b
            norm_wr = torch.norm(wr)
            norm_wb = torch.norm(wb) + EPS_RES
            R_pred = norm_wr / norm_wb
            pde_loss_case = R_pred * R_pred

            total_data_loss = total_data_loss + data_loss_case
            total_pde_loss  = total_pde_loss  + pde_loss_case

            with torch.no_grad():
                N = x_true.shape[0]
                rel_err_case = torch.norm(diff) / (torch.norm(x_true) + EPS_DATA)
                rmse_case    = torch.sqrt(torch.sum(diff * diff) / N)

                sum_rel_err_tr += rel_err_case.item()
                sum_R_pred_tr  += R_pred.detach().item()
                sum_rmse_tr    += rmse_case.item()

                # ★ 追加: x_true, x_pred の RMS を計算して加算
                rms_x_true = torch.sqrt(torch.sum(x_true * x_true) / N)
                rms_x_pred = torch.sqrt(torch.sum(x_pred * x_pred) / N)
                sum_x_true_rms_tr += rms_x_true.item()
                sum_x_pred_rms_tr += rms_x_pred.item()


        total_data_loss = total_data_loss / num_train
        total_pde_loss  = total_pde_loss  / num_train
        loss = LAMBDA_DATA * total_data_loss + LAMBDA_PDE * total_pde_loss

        # ★ 追加: 訓練データに対する RMS の平均
        avg_x_true_rms_tr = sum_x_true_rms_tr / num_train
        avg_x_pred_rms_tr = sum_x_pred_rms_tr / num_train

        loss.backward()
        optimizer.step()


        # --- ロギング（train + val） ---
        if epoch % PLOT_INTERVAL == 0 or epoch == 1:
            avg_rel_err_tr = sum_rel_err_tr / num_train
            avg_R_pred_tr  = sum_R_pred_tr / num_train
            avg_rmse_tr    = sum_rmse_tr / num_train

            avg_rel_err_val = None
            avg_R_pred_val  = None
            avg_rmse_val    = None

            if num_val > 0:
                model.eval()
                sum_rel_err_val = 0.0
                sum_R_pred_val  = 0.0
                sum_rmse_val    = 0.0
                with torch.no_grad():
                    for cs in cases_val:
                        feats      = cs["feats"]
                        edge_index = cs["edge_index"]
                        x_true     = cs["x_true"]
                        b          = cs["b"]
                        row_ptr    = cs["row_ptr"]
                        col_ind    = cs["col_ind"]
                        vals       = cs["vals"]
                        row_idx    = cs["row_idx"]
                        w_pde      = cs["w_pde"]

                        x_pred_norm = model(feats, edge_index)
                        x_pred = x_pred_norm * x_std_t + x_mean_t

                        diff = x_pred - x_true
                        rel_err = torch.norm(diff) / (torch.norm(x_true) + EPS_DATA)
                        N = x_true.shape[0]
                        rmse  = torch.sqrt(torch.sum(diff * diff) / N)

                        Ax = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_pred)
                        r  = Ax - b
                        sqrt_w = torch.sqrt(w_pde)
                        wr = sqrt_w * r
                        wb = sqrt_w * b
                        norm_wr = torch.norm(wr)
                        norm_wb = torch.norm(wb) + EPS_RES
                        R_pred = norm_wr / norm_wb

                        sum_rel_err_val += rel_err.item()
                        sum_R_pred_val  += R_pred.item()
                        sum_rmse_val    += rmse.item()

                avg_rel_err_val = sum_rel_err_val / num_val
                avg_R_pred_val  = sum_R_pred_val  / num_val
                avg_rmse_val    = sum_rmse_val    / num_val

            # 履歴に追加
            history["epoch"].append(epoch)
            history["loss"].append(loss.item())
            history["data_loss"].append((LAMBDA_DATA * total_data_loss).item())
            history["pde_loss"].append((LAMBDA_PDE * total_pde_loss).item())
            history["rel_err_train"].append(avg_rel_err_tr)
            history["rel_err_val"].append(avg_rel_err_val)  # None の可能性あり

            # ★ 追加: 圧力真値/予測値の RMS（訓練平均）
            history["x_true_rms"].append(avg_x_true_rms_tr)
            history["x_pred_rms"].append(avg_x_pred_rms_tr)


            # プロット更新
            update_plot(fig, ax, history)

            # コンソールログ
            log = (
                f"[Epoch {epoch:5d}] loss={loss.item():.4e}, "
                f"data_loss={LAMBDA_DATA * total_data_loss:.4e}, "
                f"PDE_loss={LAMBDA_PDE * total_pde_loss:.4e}, "
                f"rel_err_train(avg)={avg_rel_err_tr:.4e}, "
#                f"RMSE_train(avg)={avg_rmse_tr:.4e}, "
#                f"R_pred_train(avg)={avg_R_pred_tr:.4e}"
            )
            if avg_rel_err_val is not None:
                log += (
#                    f", rel_err_val(avg)={avg_rel_err_val:.4e}, "
                    f", rel_err_val(avg)={avg_rel_err_val:.4e} "
#                    f"RMSE_val(avg)={avg_rmse_val:.4e}, "
#                    f"R_pred_val(avg)={avg_R_pred_val:.4e}"
                )
            log_print(log)

    # 学習終了後、インタラクティブモードを解除してウィンドウを保持したい場合はコメントアウト解除
    # plt.ioff()
    # plt.show()

    # --- 最終プロットの保存 ---
    # すべての history を使って最終状態の図を更新・保存
    if len(history["epoch"]) > 0:
#        lambda_data_tag = str(LAMBDA_DATA).replace('.', 'p')
#        lambda_pde_tag  = str(LAMBDA_PDE).replace('.', 'p')
        final_plot_filename = (
            f"training_history_"
#            f"LD{lambda_data_tag}_"
            f"DATA{lambda_data_tag}_"
#            f"LP{lambda_pde_tag}_"
            f"PDE{lambda_pde_tag}.png"
#            f"{timestamp}.png"
        )
        final_plot_path = os.path.join(OUTPUT_DIR, final_plot_filename)

        update_plot(fig, ax, history)
        fig.savefig(final_plot_path, dpi=200, bbox_inches='tight')
        log_print(f"[INFO] Training history figure saved to {final_plot_path}")


    # --- 実行時間の計測結果をログ出力 ---
    elapsed = time.time() - start_time
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = elapsed % 60.0
    log_print(
        f"[INFO] Total elapsed time: {elapsed:.2f} s "
        f"(~{h:02d}:{m:02d}:{s:05.2f})"
    )

    # ログファイルをクローズ
    if LOGGER_FILE is not None:
        LOGGER_FILE.close()
        LOGGER_FILE = None

    # --- 最終評価: OpenFOAM 解との PDE 残差比較を含む ---
    log_print("\n=== Final diagnostics (train cases) ===")
    model.eval()
    for cs in cases_train:
        time_str   = cs["time"]
        feats      = cs["feats"]
        edge_index = cs["edge_index"]
        x_true     = cs["x_true"]
        b          = cs["b"]
        row_ptr    = cs["row_ptr"]
        col_ind    = cs["col_ind"]
        vals       = cs["vals"]
        row_idx    = cs["row_idx"]
        w_pde      = cs["w_pde"]

        with torch.no_grad():
            x_pred_norm = model(feats, edge_index)
            x_pred = x_pred_norm * x_std_t + x_mean_t
            diff = x_pred - x_true
            N = x_true.shape[0]

            rel_err = torch.norm(diff) / (torch.norm(x_true) + EPS_DATA)
            rmse    = torch.sqrt(torch.sum(diff * diff) / N)

            # 学習で使った weighted PDE 残差
            Ax_pred_w = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_pred)
            r_pred_w  = Ax_pred_w - b
            sqrt_w    = torch.sqrt(w_pde)
            wr_pred   = sqrt_w * r_pred_w
            wb        = sqrt_w * b
            norm_wr   = torch.norm(wr_pred)
            norm_wb   = torch.norm(wb) + EPS_RES
            R_pred_w  = norm_wr / norm_wb

            # 物理的な（非加重）PDE 残差: GNN 解
            Ax_pred = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_pred)
            r_pred  = Ax_pred - b
            norm_r_pred    = torch.norm(r_pred)
            max_abs_r_pred = torch.max(torch.abs(r_pred))
            norm_b         = torch.norm(b)
            norm_Ax_pred   = torch.norm(Ax_pred)
            R_pred_over_b  = norm_r_pred / (norm_b + EPS_RES)
            R_pred_over_Ax = norm_r_pred / (norm_Ax_pred + EPS_RES)

            # 物理的な（非加重）PDE 残差: OpenFOAM 解
            Ax_true = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_true)
            r_true  = Ax_true - b
            norm_r_true    = torch.norm(r_true)
            max_abs_r_true = torch.max(torch.abs(r_true))
            norm_Ax_true   = torch.norm(Ax_true)
            R_true_over_b  = norm_r_true / (norm_b + EPS_RES)
            R_true_over_Ax = norm_r_true / (norm_Ax_true + EPS_RES)

        log_print(
            f"  [train] Case (time={time_str}, rank={RANK_STR}): "
            f"rel_err = {rel_err.item():.4e}, RMSE = {rmse.item():.4e}, "
            f"R_pred(weighted) = {R_pred_w.item():.4e}"
        )
        log_print(f"    x_true: min={x_true.min().item():.6e}, max={x_true.max().item():.6e}, "
              f"mean={x_true.mean().item():.6e}, norm={torch.norm(x_true).item():.6e}")
        log_print(f"    x_pred: min={x_pred.min().item():.6e}, max={x_pred.max().item():.6e}, "
              f"mean={x_pred.mean().item():.6e}, norm={torch.norm(x_pred).item():.6e}")
        log_print(f"    x_pred_norm: min={x_pred_norm.min().item():.6e}, "
              f"max={x_pred_norm.max().item():.6e}, mean={x_pred_norm.mean().item():.6e}")
        log_print(f"    diff (x_pred - x_true): norm={torch.norm(diff).item():.6e}")
        log_print(f"    正規化パラメータ: x_mean={x_mean_t.item():.6e}, x_std={x_std_t.item():.6e}")

        log_print("    [PDE residual comparison vs OpenFOAM]")
        log_print(
            "      GNN : "
            f"||r||_2={norm_r_pred.item():.6e}, "
            f"max|r_i|={max_abs_r_pred.item():.6e}, "
            f"||r||/||b||={R_pred_over_b.item():.5f}, "
            f"||r||/||Ax||={R_pred_over_Ax.item():.5f}"
        )
        log_print(
            "      OF  : "
            f"||r||_2={norm_r_true.item():.6e}, "
            f"max|r_i|={max_abs_r_true.item():.6e}, "
            f"||r||/||b||={R_true_over_b.item():.5f}, "
            f"||r||/||Ax||={R_true_over_Ax.item():.5f}"
        )

        # 予測結果の書き出し
        x_pred_np = x_pred.cpu().numpy().reshape(-1)
        out_path = os.path.join(OUTPUT_DIR, f"x_pred_train_{time_str}_rank{RANK_STR}.dat")
        with open(out_path, "w") as f:
            for i, val in enumerate(x_pred_np):
                f.write(f"{i} {val:.9e}\n")
        log_print(f"    [INFO] train x_pred を {out_path} に書き出しました。")

    if num_val > 0:
        log_print("\n=== Final diagnostics (val cases) ===")
        for cs in cases_val:
            time_str   = cs["time"]
            feats      = cs["feats"]
            edge_index = cs["edge_index"]
            x_true     = cs["x_true"]
            b          = cs["b"]
            row_ptr    = cs["row_ptr"]
            col_ind    = cs["col_ind"]
            vals       = cs["vals"]
            row_idx    = cs["row_idx"]
            w_pde      = cs["w_pde"]

            with torch.no_grad():
                x_pred_norm = model(feats, edge_index)
                x_pred = x_pred_norm * x_std_t + x_mean_t
                diff = x_pred - x_true
                N = x_true.shape[0]

                rel_err = torch.norm(diff) / (torch.norm(x_true) + EPS_DATA)
                rmse    = torch.sqrt(torch.sum(diff * diff) / N)

                Ax_pred_w = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_pred)
                r_pred_w  = Ax_pred_w - b
                sqrt_w    = torch.sqrt(w_pde)
                wr_pred   = sqrt_w * r_pred_w
                wb        = sqrt_w * b
                norm_wr   = torch.norm(wr_pred)
                norm_wb   = torch.norm(wb) + EPS_RES
                R_pred_w  = norm_wr / norm_wb

                Ax_pred = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_pred)
                r_pred  = Ax_pred - b
                norm_r_pred    = torch.norm(r_pred)
                max_abs_r_pred = torch.max(torch.abs(r_pred))
                norm_b         = torch.norm(b)
                norm_Ax_pred   = torch.norm(Ax_pred)
                R_pred_over_b  = norm_r_pred / (norm_b + EPS_RES)
                R_pred_over_Ax = norm_r_pred / (norm_Ax_pred + EPS_RES)

                Ax_true = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_true)
                r_true  = Ax_true - b
                norm_r_true    = torch.norm(r_true)
                max_abs_r_true = torch.max(torch.abs(r_true))
                norm_Ax_true   = torch.norm(Ax_true)
                R_true_over_b  = norm_r_true / (norm_b + EPS_RES)
                R_true_over_Ax = norm_r_true / (norm_Ax_true + EPS_RES)

            log_print(
                f"  [val]   Case (time={time_str}, rank={RANK_STR}): "
                f"rel_err = {rel_err.item():.4e}, RMSE = {rmse.item():.4e}, "
                f"R_pred(weighted) = {R_pred_w.item():.4e}"
            )
            log_print(f"    x_true: min={x_true.min().item():.6e}, max={x_true.max().item():.6e}, "
                  f"mean={x_true.mean().item():.6e}, norm={torch.norm(x_true).item():.6e}")
            log_print(f"    x_pred: min={x_pred.min().item():.6e}, max={x_pred.max().item():.6e}, "
                  f"mean={x_pred.mean().item():.6e}, norm={torch.norm(x_pred).item():.6e}")
            log_print(f"    x_pred_norm: min={x_pred_norm.min().item():.6e}, "
                  f"max={x_pred_norm.max().item():.6e}, mean={x_pred_norm.mean().item():.6e}")
            log_print(f"    diff (x_pred - x_true): norm={torch.norm(diff).item():.6e}")
            log_print(f"    正規化パラメータ: x_mean={x_mean_t.item():.6e}, x_std={x_std_t.item():.6e}")

            log_print("    [PDE residual comparison vs OpenFOAM]")
            log_print(
                "      GNN : "
                f"||r||_2={norm_r_pred.item():.6e}, "
                f"max|r_i|={max_abs_r_pred.item():.6e}, "
                f"||r||/||b||={R_pred_over_b.item():.5f}, "
                f"||r||/||Ax||={R_pred_over_Ax.item():.5f}"
            )
            log_print(
                "      OF  : "
                f"||r||_2={norm_r_true.item():.6e}, "
                f"max|r_i|={max_abs_r_true.item():.6e}, "
                f"||r||/||b||={R_true_over_b.item():.5f}, "
                f"||r||/||Ax||={R_true_over_Ax.item():.5f}"
            )

            x_pred_np = x_pred.cpu().numpy().reshape(-1)
            out_path = os.path.join(OUTPUT_DIR, f"x_pred_val_{time_str}_rank{RANK_STR}.dat")
            with open(out_path, "w") as f:
                for i, val in enumerate(x_pred_np):
                    f.write(f"{i} {val:.9e}\n")
            log_print(f"    [INFO] val x_pred を {out_path} に書き出しました。")

if __name__ == "__main__":
    train_gnn_auto_trainval_pde_weighted(DATA_DIR)

