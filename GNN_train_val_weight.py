#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_gnn_auto_trainval_pde_weighted.py

- DATA_DIR 内から自動的に pEqn_*_rank*.dat を走査し、
  全ての (time, rank) ペアを最大 MAX_NUM_CASES 件まで自動生成。
- 複数プロセス（rank）のデータを統合して学習。
- その (time, rank) リストを train/val に分割して学習。
- 損失は data loss (相対二乗誤差) + mesh-quality-weighted PDE loss。
- 追加: pressure のゲージ自由度（定数モード）を hard constraint として除去（centered投影）。
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
from typing import Optional
from mpl_toolkits.mplot3d import Axes3D  # 3Dプロット用（projection="3d"で内部的に使用）
import time
from datetime import datetime
import pickle
import hashlib
from scipy.sparse import csr_matrix
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

EPS = 1.0e-12

# ------------------------------------------------------------
# 設定
# ------------------------------------------------------------

DATA_DIR       = "./data"
OUTPUT_DIR     = "./"
NUM_EPOCHS     = 1000
LR             = 1e-3
WEIGHT_DECAY   = 1e-5
MAX_NUM_CASES  = 100   # 自動検出した time のうち先頭 MAX_NUM_CASES 件を使用
TRAIN_FRACTION = 0.8   # 全ケースのうち train に使う割合
HIDDEN_CHANNELS = 64
NUM_LAYERS      = 4

# 学習率スケジューラ（ReduceLROnPlateau）用パラメータ
USE_LR_SCHEDULER = True
LR_SCHED_FACTOR = 0.5
LR_SCHED_PATIENCE = 20
LR_SCHED_MIN_LR = 1e-6

# 学習率ウォームアップ
USE_LR_WARMUP = True
LR_WARMUP_EPOCHS = 10  # ウォームアップするエポック数

# 勾配クリッピング
USE_GRAD_CLIP = True
GRAD_CLIP_MAX_NORM = 1.0  # 勾配ノルムの最大値

# メモリ効率化オプション
USE_LAZY_LOADING = True   # データをCPUに保持し、使用時のみGPUへ転送
USE_AMP = True            # 混合精度学習（Automatic Mixed Precision）を有効化

# データキャッシュオプション（Optuna等での繰り返し学習を高速化）
USE_DATA_CACHE = True     # データをキャッシュファイルに保存し、2回目以降は高速ロード
CACHE_DIR = ".cache"      # キャッシュファイルの保存先ディレクトリ

LAMBDA_DATA = 0.1
LAMBDA_PDE  = 0.01        # PDE損失の重み（relative正規化では ||r||²/||b||² ≈ 1-20 程度になるため小さめに設定）
LAMBDA_GAUGE = 0.01       # ゲージ正則化係数（教師なし学習時の定数モード抑制用）

W_PDE_MAX = 10.0  # w_pde の最大値
USE_MESH_QUALITY_WEIGHTS = True  # メッシュ品質重みを使用（Falseで全セル等重み w=1）
USE_DIAGONAL_SCALING = True  # 対角スケーリングを適用（条件数改善のため）
# PDE損失の正規化方式
# "relative": ||r||²/||b||² (相対残差ノルム、物理的に意味があり推奨)
# "row_diag": r/diag で行ごと正規化 (値が極小になる問題あり)
# "none": ||r||²/(||Ax||²+||b||²+eps) 正規化
PDE_LOSS_NORMALIZATION = "relative"

EPS_DATA = 1e-12  # データ損失用 eps
EPS_RES  = 1e-8   # 残差正規化用 eps（安定性のため大きめに設定）
EPS_PLOT = 1e-12  # ★ログプロット用の下限値

RANDOM_SEED = 42  # train/val をランダム分割するためのシード

# 可視化の更新間隔（エポック）
PLOT_INTERVAL = 10

# 誤差場可視化用の設定
MAX_ERROR_PLOT_CASES_TRAIN = 3   # train ケースで誤差図を出す最大件数
MAX_ERROR_PLOT_CASES_VAL   = 3   # val ケースで誤差図を出す最大件数
MAX_POINTS_3D_SCATTER      = 50000  # 3D散布図でプロットする最大セル数（それ以上ならランダムサンプリング）
YSLICE_FRACTIONAL_HALF_WIDTH = 0.05  # y中央断面として扱う帯の半幅（全高さに対する 5%）

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
# ユーティリティ: (time, rank) ペアリスト自動検出
# ------------------------------------------------------------

import re
import glob

def find_time_rank_list(data_dir: str):
    """
    DATA_DIR/processor*/gnn/ 内から全ての pEqn_{time}_rank{rank}.dat を走査し、
    対応する A_csr_{time}.dat が存在する (time, rank, gnn_dir) タプルのリストを返す。
    x_{time}_rank{rank}.dat は教師なし学習モードでは省略可能。

    ディレクトリ構造:
        data/
        ├── processor2/gnn/
        │   ├── A_csr_{time}.dat
        │   ├── pEqn_{time}_rank2.dat
        │   └── x_{time}_rank2.dat  # 省略可
        ├── processor4/gnn/
        │   └── ...
        └── ...

    Returns:
        tuple: (time_rank_tuples, missing_files_info)
            - time_rank_tuples: 有効な (time, rank, gnn_dir) タプルのリスト
            - missing_files_info: 見つからなかったファイルの情報（辞書）
    """
    time_rank_tuples = []
    pattern = re.compile(r"^pEqn_(.+)_rank(\d+)\.dat$")

    # 見つからなかったファイルを追跡
    missing_pEqn = []
    missing_csr = []
    missing_x = []  # 警告用（教師なし学習では必須ではない）

    # data/processor*/gnn/ を探索
    gnn_dirs = glob.glob(os.path.join(data_dir, "processor*", "gnn"))

    if not gnn_dirs:
        # gnn ディレクトリ自体が見つからない場合
        return [], {"no_gnn_dirs": True}

    for gnn_dir in gnn_dirs:
        if not os.path.isdir(gnn_dir):
            continue

        for fn in os.listdir(gnn_dir):
            match = pattern.match(fn)
            if not match:
                continue

            time_str = match.group(1)
            rank_str = match.group(2)

            x_path   = os.path.join(gnn_dir, f"x_{time_str}_rank{rank_str}.dat")
            # CSR ファイルは A_csr_{time}.dat または A_csr_{time}_rank{rank}.dat の両形式に対応
            csr_path = os.path.join(gnn_dir, f"A_csr_{time_str}.dat")
            csr_path_with_rank = os.path.join(gnn_dir, f"A_csr_{time_str}_rank{rank_str}.dat")

            has_csr = os.path.exists(csr_path) or os.path.exists(csr_path_with_rank)
            has_x = os.path.exists(x_path)

            if has_csr:
                # pEqn と A_csr があれば有効（x は教師なし学習では省略可）
                time_rank_tuples.append((time_str, rank_str, gnn_dir))
                if not has_x:
                    missing_x.append(x_path)
            else:
                missing_csr.append(csr_path)

    # time の数値順、次に rank の数値順でソート
    time_rank_tuples = sorted(
        set(time_rank_tuples),
        key=lambda tr: (float(tr[0]), int(tr[1]))
    )

    missing_info = {
        "missing_pEqn": missing_pEqn,
        "missing_csr": missing_csr,
        "missing_x": missing_x,
    }

    return time_rank_tuples, missing_info


# ------------------------------------------------------------
# データキャッシュ機能
# ------------------------------------------------------------

def _compute_cache_key(data_dir: str, time_rank_tuples: list) -> str:
    """
    キャッシュのキー（ハッシュ）を計算する。
    データディレクトリと (time, rank, gnn_dir) タプルのリストから一意のキーを生成。
    """
    key_str = data_dir + "|" + str(sorted(time_rank_tuples))
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def _get_cache_path(data_dir: str, time_rank_tuples: list) -> str:
    """キャッシュファイルのパスを取得する。"""
    cache_key = _compute_cache_key(data_dir, time_rank_tuples)
    return os.path.join(CACHE_DIR, f"raw_cases_{cache_key}.pkl")


def _is_cache_valid(cache_path: str, time_rank_tuples: list) -> bool:
    """
    キャッシュが有効かどうかを確認する。
    - キャッシュファイルが存在するか
    - ソースファイルよりキャッシュが新しいか
    """
    if not os.path.exists(cache_path):
        return False

    cache_mtime = os.path.getmtime(cache_path)

    # 各ソースファイルの最終更新時刻をチェック
    for time_str, rank_str, gnn_dir in time_rank_tuples:
        p_path = os.path.join(gnn_dir, f"pEqn_{time_str}_rank{rank_str}.dat")
        x_path = os.path.join(gnn_dir, f"x_{time_str}_rank{rank_str}.dat")
        csr_path = os.path.join(gnn_dir, f"A_csr_{time_str}.dat")
        csr_path_with_rank = os.path.join(gnn_dir, f"A_csr_{time_str}_rank{rank_str}.dat")

        for path in [p_path, x_path]:
            if os.path.exists(path) and os.path.getmtime(path) > cache_mtime:
                return False

        # CSR ファイルは両形式に対応
        if os.path.exists(csr_path) and os.path.getmtime(csr_path) > cache_mtime:
            return False
        if os.path.exists(csr_path_with_rank) and os.path.getmtime(csr_path_with_rank) > cache_mtime:
            return False

    return True


def save_raw_cases_to_cache(raw_cases: list, cache_path: str) -> None:
    """raw_cases をキャッシュファイルに保存する。"""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(raw_cases, f, protocol=pickle.HIGHEST_PROTOCOL)
    log_print(f"[CACHE] データを {cache_path} にキャッシュしました")


def load_raw_cases_from_cache(cache_path: str) -> list:
    """キャッシュファイルから raw_cases を読み込む。"""
    with open(cache_path, "rb") as f:
        raw_cases = pickle.load(f)
    log_print(f"[CACHE] キャッシュ {cache_path} からデータを読み込みました")
    return raw_cases


def compute_affine_fit(x_true_tensor, x_pred_tensor):
    """
    x_true ≈ a * x_pred + b となるように、
    最小二乗で a, b を求める簡易診断用関数。

    Parameters
    ----------
    x_true_tensor : torch.Tensor, shape (N,)
        物理スケールの真値（正規化解除後）
    x_pred_tensor : torch.Tensor, shape (N,)
        物理スケールの予測値（正規化解除後）

    Returns
    -------
    a : float
        最適スケール係数
    b : float
        最適バイアス
    rmse_before : float
        補正前 RMSE = sqrt(mean((x_pred - x_true)^2))
    rmse_after : float
        補正後 RMSE = sqrt(mean((a*x_pred + b - x_true)^2))
    """
    # CPU / numpy に変換して 1 次元にフラット化
    xp = x_pred_tensor.detach().cpu().double().view(-1).numpy()
    yt = x_true_tensor.detach().cpu().double().view(-1).numpy()

    n = xp.size
    if n == 0:
        return 1.0, 0.0, float("nan"), float("nan")

    sx = xp.sum()
    sy = yt.sum()
    sxx = (xp * xp).sum()
    sxy = (xp * yt).sum()

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-30:
        # x_pred がほぼ定数の場合はスケールをいじれないので、そのままとみなす
        a = 1.0
        b = 0.0
    else:
        a = (n * sxy - sx * sy) / denom
        b = (sy - a * sx) / n

    rmse_before = float(np.sqrt(((xp - yt) ** 2).mean()))
    rmse_after = float(np.sqrt(((a * xp + b - yt) ** 2).mean()))

    return a, b, rmse_before, rmse_after


# ------------------------------------------------------------
# pEqn + CSR + x_true 読み込み
# ------------------------------------------------------------

def load_case_with_csr(gnn_dir: str, time_str: str, rank_str: str):
    """
    指定された gnn_dir から (time, rank) に対応するデータを読み込む。

    ファイル形式:
        - pEqn_{time}_rank{rank}.dat
        - x_{time}_rank{rank}.dat
        - A_csr_{time}.dat または A_csr_{time}_rank{rank}.dat
    """
    p_path   = os.path.join(gnn_dir, f"pEqn_{time_str}_rank{rank_str}.dat")
    x_path   = os.path.join(gnn_dir, f"x_{time_str}_rank{rank_str}.dat")

    # CSR ファイルは両形式に対応
    csr_path = os.path.join(gnn_dir, f"A_csr_{time_str}.dat")
    if not os.path.exists(csr_path):
        csr_path = os.path.join(gnn_dir, f"A_csr_{time_str}_rank{rank_str}.dat")

    if not os.path.exists(p_path):
        raise FileNotFoundError(p_path)
    # x ファイルは存在しなくてもよい（教師なし学習モード）
    has_x_true = os.path.exists(x_path)
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

    # x ファイルが存在する場合のみ読み込み（教師あり学習）
    # 存在しない場合は None（教師なし学習 / PINNs モード）
    if has_x_true:
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
    else:
        x_true_np = None

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
        "rank": rank_str,
        "gnn_dir": gnn_dir,
        "feats_np": feats_np,
        "edge_index_np": edge_index_np,
        "x_true_np": x_true_np,
        "has_x_true": has_x_true,  # 教師データの有無フラグ
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
    """
    改善版 GraphSAGE モデル。
    - LayerNorm による正規化
    - 残差接続（Skip connections）
    - 入力射影層（入力次元を隠れ次元に合わせる）
    """
    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 4):
        super().__init__()
        self.num_layers = num_layers

        # 入力射影層（残差接続のため）
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # GraphSAGE 畳み込み層
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

        # 出力層
        self.convs.append(SAGEConv(hidden_channels, 1))

    def forward(self, x, edge_index):
        # 入力射影
        x = self.input_proj(x)
        x = F.relu(x)

        # 中間層（残差接続 + LayerNorm）
        for i, (conv, norm) in enumerate(zip(self.convs[:-1], self.norms)):
            x_res = x  # 残差接続用に保存
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = x + x_res  # 残差接続

        # 出力層（活性化なし）
        x = self.convs[-1](x, edge_index)
        return x.view(-1)

# ------------------------------------------------------------
# CSR Ax
# ------------------------------------------------------------

def matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x):
    """
    CSR 形式の疎行列とベクトルの積を計算する。
    AMP 使用時に型の不一致が発生する場合は、自動的に型を揃える。
    """
    # AMP 使用時、x が half (FP16) で vals が float (FP32) の場合がある
    # 計算精度を保つため、x を vals の型に揃える
    if x.dtype != vals.dtype:
        x = x.to(vals.dtype)

    n_rows = int(row_ptr.numel() - 1)
    y = torch.zeros(n_rows, device=x.device, dtype=x.dtype)
    y.index_add_(0, row_idx, vals * x[col_ind])
    return y

# ------------------------------------------------------------
# 対角スケーリングと条件数推定
# ------------------------------------------------------------

def matvec_csr_numpy(row_ptr, col_ind, vals, x):
    """NumPy版のCSR行列-ベクトル積（scipy.sparse使用）"""
    n = len(row_ptr) - 1
    A = csr_matrix((vals.astype(np.float64), col_ind, row_ptr), shape=(n, n))
    return A @ x.astype(np.float64)


def estimate_condition_number(row_ptr_np, col_ind_np, vals_np, diag_np,
                               max_iter=50, tol=1e-6):
    """
    べき乗法で行列の条件数を推定する。

    Args:
        row_ptr_np, col_ind_np, vals_np: CSR形式の行列
        diag_np: 対角成分（逆べき乗法の前処理用）
        max_iter: 最大反復回数
        tol: 収束判定の閾値

    Returns:
        dict: {
            'lambda_max': 最大固有値（絶対値）,
            'lambda_min': 最小固有値（絶対値）,
            'condition_number': 条件数
        }
    """
    n = len(row_ptr_np) - 1
    eps = 1e-12

    # --- 最大固有値の推定（べき乗法） ---
    x = np.random.randn(n).astype(np.float64)
    x = x / (np.linalg.norm(x) + eps)

    lambda_max = 1.0
    for _ in range(max_iter):
        y = matvec_csr_numpy(row_ptr_np, col_ind_np, vals_np.astype(np.float64), x)
        lambda_new = np.dot(x, y)  # Rayleigh quotient
        y_norm = np.linalg.norm(y) + eps
        x_new = y / y_norm

        if abs(lambda_new - lambda_max) / (abs(lambda_max) + eps) < tol:
            lambda_max = abs(lambda_new)
            break
        lambda_max = abs(lambda_new)
        x = x_new

    # --- 最小固有値の推定（逆べき乗法 + 対角前処理） ---
    # A^(-1) の代わりに D^(-1)A の最小固有値を推定（計算効率のため）
    x = np.random.randn(n).astype(np.float64)
    x = x / (np.linalg.norm(x) + eps)

    diag_inv = 1.0 / (np.abs(diag_np).astype(np.float64) + eps)

    lambda_min = 1.0
    for iteration in range(max_iter):
        # y = D^(-1) * A * x
        Ax = matvec_csr_numpy(row_ptr_np, col_ind_np, vals_np.astype(np.float64), x)
        y = diag_inv * Ax

        lambda_new = np.dot(x, y)  # Rayleigh quotient
        y_norm = np.linalg.norm(y) + eps
        x_new = y / y_norm

        if abs(lambda_new - lambda_min) / (abs(lambda_min) + eps) < tol:
            lambda_min = abs(lambda_new)
            break
        lambda_min = abs(lambda_new)
        x = x_new

    # 対角前処理した行列の条件数から元の条件数を推定
    # 簡易的に、D^(-1)A の固有値範囲を使用
    condition_number = lambda_max / (lambda_min + eps)

    return {
        'lambda_max': lambda_max,
        'lambda_min': lambda_min * np.mean(np.abs(diag_np)),  # 元のスケールに戻す近似
        'condition_number': condition_number,
    }


def apply_diagonal_scaling_csr(row_ptr_np, col_ind_np, vals_np, diag_np, b_np):
    """
    対角スケーリングを CSR 行列と右辺ベクトルに適用する。

    A_scaled = D^(-1/2) * A * D^(-1/2)
    b_scaled = D^(-1/2) * b

    Args:
        row_ptr_np, col_ind_np, vals_np: CSR形式の行列
        diag_np: 対角成分
        b_np: 右辺ベクトル

    Returns:
        tuple: (vals_scaled_np, b_scaled_np, diag_sqrt_np)
            - vals_scaled_np: スケーリング後の行列値
            - b_scaled_np: スケーリング後の右辺ベクトル
            - diag_sqrt_np: sqrt(|diag|)（逆変換用）
    """
    eps = 1e-12
    n = len(diag_np)

    # sqrt(|diag|) を計算
    diag_abs = np.abs(diag_np).astype(np.float64)
    diag_sqrt = np.sqrt(diag_abs + eps).astype(np.float32)
    diag_inv_sqrt = (1.0 / diag_sqrt).astype(np.float32)

    # 行列値のスケーリング: vals_scaled[k] = vals[k] / sqrt(diag[row]) / sqrt(diag[col])
    # ベクトル化: row_ptr から各非ゼロ要素の行インデックスを復元
    row_indices = np.repeat(np.arange(n, dtype=np.int64), np.diff(row_ptr_np))
    vals_scaled = (vals_np * diag_inv_sqrt[row_indices] * diag_inv_sqrt[col_ind_np]).astype(np.float32)

    # 右辺ベクトルのスケーリング: b_scaled = D^(-1/2) * b
    b_scaled = b_np * diag_inv_sqrt

    return vals_scaled, b_scaled, diag_sqrt


# ------------------------------------------------------------
# メッシュ品質 w_pde
# ------------------------------------------------------------

def build_w_pde_from_feats(feats_np: np.ndarray,
                           w_pde_max: float = W_PDE_MAX,
                           use_mesh_quality_weights: bool = USE_MESH_QUALITY_WEIGHTS) -> np.ndarray:
    """
    メッシュ品質に基づくPDE損失の重みを計算

    Args:
        feats_np: 特徴量配列 (N, 13)
        w_pde_max: 重みの最大値
        use_mesh_quality_weights: Trueならメッシュ品質重みを計算、Falseなら全セル等重み(w=1)

    Returns:
        重みベクトル (N,)
    """
    n_cells = feats_np.shape[0]

    # メッシュ品質重みを使用しない場合は全セル等重み
    if not use_mesh_quality_weights:
        return np.ones(n_cells, dtype=np.float32)

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

def convert_raw_case_to_torch_case(rc, feat_mean, feat_std, x_mean, x_std, device, lazy_load=False,
                                    use_diagonal_scaling=USE_DIAGONAL_SCALING):
    """
    raw_case を torch テンソルに変換する。

    Parameters
    ----------
    lazy_load : bool
        True の場合、データを CPU 上に保持し、GPU への転送は行わない。
        学習時に move_case_to_device() で必要なときだけ GPU に転送する。
    use_diagonal_scaling : bool
        True の場合、対角スケーリングを適用して条件数を改善する。
        A_scaled = D^(-1/2) * A * D^(-1/2), b_scaled = D^(-1/2) * b
    """
    feats_np  = rc["feats_np"]
    x_true_np = rc["x_true_np"]
    has_x_true = rc.get("has_x_true", x_true_np is not None)

    # 対角成分を取得（feats_np[:, 3] が対角成分）
    diag_np = feats_np[:, 3].copy()

    # 対角スケーリングの適用（A,bのみをスケール。xは物理スケールのまま保持）
    vals_np = rc["vals_np"]
    b_np = rc["b_np"]
    diag_sqrt_np = None

    if use_diagonal_scaling:
        vals_np, b_np, diag_sqrt_np = apply_diagonal_scaling_csr(
            rc["row_ptr_np"], rc["col_ind_np"], rc["vals_np"], diag_np, rc["b_np"]
        )
        # NOTE:
        # ここでは x_true / x_pred は「物理スケール」のまま保持する。
        # PDE側だけ x_scaled = D^(1/2) x を用いて A_scaled x_scaled = b_scaled を評価する。

    feats_norm = (feats_np - feat_mean) / feat_std

    # x_true が存在する場合のみ正規化（教師あり学習）
    if has_x_true and x_true_np is not None:
        x_true_norm_np = (x_true_np - x_mean) / x_std
    else:
        x_true_norm_np = None

    # ★ ここで w_pde_np を計算
    w_pde_np = build_w_pde_from_feats(feats_np)

    # lazy_load が True の場合は CPU に保持、False の場合は直接 device へ
    target_device = torch.device("cpu") if lazy_load else device

    feats       = torch.from_numpy(feats_norm).float().to(target_device)
    edge_index  = torch.from_numpy(rc["edge_index_np"]).long().to(target_device)

    # x_true が存在する場合のみテンソル化
    if has_x_true and x_true_np is not None:
        x_true      = torch.from_numpy(x_true_np).float().to(target_device)
        x_true_norm = torch.from_numpy(x_true_norm_np).float().to(target_device)
    else:
        x_true      = None
        x_true_norm = None

    b       = torch.from_numpy(b_np).float().to(target_device)
    row_ptr = torch.from_numpy(rc["row_ptr_np"]).long().to(target_device)
    col_ind = torch.from_numpy(rc["col_ind_np"]).long().to(target_device)
    vals    = torch.from_numpy(vals_np).float().to(target_device)
    row_idx = torch.from_numpy(rc["row_idx_np"]).long().to(target_device)

    w_pde = torch.from_numpy(w_pde_np).float().to(target_device)

    # 対角スケーリング用の係数（逆変換用）
    if diag_sqrt_np is not None:
        diag_sqrt = torch.from_numpy(diag_sqrt_np).float().to(target_device)
    else:
        diag_sqrt = None

    # セル体積（ゲージ正則化用）
    volume_np = feats_np[:, 9].copy()
    volume = torch.from_numpy(volume_np).float().to(target_device)

    # 対角成分（行ごと正規化用）
    diag = torch.from_numpy(diag_np).float().to(target_device)

    return {
        "time": rc["time"],
        "rank": rc["rank"],
        "gnn_dir": rc["gnn_dir"],
        "feats": feats,
        "edge_index": edge_index,
        "x_true": x_true,
        "x_true_norm": x_true_norm,
        "has_x_true": has_x_true,  # 教師データの有無フラグ
        "b": b,
        "row_ptr": row_ptr,
        "col_ind": col_ind,
        "vals": vals,
        "row_idx": row_idx,
        "w_pde": w_pde,
        "w_pde_np": w_pde_np,  # ★ 分布ログ用に numpy を保持しておく
        "diag_sqrt": diag_sqrt,  # ★ 対角スケーリングの逆変換用
        "diag_sqrt_np": diag_sqrt_np,  # ★ NumPy版も保持
        "use_diagonal_scaling": use_diagonal_scaling,  # スケーリング適用フラグ
        "volume": volume,  # ★ セル体積（ゲージ正則化用）
        "diag": diag,  # ★ 対角成分（行ごと正規化用）

        # ★ 誤差場可視化用に元の座標・品質指標も持たせる
        "coords_np": feats_np[:, 0:3].copy(),   # [x, y, z]
        "skew_np": feats_np[:, 5].copy(),
        "non_ortho_np": feats_np[:, 6].copy(),
        "aspect_np": feats_np[:, 7].copy(),
        "size_jump_np": feats_np[:, 11].copy(),
    }


def move_case_to_device(cs, device):
    """
    ケースデータを指定デバイスに転送する（遅延ロード用）。
    non_blocking=True で非同期転送を行い、オーバーヘッドを軽減。
    x_true が None の場合（教師なし学習）も対応。
    """
    x_true = cs["x_true"]
    x_true_norm = cs["x_true_norm"]
    has_x_true = cs.get("has_x_true", x_true is not None)
    diag_sqrt = cs.get("diag_sqrt")

    return {
        "time": cs["time"],
        "rank": cs["rank"],
        "gnn_dir": cs["gnn_dir"],
        "feats": cs["feats"].to(device, non_blocking=True),
        "edge_index": cs["edge_index"].to(device, non_blocking=True),
        "x_true": x_true.to(device, non_blocking=True) if x_true is not None else None,
        "x_true_norm": x_true_norm.to(device, non_blocking=True) if x_true_norm is not None else None,
        "has_x_true": has_x_true,
        "b": cs["b"].to(device, non_blocking=True),
        "row_ptr": cs["row_ptr"].to(device, non_blocking=True),
        "col_ind": cs["col_ind"].to(device, non_blocking=True),
        "vals": cs["vals"].to(device, non_blocking=True),
        "row_idx": cs["row_idx"].to(device, non_blocking=True),
        "w_pde": cs["w_pde"].to(device, non_blocking=True),
        "w_pde_np": cs["w_pde_np"],
        "diag_sqrt": diag_sqrt.to(device, non_blocking=True) if diag_sqrt is not None else None,
        "diag_sqrt_np": cs.get("diag_sqrt_np"),
        "use_diagonal_scaling": cs.get("use_diagonal_scaling", False),
        "volume": cs["volume"].to(device, non_blocking=True),  # ★ セル体積
        "diag": cs["diag"].to(device, non_blocking=True),  # ★ 対角成分
        "coords_np": cs["coords_np"],
        "skew_np": cs["skew_np"],
        "non_ortho_np": cs["non_ortho_np"],
        "aspect_np": cs["aspect_np"],
        "size_jump_np": cs["size_jump_np"],
    }


def evaluate_validation_cases(
    model,
    cases_val,
    device,
    x_std_t,
    x_mean_t,
    use_amp_actual,
):
    """
    検証データに対する評価を行う共通関数。

    Returns:
        tuple: (avg_rel_err_val, avg_rmse_val, avg_R_pred_val, num_val_with_x)
    """
    num_val = len(cases_val)
    if num_val == 0:
        return None, None, None, 0

    sum_rel_err_val = 0.0
    sum_R_pred_val = 0.0
    sum_rmse_val = 0.0
    num_val_with_x = 0

    with torch.no_grad():
        for cs in cases_val:
            # 遅延ロードの場合、ケースデータを GPU に転送
            if USE_LAZY_LOADING:
                cs_gpu = move_case_to_device(cs, device)
            else:
                cs_gpu = cs

            feats = cs_gpu["feats"]
            edge_index = cs_gpu["edge_index"]
            x_true = cs_gpu["x_true"]
            b = cs_gpu["b"]
            row_ptr = cs_gpu["row_ptr"]
            col_ind = cs_gpu["col_ind"]
            vals = cs_gpu["vals"]
            row_idx = cs_gpu["row_idx"]
            w_pde = cs_gpu["w_pde"]
            has_x_true = cs_gpu.get("has_x_true", x_true is not None)
            diag_sqrt = cs_gpu.get("diag_sqrt", None)
            use_dscale = cs_gpu.get("use_diagonal_scaling", False) and (diag_sqrt is not None)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp_actual):
                x_pred_norm = model(feats, edge_index)
                x_pred = x_pred_norm * x_std_t + x_mean_t

            # rel_err, RMSE: x_true がある場合のみ計算
            if has_x_true and x_true is not None:
                # ゲージ不変評価: 両者を平均ゼロに正規化してから比較
                x_pred_centered = x_pred - torch.mean(x_pred)
                x_true_centered = x_true - torch.mean(x_true)
                diff = x_pred_centered - x_true_centered
                rel_err = torch.norm(diff) / (torch.norm(x_true_centered) + EPS_DATA)
                N = x_true.shape[0]
                rmse = torch.sqrt(torch.sum(diff * diff) / N)
                sum_rel_err_val += rel_err.item()
                sum_rmse_val += rmse.item()
                num_val_with_x += 1

            x_for_pde = (x_pred * diag_sqrt) if use_dscale else x_pred
            Ax = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_for_pde)
            r = Ax - b
            sqrt_w = torch.sqrt(w_pde)
            wr = sqrt_w * r
            wb = sqrt_w * b
            norm_wr = torch.norm(wr)
            norm_wb = torch.norm(wb) + EPS_RES
            R_pred = norm_wr / norm_wb

            sum_R_pred_val += R_pred.item()

            # 遅延ロードの場合、参照を解放
            if USE_LAZY_LOADING:
                del cs_gpu

    avg_R_pred_val = sum_R_pred_val / num_val
    if num_val_with_x > 0:
        avg_rel_err_val = sum_rel_err_val / num_val_with_x
        avg_rmse_val = sum_rmse_val / num_val_with_x
    else:
        # 教師なし学習: PDE 残差を指標として使用
        avg_rel_err_val = avg_R_pred_val
        avg_rmse_val = 0.0

    return avg_rel_err_val, avg_rmse_val, avg_R_pred_val, num_val_with_x


def save_error_field_plots(cs, x_pred, x_true, prefix, output_dir=OUTPUT_DIR):
    """
    誤差場 (x_pred - x_true) の 3D 散布図と、
    y ≒ 中央断面での 2D カラーマップ（誤差 vs w_pde）を保存する。

    さらに |誤差| と w_pde の簡単な統計（相関係数、誤差上位5%セルの平均w_pde など）をログ出力する。
    """
    # ---- Torch -> NumPy ----
    x_pred_np = x_pred.detach().cpu().numpy().reshape(-1)
    x_true_np = x_true.detach().cpu().numpy().reshape(-1)
    err       = x_pred_np - x_true_np
    abs_err   = np.abs(err)

    coords    = cs["coords_np"]      # (N, 3) : x, y, z
    w_pde_np  = cs["w_pde_np"]       # (N,)

    N = coords.shape[0]
    if err.shape[0] != N:
        log_print(f"    [WARN] 誤差場可視化: 座標数 N={N} と解ベクトル長={err.shape[0]} が一致しません ({prefix})。")
        return

    # ============================
    # 1) 3D 散布図 (x, y, z, color = x_pred - x_true)
    # ============================
    if N > MAX_POINTS_3D_SCATTER:
        idx = np.random.choice(N, size=MAX_POINTS_3D_SCATTER, replace=False)
        log_print(f"    [PLOT] 3D 散布図用に {N} セル中 {idx.size} セルをサンプリングしました ({prefix}).")
    else:
        idx = np.arange(N)

    xs = coords[idx, 0]
    ys = coords[idx, 1]
    zs = coords[idx, 2]
    err_sample = err[idx]

    vmax = np.max(np.abs(err_sample)) + 1e-20
    vmin = -vmax

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xs, ys, zs, c=err_sample, s=2, cmap="coolwarm", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("x_pred - x_true")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"誤差場 3D散布図 ({prefix})")
    fig.tight_layout()

    out3d = os.path.join(output_dir, f"error3d_{prefix}.png")
    fig.savefig(out3d, dpi=200)
    plt.close(fig)

    log_print(f"    [PLOT] 誤差場 3D 散布図を {out3d} に保存しました。")

    # ============================
    # 2) y ≒ 中央断面での 2D カラーマップ
    #    左: |x_pred - x_true|, 右: w_pde
    # ============================
    y = coords[:, 1]
    y_min, y_max = float(y.min()), float(y.max())
    if y_max > y_min:
        y_mid = 0.5 * (y_min + y_max)
        band  = YSLICE_FRACTIONAL_HALF_WIDTH * (y_max - y_min)
    else:
        # 全セル同一 y の場合
        y_mid = y_min
        band  = 1e-6

    mask = np.abs(y - y_mid) <= band
    n_slice = int(np.count_nonzero(mask))

    if n_slice < 10:
        log_print(f"    [PLOT] y≈中央断面のセル数が {n_slice} と少ないため 2D カラーマップをスキップします ({prefix}).")
    else:
        xs2       = coords[mask, 0]
        zs2       = coords[mask, 2]
        abs_err2  = abs_err[mask]
        w_pde2    = w_pde_np[mask]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sc0 = axes[0].scatter(xs2, zs2, c=abs_err2, s=5)
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("z")
        axes[0].set_title("誤差場 |x_pred - x_true| (y ≒ 中央断面)")
        cbar0 = fig.colorbar(sc0, ax=axes[0])
        cbar0.set_label("|x_pred - x_true|")

        sc1 = axes[1].scatter(xs2, zs2, c=w_pde2, s=5)
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("z")
        axes[1].set_title("w_pde (メッシュ品質重み, y ≒ 中央断面)")
        cbar1 = fig.colorbar(sc1, ax=axes[1])
        cbar1.set_label("w_pde")

        fig.tight_layout()
        out2d = os.path.join(output_dir, f"error2d_yMid_{prefix}.png")
        fig.savefig(out2d, dpi=200)
        plt.close(fig)

        log_print(f"    [PLOT] 誤差場と w_pde の 2D カラーマップを {out2d} に保存しました。")

    # ============================
    # 3) |誤差| と w_pde の簡単な統計
    # ============================
    if N >= 10:
        if np.std(abs_err) > 0.0 and np.std(w_pde_np) > 0.0:
            corr = float(np.corrcoef(abs_err, w_pde_np)[0, 1])
        else:
            corr = float("nan")

        top_frac = 0.05  # 誤差上位5%を見る
        k = max(1, int(top_frac * N))
        idx_sorted = np.argsort(-abs_err)  # 大きい順
        top_idx = idx_sorted[:k]

        mean_w_all = float(w_pde_np.mean())
        mean_w_top = float(w_pde_np[top_idx].mean())

        log_print(
            "    [STATS] |誤差| と w_pde の簡易統計: "
            f"corr(|err|, w_pde)={corr:.3f}, "
            f"top{int(top_frac*100)}%誤差セルの平均w_pde={mean_w_top:.3e}, "
            f"全セル平均w_pde={mean_w_all:.3e}"
        )


def save_pressure_comparison_plots(cs, x_pred, x_true, prefix, output_dir=OUTPUT_DIR):
    """
    圧力場の真値と予測値を比較する2D可視化を保存する。

    出力:
    1. 2D断面での圧力場比較（真値、予測値、差分の3パネル）
    2. 散布図（x_true vs x_pred）
    3. PDE残差のヒストグラム
    """
    import warnings

    # ---- Torch -> NumPy ----
    x_pred_np = x_pred.detach().cpu().numpy().reshape(-1)
    x_true_np = x_true.detach().cpu().numpy().reshape(-1)

    # ゲージ補正（平均を揃える）
    x_pred_centered = x_pred_np - np.mean(x_pred_np)
    x_true_centered = x_true_np - np.mean(x_true_np)
    diff = x_pred_centered - x_true_centered

    coords = cs["coords_np"]  # (N, 3): x, y, z
    N = coords.shape[0]

    if x_pred_np.shape[0] != N:
        log_print(f"    [WARN] 可視化: 座標数 N={N} と解ベクトル長={x_pred_np.shape[0]} が一致しません ({prefix})。")
        return

    # ============================
    # 1) 2D断面での圧力場比較（y ≒ 中央）
    # ============================
    y = coords[:, 1]
    y_min, y_max = float(y.min()), float(y.max())
    if y_max > y_min:
        y_mid = 0.5 * (y_min + y_max)
        band = YSLICE_FRACTIONAL_HALF_WIDTH * (y_max - y_min)
    else:
        y_mid = y_min
        band = 1e-6

    mask = np.abs(y - y_mid) <= band
    n_slice = int(np.count_nonzero(mask))

    if n_slice >= 10:
        xs = coords[mask, 0]
        zs = coords[mask, 2]
        true_slice = x_true_centered[mask]
        pred_slice = x_pred_centered[mask]
        diff_slice = diff[mask]

        # カラースケールを統一
        vmin_p = min(true_slice.min(), pred_slice.min())
        vmax_p = max(true_slice.max(), pred_slice.max())
        vabs_diff = max(abs(diff_slice.min()), abs(diff_slice.max()))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 真値
        sc0 = axes[0].scatter(xs, zs, c=true_slice, s=5, cmap="viridis", vmin=vmin_p, vmax=vmax_p)
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("z")
        axes[0].set_title("真値 (x_true, ゲージ補正済み)")
        fig.colorbar(sc0, ax=axes[0], label="Pressure")

        # 予測値
        sc1 = axes[1].scatter(xs, zs, c=pred_slice, s=5, cmap="viridis", vmin=vmin_p, vmax=vmax_p)
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("z")
        axes[1].set_title("予測値 (x_pred, ゲージ補正済み)")
        fig.colorbar(sc1, ax=axes[1], label="Pressure")

        # 差分
        sc2 = axes[2].scatter(xs, zs, c=diff_slice, s=5, cmap="coolwarm", vmin=-vabs_diff, vmax=vabs_diff)
        axes[2].set_aspect("equal", adjustable="box")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("z")
        axes[2].set_title("差分 (x_pred - x_true)")
        fig.colorbar(sc2, ax=axes[2], label="Difference")

        fig.suptitle(f"圧力場比較 ({prefix}, y≒中央断面, n={n_slice}セル)", fontsize=12)
        fig.tight_layout()

        out_compare = os.path.join(output_dir, f"pressure_comparison_{prefix}.png")
        fig.savefig(out_compare, dpi=200)
        plt.close(fig)
        log_print(f"    [PLOT] 圧力場比較図を {out_compare} に保存しました。")

    # ============================
    # 2) 散布図（x_true vs x_pred）
    # ============================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # サンプリング（大規模メッシュ対応）
    if N > MAX_POINTS_3D_SCATTER:
        idx = np.random.choice(N, size=MAX_POINTS_3D_SCATTER, replace=False)
    else:
        idx = np.arange(N)

    # 散布図
    axes[0].scatter(x_true_centered[idx], x_pred_centered[idx], s=1, alpha=0.3, c='blue')

    # 45度線
    lim_min = min(x_true_centered[idx].min(), x_pred_centered[idx].min())
    lim_max = max(x_true_centered[idx].max(), x_pred_centered[idx].max())
    axes[0].plot([lim_min, lim_max], [lim_min, lim_max], 'r-', lw=2, label='y=x (理想)')

    # 回帰直線
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if np.std(x_true_centered[idx]) > 0:
            coef = np.polyfit(x_true_centered[idx], x_pred_centered[idx], 1)
            poly = np.poly1d(coef)
            x_line = np.linspace(lim_min, lim_max, 100)
            axes[0].plot(x_line, poly(x_line), 'g--', lw=1.5,
                        label=f'回帰: y={coef[0]:.4f}x+{coef[1]:.2e}')

    axes[0].set_xlabel("x_true (ゲージ補正済み)")
    axes[0].set_ylabel("x_pred (ゲージ補正済み)")
    axes[0].set_title("真値 vs 予測値")
    axes[0].legend(loc='upper left')
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].grid(True, alpha=0.3)

    # 差分のヒストグラム
    axes[1].hist(diff, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', lw=2, label='ゼロ')
    axes[1].axvline(np.mean(diff), color='green', linestyle='-', lw=2,
                   label=f'平均: {np.mean(diff):.2e}')
    axes[1].set_xlabel("差分 (x_pred - x_true)")
    axes[1].set_ylabel("確率密度")
    axes[1].set_title(f"差分ヒストグラム (std={np.std(diff):.2e})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 相関係数とRMSEを注記
    if np.std(x_true_centered) > 0 and np.std(x_pred_centered) > 0:
        corr = np.corrcoef(x_true_centered, x_pred_centered)[0, 1]
    else:
        corr = float('nan')
    rmse = np.sqrt(np.mean(diff**2))
    rel_err = np.linalg.norm(diff) / (np.linalg.norm(x_true_centered) + 1e-12)

    fig.suptitle(f"予測精度評価 ({prefix}): R={corr:.4f}, RMSE={rmse:.2e}, RelErr={rel_err:.2%}", fontsize=12)
    fig.tight_layout()

    out_scatter = os.path.join(output_dir, f"scatter_comparison_{prefix}.png")
    fig.savefig(out_scatter, dpi=200)
    plt.close(fig)
    log_print(f"    [PLOT] 散布図・ヒストグラムを {out_scatter} に保存しました。")


# ------------------------------------------------------------
# 可視化ユーティリティ
# ------------------------------------------------------------

def init_plot():
    plt.ion()
    # 横に 2 つのサブプロット（左：損失, 右：相対誤差）
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # タイトルに係数を表示
    fig.suptitle(
        f"データ損失係数: {LAMBDA_DATA:g}, PDE損失係数: {LAMBDA_PDE:g}",
        fontsize=12
    )

    # レイアウトは update_plot 側で tight_layout をかける
    return fig, axes

def update_plot(fig, axes, history):
    ax_loss, ax_rel = axes  # 左：損失, 右：相対誤差

    ax_loss.clear()
    ax_rel.clear()

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

    # 下限を切ってログスケールに耐えられるようにする
    loss_safe      = np.clip(loss,      EPS_PLOT, None)
    data_loss_safe = np.clip(data_loss, EPS_PLOT, None)
    pde_loss_safe  = np.clip(pde_loss,  EPS_PLOT, None)
    rel_tr_safe    = np.clip(rel_tr,    EPS_PLOT, None)

    rel_val_safe = rel_val.copy()
    mask = np.isfinite(rel_val_safe)
    rel_val_safe[mask] = np.clip(rel_val_safe[mask], EPS_PLOT, None)

    # --- 左グラフ：損失系（総損失・データ損失・PDE損失） ---
    ax_loss.plot(epochs, loss_safe,      label="総損失",      linewidth=2)
    ax_loss.plot(epochs, data_loss_safe, label="データ損失",  linewidth=1.5, linestyle="--")
    ax_loss.plot(epochs, pde_loss_safe,  label="PDE損失",    linewidth=1.5, linestyle="--")

    ax_loss.set_xlabel("エポック数")
    ax_loss.set_ylabel("損失")
    ax_loss.set_yscale("log")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    # --- 右グラフ：相対誤差（train/val） ---
    ax_rel.plot(epochs, rel_tr_safe,  label="相対誤差（訓練データ）", linewidth=1.5)
    ax_rel.plot(epochs, rel_val_safe, label="相対誤差（テストデータ）", linewidth=1.5)

    ax_rel.set_xlabel("エポック数")
    ax_rel.set_ylabel("相対誤差")
    ax_rel.set_yscale("log")
    ax_rel.grid(True, alpha=0.3)
    ax_rel.legend()

    # 図全体のレイアウト調整
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.90])

    plt.pause(0.01)

# ------------------------------------------------------------
# メイン: train/val 分離版
# ------------------------------------------------------------

def train_gnn_auto_trainval_pde_weighted(
    data_dir: str,
    *,
    enable_plot: bool = True,
    return_history: bool = False,
    enable_error_plots: bool = False,  # ★ 追加：誤差場プロットを出すかどうか
):
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

    # --- (time, rank, gnn_dir) タプルリスト検出 & 分割 ---
    all_time_rank_tuples, missing_info = find_time_rank_list(data_dir)

    if not all_time_rank_tuples:
        # 見つからなかったファイルに応じてエラーメッセージを生成
        if missing_info.get("no_gnn_dirs"):
            raise RuntimeError(
                f"{data_dir}/processor*/gnn/ ディレクトリが見つかりませんでした。"
            )

        error_messages = []
        if missing_info.get("missing_csr"):
            # CSR ファイルが見つからなかった場合
            error_messages.append("A_csr_*.dat が見つかりませんでした。")
        if not missing_info.get("missing_csr"):
            # pEqn ファイルが見つからなかった場合（CSR はあるのに pEqn がない）
            error_messages.append("pEqn_*_rank*.dat が見つかりませんでした。")

        if error_messages:
            raise RuntimeError(
                f"{data_dir}/processor*/gnn/ 内に " + " ".join(error_messages)
            )
        else:
            raise RuntimeError(
                f"{data_dir}/processor*/gnn/ 内に有効なデータが見つかりませんでした。"
            )

    # x ファイルが見つからなかった場合は警告を表示（教師なし学習モードで続行）
    if missing_info.get("missing_x"):
        num_missing_x = len(missing_info["missing_x"])
        log_print(f"[WARN] x_*_rank*.dat が {num_missing_x} 件見つかりませんでした。教師なし学習モードで続行します。")

    # 検出されたランクの一覧をログ出力
    all_ranks = sorted(set(r for _, r, _ in all_time_rank_tuples), key=int)
    all_times_unique = sorted(set(t for t, _, _ in all_time_rank_tuples), key=float)
    all_gnn_dirs = sorted(set(g for _, _, g in all_time_rank_tuples))
    log_print(f"[INFO] 検出された rank 一覧: {all_ranks}")
    log_print(f"[INFO] 検出された time 一覧: {all_times_unique[:10]}{'...' if len(all_times_unique) > 10 else ''}")
    log_print(f"[INFO] 検出された gnn_dir 数: {len(all_gnn_dirs)}")

    # 以降の print(...) はすべて log_print(...) に置き換え
    random.seed(RANDOM_SEED)
    random.shuffle(all_time_rank_tuples)

    all_time_rank_tuples = all_time_rank_tuples[:MAX_NUM_CASES]
    n_total = len(all_time_rank_tuples)
    n_train = max(1, int(n_total * TRAIN_FRACTION))
    n_val   = n_total - n_train

    tuples_train = all_time_rank_tuples[:n_train]
    tuples_val   = all_time_rank_tuples[n_train:]

    log_print(f"[INFO] 検出された (time, rank) ペア数 (使用分) = {n_total}")
    log_print(f"[INFO] train: {n_train} cases, val: {n_val} cases (TRAIN_FRACTION={TRAIN_FRACTION})")
    log_print("=== 使用する train ケース (time, rank) ===")
    for t, r, g in tuples_train:
        log_print(f"  time={t}, rank={r}")
    log_print("=== 使用する val ケース (time, rank) ===")
    if tuples_val:
        for t, r, g in tuples_val:
            log_print(f"  time={t}, rank={r}")
    else:
        log_print("  (val ケースなし)")
    log_print("===========================================")


    # --- raw ケース読み込み（train + val 両方） ---
    # キャッシュが有効な場合はキャッシュから読み込み、そうでなければファイルから読み込んでキャッシュ
    raw_cases_all = []
    cache_path = _get_cache_path(data_dir, all_time_rank_tuples) if USE_DATA_CACHE else None

    if USE_DATA_CACHE and _is_cache_valid(cache_path, all_time_rank_tuples):
        # キャッシュから読み込み（高速）
        raw_cases_all = load_raw_cases_from_cache(cache_path)
    else:
        # ファイルから読み込み
        for t, r, g in all_time_rank_tuples:
            log_print(f"[LOAD] time={t}, rank={r} のグラフ+PDE情報を読み込み中...")
            rc = load_case_with_csr(g, t, r)
            raw_cases_all.append(rc)

        # キャッシュに保存
        if USE_DATA_CACHE:
            save_raw_cases_to_cache(raw_cases_all, cache_path)

    # train/val に分割
    raw_cases_train = []
    raw_cases_val   = []
    train_set = set(tuples_train)

    for rc in raw_cases_all:
        key = (rc["time"], rc["rank"], rc["gnn_dir"])
        if key in train_set:
            raw_cases_train.append(rc)
        else:
            raw_cases_val.append(rc)

    # 特徴量次元数の一貫性チェック（セル数は rank ごとに異なる可能性あり）
    nFeat = raw_cases_train[0]["feats_np"].shape[1]
    for rc in raw_cases_train + raw_cases_val:
        if rc["feats_np"].shape[1] != nFeat:
            raise RuntimeError("全ケースで nFeatures が一致していません。")

    total_cells = sum(rc["feats_np"].shape[0] for rc in raw_cases_train + raw_cases_val)
    log_print(f"[INFO] nFeatures = {nFeat}, 総セル数 (全ケース合計) = {total_cells}")

    # --- 教師なし学習モード判定 ---
    # 全ケースの has_x_true を確認
    cases_with_x = [rc for rc in (raw_cases_train + raw_cases_val) if rc.get("has_x_true", False)]
    unsupervised_mode = len(cases_with_x) == 0

    if unsupervised_mode:
        log_print("[INFO] *** 教師なし学習モード（PINNs）: x_*_rank*.dat が見つかりません ***")
        log_print("[INFO] *** 損失関数は PDE 損失のみを使用します ***")

    # --- グローバル正規化: train+val 全体で統計を取る ---
    all_feats = np.concatenate(
        [rc["feats_np"] for rc in (raw_cases_train + raw_cases_val)], axis=0
    )

    feat_mean = all_feats.mean(axis=0, keepdims=True)
    feat_std  = all_feats.std(axis=0, keepdims=True) + 1e-12

    # x_true の統計（教師あり学習の場合のみ）
    if not unsupervised_mode:
        all_xtrue = np.concatenate(
            [rc["x_true_np"] for rc in cases_with_x], axis=0
        )
        x_mean = all_xtrue.mean()
        x_std  = all_xtrue.std() + 1e-12
        log_print(
            f"[INFO] x_true (cases with ground truth): "
            f"min={all_xtrue.min():.3e}, max={all_xtrue.max():.3e}, mean={x_mean:.3e}"
        )
    else:
        # 教師なし学習の場合、b と対角成分から出力スケールを推定
        # 次元解析: Ax = b より、x のスケールは ||b|| / ||diag|| 程度
        all_b = np.concatenate([rc["b_np"] for rc in raw_cases_train], axis=0)
        all_diag = np.concatenate([rc["feats_np"][:, 3] for rc in raw_cases_train], axis=0)

        b_rms = np.sqrt(np.mean(all_b**2)) + 1e-12
        diag_rms = np.sqrt(np.mean(all_diag**2)) + 1e-12

        # x のスケールを推定（ゲージ正則化で平均は 0 に近づくと仮定）
        x_mean = 0.0
        x_std = b_rms / diag_rms
        log_print(
            f"[INFO] x_true 統計: 教師なし学習モード（b と diag から推定）"
            f" mean={x_mean:.3e}, std={x_std:.3e} (b_rms={b_rms:.3e}, diag_rms={diag_rms:.3e})"
        )

    x_mean_t = torch.tensor(x_mean, dtype=torch.float32, device=device)
    x_std_t  = torch.tensor(x_std,  dtype=torch.float32, device=device)

    # --- rank ごとの x_true 統計（train ケースのみ） ---
    #     data loss を rank ごとに正規化するための mean/std
    train_ranks = sorted({int(rc["rank"]) for rc in raw_cases_train})
    num_ranks = max(train_ranks) + 1

    sums   = np.zeros(num_ranks, dtype=np.float64)
    sqsums = np.zeros(num_ranks, dtype=np.float64)
    counts = np.zeros(num_ranks, dtype=np.int64)

    # 教師あり学習の場合のみ rank ごと統計を計算
    if not unsupervised_mode:
        for rc in raw_cases_train:
            if not rc.get("has_x_true", False):
                continue
            r = int(rc["rank"])
            x = rc["x_true_np"].astype(np.float64).reshape(-1)
            sums[r]   += x.sum()
            sqsums[r] += np.square(x).sum()
            counts[r] += x.size

    # 初期値としてグローバル mean/std を入れておき、train に存在する rank だけ上書き
    x_mean_rank = np.full(num_ranks, x_mean, dtype=np.float64)
    x_std_rank  = np.full(num_ranks, x_std,  dtype=np.float64)

    for r in range(num_ranks):
        if counts[r] > 0:
            mean_r = sums[r] / counts[r]
            var_r  = sqsums[r] / counts[r] - mean_r * mean_r
            std_r  = np.sqrt(max(var_r, 1e-24))
            x_mean_rank[r] = mean_r
            x_std_rank[r]  = std_r

    log_print("[INFO] rank-wise x_true statistics (train only):")
    for r in range(num_ranks):
        log_print(
            f"  rank={r}: count={counts[r]}, "
            f"mean={x_mean_rank[r]:.3e}, std={x_std_rank[r]:.3e}"
        )

    # torch.Tensor (device 上) として保持
    x_mean_rank_t = torch.from_numpy(x_mean_rank.astype(np.float32)).to(device)
    x_std_rank_t  = torch.from_numpy(x_std_rank.astype(np.float32)).to(device)

    # --- torch ケース化 & w_pde 統計 ---
    # USE_LAZY_LOADING が True の場合、データは CPU に保持され、学習時に GPU へ転送される
    cases_train = []
    cases_val   = []
    w_all_list  = []

    if USE_LAZY_LOADING:
        log_print("[INFO] 遅延GPU転送モード: データはCPUに保持され、使用時のみGPUへ転送されます")

    for rc in raw_cases_train:
        cs = convert_raw_case_to_torch_case(
            rc, feat_mean, feat_std, x_mean, x_std, device,
            lazy_load=USE_LAZY_LOADING
        )
        cases_train.append(cs)
        w_all_list.append(cs["w_pde_np"].reshape(-1))

    for rc in raw_cases_val:
        cs = convert_raw_case_to_torch_case(
            rc, feat_mean, feat_std, x_mean, x_std, device,
            lazy_load=USE_LAZY_LOADING
        )
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

    # --- 条件数の推定（最初のケースで計算） ---
    if raw_cases_train:
        rc0 = raw_cases_train[0]
        diag_np = rc0["feats_np"][:, 3]  # 対角成分

        log_print("=== Condition number estimation (first training case) ===")

        # スケーリング前の条件数
        cond_before = estimate_condition_number(
            rc0["row_ptr_np"], rc0["col_ind_np"], rc0["vals_np"], diag_np
        )
        log_print(f"  [Before scaling] λ_max = {cond_before['lambda_max']:.6e}, "
                  f"λ_min = {cond_before['lambda_min']:.6e}, "
                  f"κ(A) = {cond_before['condition_number']:.6e}")

        # スケーリング後の条件数
        if USE_DIAGONAL_SCALING:
            vals_scaled, _, _ = apply_diagonal_scaling_csr(
                rc0["row_ptr_np"], rc0["col_ind_np"], rc0["vals_np"], diag_np, rc0["b_np"]
            )
            # スケーリング後は対角成分が1になる
            diag_scaled = np.ones_like(diag_np)
            cond_after = estimate_condition_number(
                rc0["row_ptr_np"], rc0["col_ind_np"], vals_scaled, diag_scaled
            )
            log_print(f"  [After scaling]  λ_max = {cond_after['lambda_max']:.6e}, "
                      f"λ_min = {cond_after['lambda_min']:.6e}, "
                      f"κ(Ã) = {cond_after['condition_number']:.6e}")
            improvement = cond_before['condition_number'] / (cond_after['condition_number'] + 1e-12)
            log_print(f"  Condition number improvement: {improvement:.2f}x")
        else:
            log_print("  [Diagonal scaling disabled]")

        log_print("==========================================================================")

    num_train = len(cases_train)
    num_val   = len(cases_val)
    num_train_with_x = sum(1 for cs in cases_train if cs.get("has_x_true", False))

    # --- モデル定義 ---
    model = SimpleSAGE(
        in_channels=nFeat,
        hidden_channels=HIDDEN_CHANNELS,
        num_layers=NUM_LAYERS,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = None
    if USE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=LR_SCHED_FACTOR,
            patience=LR_SCHED_PATIENCE,
            min_lr=LR_SCHED_MIN_LR,
            verbose=False,
        )

    # --- AMP (混合精度学習) の設定 ---
    use_amp_actual = USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp_actual)
    if use_amp_actual:
        log_print("[INFO] 混合精度学習 (AMP) が有効です")
    else:
        if USE_AMP and device.type != "cuda":
            log_print("[INFO] AMP は CUDA デバイスでのみ有効です。CPU モードでは無効化されます")

    # 学習モードの表示と検証
    if LAMBDA_DATA == 0 and LAMBDA_PDE == 0:
        raise ValueError(
            "LAMBDA_DATA と LAMBDA_PDE が両方 0 です。"
            "少なくとも一方は正の値を設定してください。"
        )

    if LAMBDA_DATA == 0:
        learning_mode = "完全な教師なし学習 (PDE損失のみ)"
    elif LAMBDA_PDE == 0:
        learning_mode = "完全な教師あり学習 (データ損失のみ)"
    else:
        learning_mode = "ハイブリッド学習 (データ損失 + PDE損失)"

    log_print(f"=== Training start: {learning_mode} ===")
    log_print(f"    LAMBDA_DATA={LAMBDA_DATA}, LAMBDA_PDE={LAMBDA_PDE}, LAMBDA_GAUGE={LAMBDA_GAUGE}")

    # --- 可視化用の準備 ---
    fig, axes = (None, None)
    if enable_plot:
        fig, axes = init_plot()
    history = {
        "epoch": [],
        "loss": [],
        "data_loss": [],
        "pde_loss": [],
        "gauge_loss": [],  # ゲージ損失（教師なし学習時のみ）
        "rel_err_train": [],
        "rel_err_val": [],  # val が無いときは None
    }

    # --- 学習ループ ---
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # 学習率ウォームアップ
        if USE_LR_WARMUP and epoch <= LR_WARMUP_EPOCHS:
            warmup_factor = epoch / LR_WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR * warmup_factor

        total_data_loss = 0.0
        total_pde_loss  = 0.0
        total_gauge_loss = 0.0
        sum_rel_err_tr  = 0.0
        sum_R_pred_tr   = 0.0
        sum_rmse_tr     = 0.0
        num_cases_with_x = 0  # データ損失を計算したケース数

        # -------- train で勾配計算 --------
        for cs in cases_train:
            # 遅延ロードの場合、ケースデータを GPU に転送
            if USE_LAZY_LOADING:
                cs_gpu = move_case_to_device(cs, device)
            else:
                cs_gpu = cs

            feats       = cs_gpu["feats"]
            edge_index  = cs_gpu["edge_index"]
            x_true      = cs_gpu["x_true"]  # 教師なし学習の場合は None
            b           = cs_gpu["b"]
            row_ptr     = cs_gpu["row_ptr"]
            col_ind     = cs_gpu["col_ind"]
            vals        = cs_gpu["vals"]
            row_idx     = cs_gpu["row_idx"]
            w_pde       = cs_gpu["w_pde"]
            has_x_true  = cs_gpu.get("has_x_true", x_true is not None)
            diag_sqrt   = cs_gpu.get("diag_sqrt", None)
            use_dscale  = cs_gpu.get("use_diagonal_scaling", False) and (diag_sqrt is not None)
            volume      = cs_gpu["volume"]  # セル体積（ゲージ正則化用）
            diag        = cs_gpu["diag"]    # 対角成分（行ごと正規化用）


            # AMP: autocast で順伝播と損失計算を FP16/BF16 で実行
            with torch.amp.autocast(device_type='cuda', enabled=use_amp_actual):
                # モデルは正規化スケールで出力
                x_pred_norm = model(feats, edge_index)
                # 非正規化スケールに戻す
                x_pred = x_pred_norm * x_std_t + x_mean_t

                # データ損失: x_true がある場合かつ LAMBDA_DATA > 0 のときのみ計算
                if has_x_true and x_true is not None and LAMBDA_DATA > 0:
                    # rank ごとの mean/std を用いた x の正規化（data loss 用）
                    rank_id = int(cs["rank"])
                    mean_r  = x_mean_rank_t[rank_id]
                    std_r   = x_std_rank_t[rank_id]

                    # x_true, x_pred を rank ごとに標準化
                    x_true_norm_case = (x_true - mean_r) / (std_r + 1e-12)
                    x_pred_norm_case_for_loss = (x_pred - mean_r) / (std_r + 1e-12)

                    # データ損失: rank ごとに正規化した MSE
                    data_loss_case = F.mse_loss(
                        x_pred_norm_case_for_loss,
                        x_true_norm_case
                    )
                    num_cases_with_x += 1
                else:
                    # 教師なし学習 または LAMBDA_DATA = 0: データ損失は計算しない
                    data_loss_case = None

                # PDE 損失: LAMBDA_PDE > 0 の場合のみ計算
                if LAMBDA_PDE > 0:
                    # 対角スケーリング有効時は、A_scaled x_scaled = b_scaled を評価
                    x_for_pde = (x_pred * diag_sqrt) if use_dscale else x_pred
                    Ax = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_for_pde)
                    r  = Ax - b

                    # PDE損失の正規化
                    sqrt_w = torch.sqrt(w_pde)
                    wr = sqrt_w * r
                    wb = sqrt_w * b

                    if PDE_LOSS_NORMALIZATION == "relative":
                        # 相対残差ノルム: ||w*r||² / ||w*b||²
                        # 物理的に意味があり、||r||/||b|| ≈ 1 のとき pde_loss ≈ 1
                        norm_wr_sq = torch.sum(wr * wr)
                        norm_wb_sq = torch.sum(wb * wb) + EPS_RES
                        pde_loss_case = norm_wr_sq / norm_wb_sq
                    elif PDE_LOSS_NORMALIZATION == "row_diag":
                        # 行ごと正規化（対角成分でスケール）- 値が極小になる問題あり
                        diag_abs = torch.abs(diag) + EPS_RES
                        r_normalized = r / diag_abs
                        wr_norm = sqrt_w * r_normalized
                        pde_loss_case = torch.mean(wr_norm * wr_norm)
                    else:
                        # "none": ||r||² / (||Ax||² + ||b||² + eps)
                        wAx = sqrt_w * Ax
                        norm_wr = torch.norm(wr)
                        norm_scale = torch.sqrt(torch.norm(wAx)**2 + torch.norm(wb)**2) + EPS_RES
                        pde_loss_case = (norm_wr / norm_scale) ** 2

                    # 診断用の相対残差（ログ出力用）
                    with torch.no_grad():
                        norm_wr_diag = torch.norm(sqrt_w * r)
                        norm_wb_diag = torch.norm(sqrt_w * b) + EPS_RES
                        R_pred = norm_wr_diag / norm_wb_diag
                else:
                    # LAMBDA_PDE = 0: PDE損失は計算しない（完全な教師あり学習）
                    pde_loss_case = None
                    R_pred = torch.tensor(0.0, device=device)

                # ゲージ損失: セル体積で重み付けした平均値の二乗（物理的に意味のある平均）
                # 圧力ポアソン方程式の解は定数の不定性（ゲージ自由度）があるため、
                # 体積加重平均をゼロに近づけることで解を一意に定める
                total_volume = torch.sum(volume) + EPS_RES
                weighted_mean = torch.sum(x_pred * volume) / total_volume
                gauge_loss_case = weighted_mean ** 2

            # ---- 目的関数（平均化を保ったまま、ケースごとに backward して勾配蓄積）----
            # 各損失項を条件に応じて加算
            loss_case = torch.tensor(0.0, device=device, requires_grad=True)

            # PDE損失（LAMBDA_PDE > 0 の場合のみ）
            if pde_loss_case is not None:
                loss_case = loss_case + (LAMBDA_PDE / num_train) * pde_loss_case

            # ゲージ損失（常に加算、ただしLAMBDA_GAUGE > 0 の場合のみ実効的）
            if LAMBDA_GAUGE > 0:
                loss_case = loss_case + (LAMBDA_GAUGE / num_train) * gauge_loss_case

            # データ損失（LAMBDA_DATA > 0 かつ x_true がある場合のみ）
            if data_loss_case is not None:
                loss_case = loss_case + (LAMBDA_DATA / num_train_with_x) * data_loss_case

            # 少なくとも1つの損失がある場合のみ backward
            if loss_case.requires_grad:
                scaler.scale(loss_case).backward()

            # logging用（グラフを持たない形で集計）
            if pde_loss_case is not None:
                total_pde_loss += float(pde_loss_case.detach().cpu())
            total_gauge_loss += float(gauge_loss_case.detach().cpu())
            if data_loss_case is not None:
                total_data_loss += float(data_loss_case.detach().cpu())

            with torch.no_grad():
                # rel_err, RMSE: x_true がある場合のみ計算
                if has_x_true and x_true is not None:
                    # ゲージ不変評価: 両者を平均ゼロに正規化してから比較
                    # 圧力ポアソン方程式の解は定数の不定性があるため、
                    # 公平な比較のために平均を引いてから誤差を計算
                    x_pred_centered = x_pred - torch.mean(x_pred)
                    x_true_centered = x_true - torch.mean(x_true)
                    diff = x_pred_centered - x_true_centered
                    N = x_true.shape[0]
                    rel_err_case = torch.norm(diff) / (torch.norm(x_true_centered) + EPS_DATA)
                    rmse_case    = torch.sqrt(torch.sum(diff * diff) / N)
                    sum_rel_err_tr += rel_err_case.item()
                    sum_rmse_tr    += rmse_case.item()
                sum_R_pred_tr  += R_pred.detach().item()

            # 遅延ロードの場合、参照を外す（empty_cache は通常不要・逆に遅くなる）
            if USE_LAZY_LOADING:
                del cs_gpu

        # 勾配クリッピング
        if USE_GRAD_CLIP:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)

        scaler.step(optimizer)
        scaler.update()

        # epoch平均（ログ・history用）
        avg_pde_loss = total_pde_loss / max(1, num_train) if LAMBDA_PDE > 0 else 0.0
        avg_gauge_loss = total_gauge_loss / max(1, num_train) if LAMBDA_GAUGE > 0 else 0.0

        # 損失値の計算（各項はlambda > 0の場合のみ加算）
        loss_value = 0.0
        if LAMBDA_PDE > 0:
            loss_value += LAMBDA_PDE * avg_pde_loss
        if LAMBDA_GAUGE > 0:
            loss_value += LAMBDA_GAUGE * avg_gauge_loss

        # データ損失
        if unsupervised_mode or num_cases_with_x == 0 or LAMBDA_DATA == 0:
            avg_data_loss = 0.0
        else:
            avg_data_loss = total_data_loss / max(1, num_cases_with_x)
            loss_value += LAMBDA_DATA * avg_data_loss
        avg_rel_err_val = None
        avg_R_pred_val = None
        avg_rmse_val = None

        # スケジューラ用・ロギング用に検証誤差を計算（必要なときのみ）
        need_val_eval = num_val > 0 and (scheduler is not None or epoch % PLOT_INTERVAL == 0)
        if need_val_eval:
            model.eval()
            avg_rel_err_val, avg_rmse_val, avg_R_pred_val, _ = evaluate_validation_cases(
                model, cases_val, device, x_std_t, x_mean_t, use_amp_actual
            )

        # 学習率スケジューラを更新（検証誤差があればそれを監視）
        if scheduler is not None:
            metric_for_scheduler = avg_rel_err_val if avg_rel_err_val is not None else float(loss_value)
            scheduler.step(metric_for_scheduler)


        # --- ロギング（train + val） ---
        if epoch % PLOT_INTERVAL == 0 or epoch == 1:
            # 教師あり学習の場合のみ相対誤差を計算
            if unsupervised_mode or num_cases_with_x == 0:
                avg_rel_err_tr = sum_R_pred_tr / num_train  # PDE 残差を代用
                avg_rmse_tr    = 0.0
            else:
                avg_rel_err_tr = sum_rel_err_tr / num_cases_with_x
                avg_rmse_tr    = sum_rmse_tr / num_cases_with_x
            avg_R_pred_tr  = sum_R_pred_tr / num_train

            current_lr = optimizer.param_groups[0]["lr"]

            if num_val > 0 and avg_rel_err_val is None:
                # まだ val を計算していない場合のみ算出
                model.eval()
                avg_rel_err_val, avg_rmse_val, avg_R_pred_val, _ = evaluate_validation_cases(
                    model, cases_val, device, x_std_t, x_mean_t, use_amp_actual
                )

            # 履歴に追加
            history["epoch"].append(epoch)
            history["loss"].append(float(loss_value))
            history["data_loss"].append(float(LAMBDA_DATA * avg_data_loss))
            history["pde_loss"].append(float(LAMBDA_PDE * avg_pde_loss))
            history["gauge_loss"].append(float(LAMBDA_GAUGE * avg_gauge_loss))
            history["rel_err_train"].append(float(avg_rel_err_tr))
            history["rel_err_val"].append(None if avg_rel_err_val is None else float(avg_rel_err_val))

            # プロット更新
            if enable_plot:
                update_plot(fig, axes, history)

            # コンソールログ
            log = (
                f"[Epoch {epoch:5d}] loss={loss_value:.4e}, "
                f"lr={current_lr:.3e}, "
                f"data_loss={LAMBDA_DATA * avg_data_loss:.4e}, "
                f"PDE_loss={LAMBDA_PDE * avg_pde_loss:.4e}, "
            )
            if unsupervised_mode or num_cases_with_x == 0:
                # 教師なし学習: ゲージ損失も表示
                log += f"gauge_loss={LAMBDA_GAUGE * avg_gauge_loss:.4e}, "
            log += (
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
    if enable_plot and len(history["epoch"]) > 0:
        final_plot_filename = (
            f"training_history_"
            f"DATA{lambda_data_tag}_"
            f"PDE{lambda_pde_tag}.png"
        )
        final_plot_path = os.path.join(OUTPUT_DIR, final_plot_filename)

        update_plot(fig, axes, history)
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

    # ★ ここでカウンタを初期化（関数のこのスコープ内）
    num_error_plots_train = 0

    for cs in cases_train:
        time_str   = cs["time"]
        rank_str   = cs["rank"]

        # 遅延ロードの場合、ケースデータを GPU に転送
        if USE_LAZY_LOADING:
            cs_gpu = move_case_to_device(cs, device)
        else:
            cs_gpu = cs

        feats      = cs_gpu["feats"]
        edge_index = cs_gpu["edge_index"]
        x_true     = cs_gpu["x_true"]
        b          = cs_gpu["b"]
        row_ptr    = cs_gpu["row_ptr"]
        col_ind    = cs_gpu["col_ind"]
        vals       = cs_gpu["vals"]
        row_idx    = cs_gpu["row_idx"]
        w_pde      = cs_gpu["w_pde"]
        has_x_true = cs_gpu.get("has_x_true", x_true is not None)
        diag_sqrt  = cs_gpu.get("diag_sqrt", None)
        use_dscale = cs_gpu.get("use_diagonal_scaling", False) and (diag_sqrt is not None)

        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=use_amp_actual):
                x_pred_norm = model(feats, edge_index)
                x_pred = x_pred_norm * x_std_t + x_mean_t

            # 学習で使った weighted PDE 残差
            x_for_pde = (x_pred * diag_sqrt) if use_dscale else x_pred
            Ax_pred_w = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_for_pde)
            r_pred_w  = Ax_pred_w - b
            sqrt_w    = torch.sqrt(w_pde)
            wr_pred   = sqrt_w * r_pred_w
            wb        = sqrt_w * b
            norm_wr   = torch.norm(wr_pred)
            norm_wb   = torch.norm(wb) + EPS_RES
            R_pred_w  = norm_wr / norm_wb

            # 物理的な（非加重）PDE 残差（対角スケール時は r_phys = D^(1/2) r_scaled に戻す）
            Ax_pred = Ax_pred_w
            r_scaled = Ax_pred - b
            r_pred  = (diag_sqrt * r_scaled) if use_dscale else r_scaled
            norm_r_pred    = torch.norm(r_pred)
            max_abs_r_pred = torch.max(torch.abs(r_pred))
            b_phys         = (diag_sqrt * b) if use_dscale else b
            norm_b         = torch.norm(b_phys)
            norm_Ax_pred   = torch.norm((b_phys + r_pred))
            R_pred_over_b  = norm_r_pred / (norm_b + EPS_RES)
            R_pred_over_Ax = norm_r_pred / (norm_Ax_pred + EPS_RES)

            # 教師あり学習の場合のみ x_true との比較
            if has_x_true and x_true is not None:
                diff = x_pred - x_true
                N = x_true.shape[0]
                rel_err = torch.norm(diff) / (torch.norm(x_true) + EPS_DATA)
                rmse    = torch.sqrt(torch.sum(diff * diff) / N)

                # 物理的な（非加重）PDE 残差: OpenFOAM 解
                x_true_for_pde = (x_true * diag_sqrt) if use_dscale else x_true
                Ax_true = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_true_for_pde)
                r_scaled_true = Ax_true - b
                r_true  = (diag_sqrt * r_scaled_true) if use_dscale else r_scaled_true
                norm_r_true    = torch.norm(r_true)
                max_abs_r_true = torch.max(torch.abs(r_true))
                norm_Ax_true   = torch.norm((b_phys + r_true))
                R_true_over_b  = norm_r_true / (norm_b + EPS_RES)
                R_true_over_Ax = norm_r_true / (norm_Ax_true + EPS_RES)

        if has_x_true and x_true is not None:
            log_print(
                f"  [train] Case (time={time_str}, rank={rank_str}): "
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

            # --- ここでスケール診断 ---
            a, b_fit, rmse_before, rmse_after = compute_affine_fit(x_true, x_pred)
            log_print(
                f"    [Affine fit x_pred->x_true] "
                f"a={a:.3e}, b={b_fit:.3e}, "
                f"RMSE_before={rmse_before:.3e}, RMSE_after={rmse_after:.3e}, "
                f"RMSE_ratio={rmse_after / rmse_before:.3f}"
            )
        else:
            # 教師なし学習: PDE 残差のみ表示
            log_print(
                f"  [train] Case (time={time_str}, rank={rank_str}) [教師なし学習]: "
                f"R_pred(weighted) = {R_pred_w.item():.4e}"
            )
            log_print(f"    x_pred: min={x_pred.min().item():.6e}, max={x_pred.max().item():.6e}, "
                  f"mean={x_pred.mean().item():.6e}, norm={torch.norm(x_pred).item():.6e}")
            log_print(
                "    [PDE residual (GNN)]"
                f" ||r||_2={norm_r_pred.item():.6e}, "
                f"max|r_i|={max_abs_r_pred.item():.6e}, "
                f"||r||/||b||={R_pred_over_b.item():.5f}, "
                f"||r||/||Ax||={R_pred_over_Ax.item():.5f}"
            )

        # 予測結果の書き出し
        x_pred_np = x_pred.cpu().numpy().reshape(-1)
        out_path = os.path.join(OUTPUT_DIR, f"x_pred_train_{time_str}_rank{rank_str}.dat")
        with open(out_path, "w") as f:
            for i, val in enumerate(x_pred_np):
                f.write(f"{i} {val:.9e}\n")
        log_print(f"    [INFO] train x_pred を {out_path} に書き出しました。")

        # ★ 誤差場の可視化（train ケース、x_true がある場合のみ）
        if enable_error_plots and has_x_true and x_true is not None and num_error_plots_train < MAX_ERROR_PLOT_CASES_TRAIN:
            prefix = f"train_time{time_str}_rank{rank_str}"
            save_error_field_plots(cs, x_pred, x_true, prefix)
            save_pressure_comparison_plots(cs, x_pred, x_true, prefix)
            num_error_plots_train += 1

        # 遅延ロードの場合、GPU メモリを解放
        if USE_LAZY_LOADING:
            del cs_gpu
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if num_val > 0:
        log_print("\n=== Final diagnostics (val cases) ===")

        # ★ val 側のカウンタもここで初期化
        num_error_plots_val = 0

        for cs in cases_val:
            time_str   = cs["time"]
            rank_str   = cs["rank"]

            # 遅延ロードの場合、ケースデータを GPU に転送
            if USE_LAZY_LOADING:
                cs_gpu = move_case_to_device(cs, device)
            else:
                cs_gpu = cs

            feats      = cs_gpu["feats"]
            edge_index = cs_gpu["edge_index"]
            x_true     = cs_gpu["x_true"]
            b          = cs_gpu["b"]
            row_ptr    = cs_gpu["row_ptr"]
            col_ind    = cs_gpu["col_ind"]
            vals       = cs_gpu["vals"]
            row_idx    = cs_gpu["row_idx"]
            w_pde      = cs_gpu["w_pde"]
            has_x_true = cs_gpu.get("has_x_true", x_true is not None)
            diag_sqrt  = cs_gpu.get("diag_sqrt", None)
            use_dscale = cs_gpu.get("use_diagonal_scaling", False) and (diag_sqrt is not None)

            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', enabled=use_amp_actual):
                    x_pred_norm = model(feats, edge_index)
                    x_pred = x_pred_norm * x_std_t + x_mean_t

                # 学習で使った weighted PDE 残差（対角スケーリング適用）
                x_for_pde = (x_pred * diag_sqrt) if use_dscale else x_pred
                Ax_pred_w = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_for_pde)
                r_pred_w  = Ax_pred_w - b
                sqrt_w    = torch.sqrt(w_pde)
                wr_pred   = sqrt_w * r_pred_w
                wb        = sqrt_w * b
                norm_wr   = torch.norm(wr_pred)
                norm_wb   = torch.norm(wb) + EPS_RES
                R_pred_w  = norm_wr / norm_wb

                # 物理的な（非加重）PDE 残差（対角スケール時は r_phys = D^(1/2) r_scaled に戻す）
                Ax_pred = Ax_pred_w
                r_scaled = Ax_pred - b
                r_pred  = (diag_sqrt * r_scaled) if use_dscale else r_scaled
                norm_r_pred    = torch.norm(r_pred)
                max_abs_r_pred = torch.max(torch.abs(r_pred))
                b_phys         = (diag_sqrt * b) if use_dscale else b
                norm_b         = torch.norm(b_phys)
                norm_Ax_pred   = torch.norm((b_phys + r_pred))
                R_pred_over_b  = norm_r_pred / (norm_b + EPS_RES)
                R_pred_over_Ax = norm_r_pred / (norm_Ax_pred + EPS_RES)

                # 教師あり学習の場合のみ x_true との比較
                if has_x_true and x_true is not None:
                    diff = x_pred - x_true
                    N = x_true.shape[0]
                    rel_err = torch.norm(diff) / (torch.norm(x_true) + EPS_DATA)
                    rmse    = torch.sqrt(torch.sum(diff * diff) / N)

                    # 物理的な（非加重）PDE 残差: OpenFOAM 解
                    x_true_for_pde = (x_true * diag_sqrt) if use_dscale else x_true
                    Ax_true = matvec_csr_torch(row_ptr, col_ind, vals, row_idx, x_true_for_pde)
                    r_scaled_true = Ax_true - b
                    r_true  = (diag_sqrt * r_scaled_true) if use_dscale else r_scaled_true
                    norm_r_true    = torch.norm(r_true)
                    max_abs_r_true = torch.max(torch.abs(r_true))
                    norm_Ax_true   = torch.norm((b_phys + r_true))
                    R_true_over_b  = norm_r_true / (norm_b + EPS_RES)
                    R_true_over_Ax = norm_r_true / (norm_Ax_true + EPS_RES)

            if has_x_true and x_true is not None:
                log_print(
                    f"  [val]   Case (time={time_str}, rank={rank_str}): "
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

                # --- ここでスケール診断 ---
                a, b_fit, rmse_before, rmse_after = compute_affine_fit(x_true, x_pred)
                log_print(
                    f"    [Affine fit x_pred->x_true] "
                    f"a={a:.3e}, b={b_fit:.3e}, "
                    f"RMSE_before={rmse_before:.3e}, RMSE_after={rmse_after:.3e}, "
                    f"RMSE_ratio={rmse_after / rmse_before:.3f}"
                )
            else:
                # 教師なし学習: PDE 残差のみ表示
                log_print(
                    f"  [val]   Case (time={time_str}, rank={rank_str}) [教師なし学習]: "
                    f"R_pred(weighted) = {R_pred_w.item():.4e}"
                )
                log_print(f"    x_pred: min={x_pred.min().item():.6e}, max={x_pred.max().item():.6e}, "
                      f"mean={x_pred.mean().item():.6e}, norm={torch.norm(x_pred).item():.6e}")
                log_print(
                    "    [PDE residual (GNN)]"
                    f" ||r||_2={norm_r_pred.item():.6e}, "
                    f"max|r_i|={max_abs_r_pred.item():.6e}, "
                    f"||r||/||b||={R_pred_over_b.item():.5f}, "
                    f"||r||/||Ax||={R_pred_over_Ax.item():.5f}"
                )

            x_pred_np = x_pred.cpu().numpy().reshape(-1)
            out_path = os.path.join(OUTPUT_DIR, f"x_pred_val_{time_str}_rank{rank_str}.dat")
            with open(out_path, "w") as f:
                for i, val in enumerate(x_pred_np):
                    f.write(f"{i} {val:.9e}\n")
            log_print(f"    [INFO] val x_pred を {out_path} に書き出しました。")

            # ★ 誤差場の可視化（val ケース、x_true がある場合のみ）
            if enable_error_plots and has_x_true and x_true is not None and num_error_plots_val < MAX_ERROR_PLOT_CASES_VAL:
                prefix = f"val_time{time_str}_rank{rank_str}"
                save_error_field_plots(cs, x_pred, x_true, prefix)
                save_pressure_comparison_plots(cs, x_pred, x_true, prefix)
                num_error_plots_val += 1

            # 遅延ロードの場合、GPU メモリを解放
            if USE_LAZY_LOADING:
                del cs_gpu
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if return_history:
        return history

if __name__ == "__main__":
    train_gnn_auto_trainval_pde_weighted(DATA_DIR)

