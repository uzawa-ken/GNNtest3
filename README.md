# GNN PDE ソルバー

OpenFOAM の数値解析結果を用いてグラフニューラルネットワーク（GNN）を訓練し、圧力場を予測するシステムです。メッシュ品質に基づく加重 PDE 損失を採用しています。

## 概要

このプロジェクトは、CFD（数値流体力学）シミュレーションの高速代替モデルの構築を目的としています。OpenFOAM で生成されたメッシュ情報と解析結果から、GNN を用いて圧力場の予測を行います。

### 主な特徴

- **複数プロセス対応**: MPI 並列計算で生成された複数 rank のデータを統合して学習
- **自動データ検出**: 指定ディレクトリから全ての (time, rank) ペアを自動的に検出
- **物理ベース学習**: PDE 残差を損失関数に組み込むことで物理的整合性を確保
- **メッシュ品質適応**: スキュー、非直交性、アスペクト比などのメッシュ品質指標に基づいて学習重みを動的に調整
- **リアルタイム可視化**: 訓練過程をリアルタイムでグラフ表示

## 必要条件

### 依存パッケージ

```bash
pip install torch
pip install torch-geometric
pip install numpy
pip install matplotlib
```

### 日本語フォント（オプション）

グラフの日本語表示のため、以下のいずれかのフォントをインストールしてください：
- IPAexGothic
- Noto Sans CJK JP

## ディレクトリ構造

```
GNNtest3/
├── README.md                          # このファイル
├── GNN_train_val_weight.py            # メインスクリプト
└── data/                              # データディレクトリ（要作成）
    ├── pEqn_{time}_rank0.dat          # rank 0 の PDE 方程式情報
    ├── pEqn_{time}_rank1.dat          # rank 1 の PDE 方程式情報
    ├── ...                            # 他の rank のファイル
    ├── x_{time}_rank0.dat             # rank 0 の OpenFOAM 解
    ├── x_{time}_rank1.dat             # rank 1 の OpenFOAM 解
    ├── ...
    ├── A_csr_{time}_rank0.dat         # rank 0 の係数行列（CSR 形式）
    ├── A_csr_{time}_rank1.dat         # rank 1 の係数行列（CSR 形式）
    └── ...
```

### 複数プロセス対応

本プログラムは MPI 並列計算で領域分割された OpenFOAM の結果に対応しています：

- 各 rank（プロセス）のデータは独立したグラフとして扱われます
- 全ての rank のデータを自動検出し、統合して学習を行います
- 各 rank は異なるセル数を持つことができます
- (time, rank) ペアを単位として train/val 分割を行います

## データ形式

### pEqn ファイル（`pEqn_{time}_rank{N}.dat`）

メッシュのセル情報、エッジ情報、境界面情報を含むファイルです。

```
nCells {セル数}
nFaces {面数}
CELLS
{cell_id} {x} {y} {z} {diag} {b} {skew} {non_ortho} {aspect_ratio} {size_jump} {volume} {owner_nb} {neighbour_nb}
...
EDGES
{edge_id} {lower_cell} {upper_cell} {weight1} {weight2}
...
WALL_FACES
{face_id} {cell_id} {diag_contrib} {b_contrib}
...
```

### x ファイル（`x_{time}_rank{N}.dat`）

OpenFOAM で計算された圧力値（正解データ）です。

```
{cell_id} {pressure_value}
{cell_id} {pressure_value}
...
```

### CSR 行列ファイル（`A_csr_{time}_rank{N}.dat`）

PDE の係数行列を CSR（Compressed Sparse Row）形式で格納したファイルです。各 rank ごとに独立したファイルが必要です。

```
nRows {行数}
nCols {列数}
nnz {非ゼロ要素数}
ROW_PTR
{row_ptr の値...}
COL_IND
{col_ind の値...}
VALUES
{values の値...}
```

## 使用方法

### 1. データの準備

OpenFOAM の解析結果から上記形式のデータファイルを生成し、`./data` ディレクトリに配置してください。全ての rank のファイルを同一ディレクトリに配置します。

### 2. 設定の調整

`GNN_train_val_weight.py` 内の設定パラメータを必要に応じて変更してください：

```python
# データディレクトリ
DATA_DIR       = "./data"
OUTPUT_DIR     = "./"

# 学習パラメータ
NUM_EPOCHS     = 1000          # エポック数
LR             = 1e-3          # 学習率
WEIGHT_DECAY   = 1e-5          # L2 正則化
MAX_NUM_CASES  = 100           # 使用する (time, rank) ペア数の上限
TRAIN_FRACTION = 0.8           # 訓練データの割合

# 損失関数の重み
LAMBDA_DATA = 0.1              # データ損失の重み
LAMBDA_PDE  = 0.0001           # PDE 損失の重み
```

### 3. 実行

```bash
python GNN_train_val_weight.py
```

実行時に以下の情報がログ出力されます：
- 検出された rank 一覧
- 検出された time 一覧
- 使用する (time, rank) ペアの総数
- train/val 分割の詳細

## モデルアーキテクチャ

GraphSAGE ベースの 4 層ニューラルネットワークを使用しています。

| 層 | 入力次元 | 出力次元 | 活性化関数 |
|---|---------|---------|----------|
| conv1 | 13 | 64 | ReLU |
| conv2 | 64 | 64 | ReLU |
| conv3 | 64 | 64 | ReLU |
| conv4 | 64 | 1 | なし |

### 入力特徴量（13 次元）

1. x 座標
2. y 座標
3. z 座標
4. 対角成分
5. RHS 値（b）
6. スキュー
7. 非直交性
8. アスペクト比
9. サイズジャンプ
10. 体積
11. owner 隣接数
12. neighbour 隣接数
13. 総隣接数

## 損失関数

総損失は以下の 2 つの項の重み付き和です：

```
Loss = LAMBDA_DATA × L_data + LAMBDA_PDE × L_pde
```

### データ損失（L_data）

GNN の予測値と OpenFOAM の解との相対二乗誤差：

```
L_data = ||x_pred - x_true||² / (||x_true||² + ε)
```

### PDE 損失（L_pde）

メッシュ品質に基づく加重 PDE 残差：

```
L_pde = Σ w_i × |r_i|² / (Σ w_i × |b_i|² + ε)
```

ここで、`w_i` はメッシュ品質から計算される重みで、品質の悪いセルほど大きな重みが付与されます。

## 出力ファイル

実行後、以下のファイルが生成されます：

| ファイル名 | 説明 |
|-----------|------|
| `log_DATA{λ1}_PDE{λ2}.txt` | 訓練ログ（エポックごとの損失値など） |
| `training_history_DATA{λ1}_PDE{λ2}.png` | 訓練曲線のグラフ |
| `x_pred_train_{time}_rank{N}.dat` | 訓練データの予測結果（各 rank ごと） |
| `x_pred_val_{time}_rank{N}.dat` | 検証データの予測結果（各 rank ごと） |

## 技術的詳細

### メッシュ品質重み

メッシュ品質指標から重み `w_pde` を以下のように計算します：

```python
w_pde = 1.0 + skew_norm + non_ortho_norm + aspect_norm + size_jump_norm
w_pde = min(w_pde, W_PDE_MAX)  # 上限は 10.0
```

品質の悪いセル（スキューが大きい、非直交性が高いなど）では重みが大きくなり、PDE 残差の寄与が増加します。

### CSR 行列演算

係数行列 A はメモリ効率のため CSR 形式で保持し、`matvec_csr_torch` 関数で行列-ベクトル積を計算します。

### 複数 rank の統合学習

- 各 rank のグラフは独立したサンプルとして扱われます
- 正規化統計量（平均・標準偏差）は全 rank のデータから計算されます
- 各エポックで全ての (time, rank) ケースを用いて損失を計算し、モデルを更新します

## ライセンス

このプロジェクトのライセンスについては、リポジトリの所有者にお問い合わせください。

## 参考文献

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [GraphSAGE: Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)
- [OpenFOAM](https://www.openfoam.com/)
