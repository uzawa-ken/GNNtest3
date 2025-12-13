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
- **教師なし学習対応**: OpenFOAM の解がない場合は自動的に PDE 損失のみで学習（PINNs 的アプローチ）

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
└── data/                              # データディレクトリ
    ├── processor2/
    │   └── gnn/
    │       ├── A_csr_{time}.dat       # 係数行列（CSR 形式）
    │       ├── pEqn_{time}_rank2.dat  # PDE 方程式情報
    │       └── x_{time}_rank2.dat     # OpenFOAM の解
    ├── processor4/
    │   └── gnn/
    │       ├── A_csr_{time}.dat
    │       ├── pEqn_{time}_rank4.dat
    │       └── x_{time}_rank4.dat
    ├── processor5/
    │   └── gnn/
    │       └── ...
    └── processor7/
        └── gnn/
            └── ...
```

### 複数プロセス対応

本プログラムは MPI 並列計算で領域分割された OpenFOAM の結果に対応しています：

- 各 rank（プロセス）のデータは独立したグラフとして扱われます
- `data/processor*/gnn/` ディレクトリを自動探索し、全ての rank のデータを検出します
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

### CSR 行列ファイル（`A_csr_{time}.dat`）

PDE の係数行列を CSR（Compressed Sparse Row）形式で格納したファイルです。各 processor ディレクトリ内に配置します（rank 番号なし）。

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

OpenFOAM の解析結果から上記形式のデータファイルを生成し、`./data/processor{N}/gnn/` ディレクトリに配置してください。

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
HIDDEN_CHANNELS = 64           # 中間層のチャネル数
NUM_LAYERS      = 4            # GraphSAGE の層数

# 学習率スケジューラ（ReduceLROnPlateau）
USE_LR_SCHEDULER = True        # 検証誤差が停滞したら学習率を下げる
LR_SCHED_FACTOR = 0.5          # 学習率を何倍に下げるか
LR_SCHED_PATIENCE = 20         # 何エポック改善が無ければ下げるか
LR_SCHED_MIN_LR = 1e-6         # 学習率の下限

# メモリ効率化オプション
USE_LAZY_LOADING = True        # 遅延GPU転送（大規模データ向け）
USE_AMP = True                 # 混合精度学習（Automatic Mixed Precision）

# データキャッシュオプション（Optuna等での繰り返し学習を高速化）
USE_DATA_CACHE = True          # データをキャッシュファイルに保存し、2回目以降は高速ロード
CACHE_DIR = ".cache"           # キャッシュファイルの保存先ディレクトリ

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
- 検出された gnn_dir 数
- 使用する (time, rank) ペアの総数
- train/val 分割の詳細

### 4. ハイパーパラメータの自動探索（Optuna）

最終検証誤差（相対誤差）を最小化するように、学習率や損失の重みを自動探索するサンプルスクリプトを用意しています。

1. Optuna をインストール

    ```bash
    pip install optuna
    ```

2. 探索を実行

    ```bash
    python hyperparameter_search_optuna.py --trials 20 --data_dir ./data --num_epochs 200 \
        --train_fraction 0.8 --random_seed 42
    ```

    - `--trials`: 試行回数（多いほど精度向上が見込まれますが計算時間が増加します）
    - `--num_epochs`: 1 試行あたりのエポック数（短めに設定すると探索が高速化します）
    - `--max_num_cases`: 1 試行で使用する (time, rank) ペアの最大数（デフォルト 30）
    - `--train_fraction`: train/val 分割の割合（デフォルト 0.8）
    - `--random_seed`: Optuna のサンプラーと学習の乱数シード（デフォルト 42）
    - `--log_file`: 試行番号と検証誤差（タブ区切り）を試行ごとに追記するログファイル（デフォルト `optuna_trials_history.tsv`）
    - （自動探索対象）`lr`, `weight_decay`, `lambda_data`, `lambda_pde`, `hidden_channels`, `num_layers`

3. 実行後、最小の検証誤差と最適パラメータがコンソールに表示されます。また、ログファイルには各試行の番号と検証誤差が時系列で追記されます。

### 5. メモリ効率化オプション

大規模グラフ（セル数が多い、または多数のケースを学習する場合）でGPUメモリ不足が発生する場合、以下のオプションが有効です。

#### 遅延GPU転送（Lazy Loading）

```python
USE_LAZY_LOADING = True  # デフォルト: 有効
```

- **動作**: データを CPU メモリに保持し、学習時に必要なケースのみ GPU に転送
- **効果**: GPU メモリ使用量を大幅に削減（ケース数に依存しなくなる）
- **トレードオフ**: データ転送のオーバーヘッドにより、学習速度がやや低下

#### 混合精度学習（AMP: Automatic Mixed Precision）

```python
USE_AMP = True  # デフォルト: 有効（CUDA 環境のみ）
```

- **動作**: 順伝播と損失計算を FP16/BF16 で実行し、勾配スケーリングで数値安定性を確保
- **効果**: GPU メモリ使用量を約 50% 削減、学習速度が向上（最新 GPU では特に効果的）
- **要件**: CUDA 対応 GPU が必要（CPU モードでは自動的に無効化）

#### Optuna での使用

```bash
# 両方有効（デフォルト）
python hyperparameter_search_optuna.py --trials 20 --data_dir ./data

# 遅延GPU転送を無効化
python hyperparameter_search_optuna.py --no_lazy_loading --trials 20 --data_dir ./data

# AMP を無効化
python hyperparameter_search_optuna.py --no_amp --trials 20 --data_dir ./data
```

### 6. データキャッシュ機能

Optuna などで繰り返し学習を行う場合、データの読み込みとグラフ作成が毎回発生し、時間がかかります。データキャッシュ機能を使用すると、初回のみファイルから読み込み、2回目以降はキャッシュから高速にロードできます。

#### 基本設定

```python
USE_DATA_CACHE = True     # デフォルト: 有効
CACHE_DIR = ".cache"      # キャッシュファイルの保存先
```

- **動作**: 初回実行時に NumPy 配列レベルのデータを pickle 形式でキャッシュに保存。2回目以降はキャッシュから高速に読み込み
- **効果**: Optuna の 2 試行目以降でデータ読み込み時間を大幅に短縮（特に大規模データで効果的）
- **自動無効化**: ソースファイル（pEqn, x, A_csr）がキャッシュより新しい場合は自動的に再読み込み

#### キャッシュの仕組み

1. データディレクトリと (time, rank) ペアのリストからハッシュキーを生成
2. `.cache/raw_cases_{hash}.pkl` としてキャッシュを保存
3. 読み込み時にソースファイルの更新日時をチェックし、古いキャッシュは自動的に再作成

#### Optuna での使用

```bash
# キャッシュ有効（デフォルト）
python hyperparameter_search_optuna.py --trials 20 --data_dir ./data

# キャッシュを無効化（毎回ファイルから読み込む）
python hyperparameter_search_optuna.py --no_cache --trials 20 --data_dir ./data

# キャッシュディレクトリを指定
python hyperparameter_search_optuna.py --cache_dir /tmp/gnn_cache --trials 20 --data_dir ./data
```

#### キャッシュのクリア

キャッシュを手動でクリアしたい場合は、キャッシュディレクトリを削除してください：

```bash
rm -rf .cache
```

### 7. 教師なし学習モード（PINNs）

`x_*_rank*.dat` ファイル（OpenFOAM の解）が存在しない場合、自動的に教師なし学習モード（Physics-Informed Neural Networks, PINNs 的アプローチ）に切り替わります。

#### 動作

- **自動検出**: データ読み込み時に `x_*_rank*.dat` の存在を確認
- **損失関数**: データ損失を使用せず、PDE 損失（`LAMBDA_PDE * pde_loss`）のみで学習
- **評価指標**: 相対誤差の代わりに PDE 残差 (`R_pred`) を使用
- **診断出力**: OpenFOAM 解との比較は省略し、GNN 解の PDE 残差のみを表示

#### 使用例

```
data/
├── processor0/
│   └── gnn/
│       ├── A_csr_{time}.dat       # 必須: 係数行列
│       ├── pEqn_{time}_rank0.dat  # 必須: PDE 方程式情報
│       └── (x_{time}_rank0.dat)   # 省略可: OpenFOAM の解
```

上記のように `x_*_rank*.dat` を省略すると、自動的に教師なし学習モードで実行されます。

#### 注意事項

- 教師なし学習では、学習の収束判定に PDE 残差を使用します
- 一部のケースのみ `x_*_rank*.dat` がある場合は、そのケースのみでデータ損失を計算します（ハイブリッドモード）
- 誤差場の可視化は `x_*_rank*.dat` が存在するケースのみで行われます

## 検証誤差が頭打ちになるときのチェックリスト

- **学習率スケジューラを有効化**: デフォルトで `ReduceLROnPlateau` を使い、検証誤差が一定期間改善しないときに学習率を自動で 0.5 倍に下げます（最小学習率 `1e-6`）。`USE_LR_SCHEDULER` を `True` のままにしてください。
- **エポック数を増やす**: 収束に時間がかかる場合は `NUM_EPOCHS` や Optuna の `--num_epochs` を伸ばすと改善することがあります。
- **train/val 分割を見直す**: `TRAIN_FRACTION` を 0.8 以上にして学習データを増やすか、`max_num_cases` を増やしてサンプル多様性を上げてください。
- **損失重みのバランス**: `lambda_pde` が大きすぎるとデータ適合が弱くなる場合があります。`lambda_data` と併せて探索範囲を広げるか、`hyperparameter_search_optuna.py` で試行回数を増やしてください。
- **隠れチャネル / 層数**: `hidden_channels` と `num_layers` を広めに探索すると表現力不足を避けられます。

## モデルアーキテクチャ

GraphSAGE ベースの多層ニューラルネットワークを使用しています（デフォルトは 64 チャネル・4 層、Optuna から `hidden_channels` と `num_layers` を調整可能）。

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

$$
\mathcal{L} = \lambda_{\text{data}} \cdot L_{\text{data}} + \lambda_{\text{pde}} \cdot L_{\text{pde}}
$$

デフォルト値: $\lambda_{\text{data}} = 0.1$, $\lambda_{\text{pde}} = 0.0001$

### データ損失（$L_{\text{data}}$）

GNN の予測値と OpenFOAM の解との rank ごとに正規化した平均二乗誤差：

$$
L_{\text{data}} = \frac{1}{N_{\text{cases}}} \sum_{k=1}^{N_{\text{cases}}} \text{MSE}\left( \tilde{x}^{(k)}_{\text{pred}}, \tilde{x}^{(k)}_{\text{true}} \right)
$$

ここで、$\tilde{x}$ は rank ごとの平均・標準偏差で正規化された値：

$$
\tilde{x}_i = \frac{x_i - \mu_{\text{rank}}}{\sigma_{\text{rank}} + \epsilon}
$$

**注意**: 教師なし学習モード（`x_*_rank*.dat` がない場合）では、$L_{\text{data}} = 0$ となり、PDE 損失のみで学習します。

### PDE 損失（$L_{\text{pde}}$）

メッシュ品質に基づく加重 PDE 残差の二乗：

$$
L_{\text{pde}} = \frac{1}{N_{\text{cases}}} \sum_{k=1}^{N_{\text{cases}}} R^{(k)2}
$$

ここで、$R^{(k)}$ は各ケースの重み付き相対残差：

$$
R^{(k)} = \frac{\| \sqrt{w} \odot r \|_2}{\| \sqrt{w} \odot b \|_2 + \epsilon}, \quad r = Ax_{\text{pred}} - b
$$

- $A$: 係数行列（CSR 形式）
- $b$: 右辺ベクトル（ソース項）
- $w$: メッシュ品質に基づく重みベクトル
- $\odot$: 要素ごとの積（Hadamard 積）

### 検証誤差（Optuna 最適化指標）

Optuna のハイパーパラメータ最適化では、検証データに対する相対誤差の平均を最小化します：

$$
\text{val\_error} = \frac{1}{N_{\text{val}}} \sum_{k=1}^{N_{\text{val}}} \frac{\| x^{(k)}_{\text{pred}} - x^{(k)}_{\text{true}} \|_2}{\| x^{(k)}_{\text{true}} \|_2 + \epsilon}
$$

教師なし学習モードでは、検証誤差の代わりに PDE 残差 $R$ を使用します。

### メッシュ品質重み（$w$）

メッシュ品質指標から重み $w_i$ を以下のように計算します：

$$
w_i = \text{clip}\left(1 + (q_{\text{skew}} - 1) + (q_{\text{nonOrtho}} - 1) + (q_{\text{aspect}} - 1) + (q_{\text{sizeJump}} - 1), \ 1, \ w_{\text{max}}\right)
$$

各品質指標は基準値で正規化されます：

| 指標 | 基準値 | 計算式 |
|-----|-------|--------|
| スキュー | 0.2 | $q_{\text{skew}} = \text{clip}(\text{skew} / 0.2, 0, 5)$ |
| 非直交性 | 10.0° | $q_{\text{nonOrtho}} = \text{clip}(\text{nonOrtho} / 10, 0, 5)$ |
| アスペクト比 | 5.0 | $q_{\text{aspect}} = \text{clip}(\text{aspect} / 5, 0, 5)$ |
| サイズジャンプ | 1.5 | $q_{\text{sizeJump}} = \text{clip}(\text{sizeJump} / 1.5, 0, 5)$ |

品質の悪いセル（スキューが大きい、非直交性が高いなど）では重みが大きくなり、PDE 残差の寄与が増加します。$w_{\text{max}} = 10.0$ で上限をクリップします。

## 出力ファイル

実行後、以下のファイルが生成されます：

| ファイル名 | 説明 |
|-----------|------|
| `log_DATA{λ1}_PDE{λ2}.txt` | 訓練ログ（エポックごとの損失値など） |
| `training_history_DATA{λ1}_PDE{λ2}.png` | 訓練曲線のグラフ |
| `x_pred_train_{time}_rank{N}.dat` | 訓練データの予測結果（各 rank ごと） |
| `x_pred_val_{time}_rank{N}.dat` | 検証データの予測結果（各 rank ごと） |

## 技術的詳細

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
