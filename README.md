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
- **安定した学習**: 勾配クリッピング、学習率ウォームアップ、LayerNorm による安定化
- **改善されたモデル**: 残差接続（Skip connections）と LayerNorm を備えた GraphSAGE

## 必要条件

### 依存パッケージ

```bash
pip install torch
pip install torch-geometric
pip install numpy
pip install matplotlib
```

### 実行時の警告について

PyTorch と Transformers の組み合わせによっては、学習開始時に下記のような警告が出る場合があります。

```
UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
```

これは Transformers 側が内部的に使用している PyTorch の非推奨 API に対する通知であり、本プロジェクトのコードが原因ではありません。現状は挙動に影響しないため無視しても問題ありませんが、警告を消したい場合は PyTorch/Transformers を最新版へ更新するか、Transformers の修正リリースをお待ちください。

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

# 学習率ウォームアップ
USE_LR_WARMUP = True           # 学習初期に学習率を徐々に上げる
LR_WARMUP_EPOCHS = 10          # ウォームアップするエポック数

# 勾配クリッピング
USE_GRAD_CLIP = True           # 勾配ノルムをクリップして学習を安定化
GRAD_CLIP_MAX_NORM = 1.0       # 勾配ノルムの最大値

# メモリ効率化オプション
USE_LAZY_LOADING = True        # 遅延GPU転送（大規模データ向け）
USE_AMP = True                 # 混合精度学習（Automatic Mixed Precision）

# データキャッシュオプション（Optuna等での繰り返し学習を高速化）
USE_DATA_CACHE = True          # データをキャッシュファイルに保存し、2回目以降は高速ロード
CACHE_DIR = ".cache"           # キャッシュファイルの保存先ディレクトリ

# 損失関数の重み（学習モードの選択）
# - LAMBDA_DATA > 0, LAMBDA_PDE > 0: ハイブリッド学習（推奨）
# - LAMBDA_DATA = 0: 完全な教師なし学習（PDE損失のみ）
# - LAMBDA_PDE  = 0: 完全な教師あり学習（データ損失のみ）
LAMBDA_DATA = 0.1              # データ損失の重み（0 で教師なし学習）
LAMBDA_PDE  = 0.01             # PDE 損失の重み（0 で教師あり学習）
LAMBDA_GAUGE = 0.01            # ゲージ正則化係数（教師なし学習時）

# メッシュ品質重みオプション
USE_MESH_QUALITY_WEIGHTS = True  # メッシュ品質重みを使用（Falseで全セル等重み w=1）

# 対角スケーリングオプション（条件数改善）
USE_DIAGONAL_SCALING = True      # 対角スケーリングを適用（A_scaled = D^(-1/2) A D^(-1/2)）

# PDE損失の正規化方式
# "relative": ||r||²/||b||² (相対残差ノルム、物理的に意味があり推奨)
# "row_diag": r/diag で行ごと正規化 (値が極小になる問題あり)
# "none": ||r||²/(||Ax||²+||b||²+eps) 正規化
PDE_LOSS_NORMALIZATION = "relative"
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
    - `--lambda_gauge`: ゲージ正則化係数（教師なし学習時の定数モード抑制用、デフォルト 0.01）
    - `--search_lambda_gauge`: ゲージ正則化係数も Optuna で探索する
    - （自動探索対象）`lr`, `weight_decay`, `lambda_data`, `lambda_pde`, `hidden_channels`, `num_layers`（`--search_lambda_gauge` 指定時は `lambda_gauge` も探索）

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

# メッシュ品質重みを無効化（全セル等重み w=1）
python hyperparameter_search_optuna.py --no_mesh_quality_weights --trials 20 --data_dir ./data

# 対角スケーリングを無効化（条件数改善を行わない）
python hyperparameter_search_optuna.py --no_diagonal_scaling --trials 20 --data_dir ./data
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

### 7. 学習モードの選択

本システムは3つの学習モードをサポートしています：

| モード | LAMBDA_DATA | LAMBDA_PDE | 用途 |
|--------|-------------|------------|------|
| ハイブリッド学習 | > 0 | > 0 | データ損失とPDE損失の両方を使用（推奨） |
| 完全な教師あり学習 | > 0 | = 0 | データ損失のみ（従来の教師あり学習） |
| 完全な教師なし学習 | = 0 | > 0 | PDE損失のみ（PINNs的アプローチ） |

#### 教師なし学習モード（PINNs）

以下のいずれかの条件で教師なし学習モードになります：
1. `x_*_rank*.dat` ファイル（OpenFOAM の解）が存在しない
2. `LAMBDA_DATA = 0` を設定

#### 動作

- **自動検出**: データ読み込み時に `x_*_rank*.dat` の存在を確認
- **損失関数**: データ損失を使用せず、PDE 損失（`LAMBDA_PDE * pde_loss`）のみで学習
- **評価指標**: 相対誤差の代わりに PDE 残差 (`R_pred`) を使用
- **診断出力**: OpenFOAM 解との比較は省略し、GNN 解の PDE 残差のみを表示

#### 完全な教師あり学習モード

`LAMBDA_PDE = 0` を設定すると、PDE損失を計算せずデータ損失のみで学習します：
- PDE残差の計算をスキップするため、わずかに高速化
- 物理的制約なしの純粋なデータドリブン学習

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

#### ゲージ正則化（定数モード制御）

圧力ポアソン方程式では、解に任意の定数を加えても PDE を満たすため、教師データなしでは解が一意に定まりません（ゲージ自由度）。この問題を解決するため、教師なし学習モードでは**ゲージ正則化**を自動的に適用します：

$$
L_\mathrm{gauge} = \left( \frac{1}{N_\mathrm{cells}} \sum_{i=1}^{N_\mathrm{cells}} x_{\mathrm{pred},i} \right)^2
$$

これにより、予測値の平均がゼロに近づくよう制約され、解の不定性が解消されます。

**総損失（教師なし学習時）**:

$$
\mathcal{L} = \lambda_\mathrm{pde} \cdot L_\mathrm{pde} + \lambda_\mathrm{gauge} \cdot L_\mathrm{gauge}
$$

デフォルト値: λ_gauge = 0.01

**評価時のゲージ不変正規化**:

教師データがある場合の評価（相対誤差計算）では、予測値と真値の両方から平均を引いてから比較します。これにより、定数オフセットの違いを無視した公平な評価が可能になります：

$$
E_\mathrm{rel} = \frac{\| (x_\mathrm{pred} - \bar{x}_\mathrm{pred}) - (x_\mathrm{true} - \bar{x}_\mathrm{true}) \|_2}{\| x_\mathrm{true} - \bar{x}_\mathrm{true} \|_2 + \epsilon}
$$

## 検証誤差が頭打ちになるときのチェックリスト

- **学習率スケジューラを有効化**: デフォルトで `ReduceLROnPlateau` を使い、検証誤差が一定期間改善しないときに学習率を自動で 0.5 倍に下げます（最小学習率 `1e-6`）。`USE_LR_SCHEDULER` を `True` のままにしてください。
- **エポック数を増やす**: 収束に時間がかかる場合は `NUM_EPOCHS` や Optuna の `--num_epochs` を伸ばすと改善することがあります。
- **train/val 分割を見直す**: `TRAIN_FRACTION` を 0.8 以上にして学習データを増やすか、`max_num_cases` を増やしてサンプル多様性を上げてください。
- **損失重みのバランス**: `lambda_pde` が大きすぎるとデータ適合が弱くなる場合があります。`lambda_data` と併せて探索範囲を広げるか、`hyperparameter_search_optuna.py` で試行回数を増やしてください。
- **隠れチャネル / 層数**: `hidden_channels` と `num_layers` を広めに探索すると表現力不足を避けられます。

## モデルアーキテクチャ

改善版 GraphSAGE ベースの多層ニューラルネットワークを使用しています（デフォルトは 64 チャネル・4 層、Optuna から `hidden_channels` と `num_layers` を調整可能）。

### アーキテクチャの特徴

- **入力射影層**: 入力特徴量を隠れ次元に射影（残差接続のため）
- **LayerNorm**: 各畳み込み層の後に正規化を適用し、学習を安定化
- **残差接続**: 中間層で残差接続を適用し、勾配の流れを改善

| 層 | 入力次元 | 出力次元 | 活性化関数 | 正規化 | 残差接続 |
|---|---------|---------|----------|--------|---------|
| input_proj | 13 | 64 | ReLU | - | - |
| conv1 | 64 | 64 | ReLU | LayerNorm | ✓ |
| conv2 | 64 | 64 | ReLU | LayerNorm | ✓ |
| conv3 | 64 | 64 | ReLU | LayerNorm | ✓ |
| conv4 | 64 | 1 | なし | - | - |

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

総損失は以下の 3 つの項の重み付き和です：

$$
\mathcal{L} = \lambda_\mathrm{data} \cdot L_\mathrm{data} + \lambda_\mathrm{pde} \cdot L_\mathrm{pde} + \lambda_\mathrm{gauge} \cdot L_\mathrm{gauge}
$$

デフォルト値: λ_data = 0.1, λ_pde = 1.0, λ_gauge = 0.01

### データ損失 (L_data)

GNN の予測値と OpenFOAM の解との rank ごとに正規化した平均二乗誤差：

$$
L_\mathrm{data} = \frac{1}{N_\mathrm{cases}} \sum_{k=1}^{N_\mathrm{cases}} \mathrm{MSE}\left( \tilde{x}^{(k)}_\mathrm{pred}, \tilde{x}^{(k)}_\mathrm{true} \right)
$$

ここで、x̃ は rank ごとの平均・標準偏差で正規化された値：

$$
\tilde{x}_i = \frac{x_i - \mu_\mathrm{rank}}{\sigma_\mathrm{rank} + \epsilon}
$$

**注意**: 教師なし学習モード（`x_*_rank*.dat` がない場合）では、L_data = 0 となり、PDE 損失のみで学習します。

### PDE 損失 (L_pde)

`PDE_LOSS_NORMALIZATION` 設定により3つの正規化方式を選択できます：

**1. `"relative"` (推奨)**: 相対残差ノルム

$$
L_\mathrm{pde} = \frac{\| \sqrt{w} \odot r \|_2^2}{\| \sqrt{w} \odot b \|_2^2 + \epsilon}, \quad r = Ax_\mathrm{pred} - b
$$

- 物理的に意味がある（||r||/||b|| ≈ 1 のとき L_pde ≈ 1）
- data_loss とバランスしやすい値域（0.01〜100 程度）
- 推奨: `LAMBDA_PDE = 0.01`

**2. `"row_diag"`**: 行ごと正規化（対角スケール）

$$
L_\mathrm{pde} = \frac{1}{N_\mathrm{cells}} \sum_{i=1}^{N_\mathrm{cells}} w_i \left( \frac{r_i}{|D_{ii}|} \right)^2
$$

- 各行のスケールを揃えるが、値が極端に小さくなる問題あり

**3. `"none"`**: 複合正規化

$$
L_\mathrm{pde} = \frac{\| \sqrt{w} \odot r \|_2^2}{\| \sqrt{w} \odot Ax \|_2^2 + \| \sqrt{w} \odot b \|_2^2 + \epsilon}
$$

**共通の記号**:
- A: 係数行列（CSR 形式）
- b: 右辺ベクトル（ソース項）
- D: A の対角成分
- w: メッシュ品質に基づく重みベクトル

### ゲージ損失 (L_gauge)

セル体積で重み付けした予測値の平均の二乗（定数モード抑制用）：

$$
L_\mathrm{gauge} = \left( \frac{\sum_{i=1}^{N_\mathrm{cells}} V_i \cdot x_{\mathrm{pred},i}}{\sum_{i=1}^{N_\mathrm{cells}} V_i} \right)^2
$$

- V_i: セル i の体積

セル体積による重み付けにより、物理的に意味のある平均値を計算します。

### 対角スケーリング（条件数改善）

`USE_DIAGONAL_SCALING = True` の場合、係数行列と右辺ベクトルに対角スケーリングを適用して条件数を改善します：

$$
\tilde{A} = D^{-1/2} A D^{-1/2}, \quad \tilde{b} = D^{-1/2} b, \quad \tilde{x} = D^{1/2} x
$$

ここで D は A の対角成分からなる対角行列です。これにより：

- スケーリング後の対角成分がすべて 1 に正規化される
- 行列の条件数が大幅に改善される（典型的には数桁）
- スパース構造（グラフ構造）は保持される
- GNN の学習が安定化し、PDE 損失の勾配が適切なスケールになる

学習開始時に、スケーリング前後の条件数推定値がログ出力されます。

### 検証誤差（Optuna 最適化指標）

Optuna のハイパーパラメータ最適化では、検証データに対する相対誤差の平均を最小化します：

$$
E_\mathrm{val} = \frac{1}{N_\mathrm{val}} \sum_{k=1}^{N_\mathrm{val}} \frac{\| x^{(k)}_\mathrm{pred} - x^{(k)}_\mathrm{true} \|_2}{\| x^{(k)}_\mathrm{true} \|_2 + \epsilon}
$$

教師なし学習モードでは、検証誤差の代わりに PDE 残差 R を使用します。

### メッシュ品質重み (w)

メッシュ品質指標から重み w_i を以下のように計算します：

$$
w_i = \mathrm{clip}\left(1 + (q_\mathrm{skew} - 1) + (q_\mathrm{nonOrtho} - 1) + (q_\mathrm{aspect} - 1) + (q_\mathrm{sizeJump} - 1), \ 1, \ w_\mathrm{max}\right)
$$

各品質指標は基準値で正規化されます：

| 指標 | 基準値 | 計算式 |
|-----|-------|--------|
| スキュー | 0.2 | q_skew = clip(skew / 0.2, 0, 5) |
| 非直交性 | 10.0° | q_nonOrtho = clip(nonOrtho / 10, 0, 5) |
| アスペクト比 | 5.0 | q_aspect = clip(aspect / 5, 0, 5) |
| サイズジャンプ | 1.5 | q_sizeJump = clip(sizeJump / 1.5, 0, 5) |

品質の悪いセル（スキューが大きい、非直交性が高いなど）では重みが大きくなり、PDE 残差の寄与が増加します。w_max = 10.0 で上限をクリップします。

## 出力ファイル

実行後、以下のファイルが生成されます：

| ファイル名 | 説明 |
|-----------|------|
| `log_DATA{λ1}_PDE{λ2}.txt` | 訓練ログ（エポックごとの損失値など） |
| `training_history_DATA{λ1}_PDE{λ2}.png` | 訓練曲線のグラフ |
| `x_pred_train_{time}_rank{N}.dat` | 訓練データの予測結果（各 rank ごと、従来形式） |
| `x_pred_val_{time}_rank{N}.dat` | 検証データの予測結果（各 rank ごと、従来形式） |
| `pressure_pred_train_{time}_rank{N}.vtk` | 予測圧力（座標付きVTK） |
| `pressure_pred_val_{time}_rank{N}.vtk` | 検証データの予測圧力（座標付きVTK） |
| `pressure_true_train_{time}_rank{N}.vtk` | 真値圧力（座標付きVTK、x_trueがある場合） |
| `pressure_true_val_{time}_rank{N}.vtk` | 検証データの真値圧力（座標付きVTK） |
| `pressure_compare_train_{time}_rank{N}.vtk` | 真値・予測値・誤差の比較（座標付きVTK） |
| `pressure_compare_val_{time}_rank{N}.vtk` | 検証データの比較（座標付きVTK） |
| `pressure_comparison_{prefix}.png` | 2D断面での圧力場比較（真値、予測値、差分） |
| `scatter_comparison_{prefix}.png` | 散布図（真値 vs 予測値）と誤差ヒストグラム |
| `error3d_{prefix}.png` | 誤差場の3D散布図 |
| `error2d_yMid_{prefix}.png` | 誤差場とメッシュ品質重みの2Dカラーマップ |

### VTK ファイル形式（3次元可視化用）

VTK ファイルは Legacy ASCII 形式（POLYDATA）で出力されます。各ファイルにはセル中心座標とスカラーデータが含まれます。

**予測値のみのVTKファイル（`pressure_pred_*.vtk`）**:
- SCALARS: `p_pred`（予測圧力）

**真値のみのVTKファイル（`pressure_true_*.vtk`）**:
- SCALARS: `p_true`（真値圧力）

**比較用VTKファイル（`pressure_compare_*.vtk`）**:
- SCALARS: `p_true`（真値圧力）、`p_pred`（予測圧力）、`error`（予測誤差）

#### VTK ファイル構造例

```vtk
# vtk DataFile Version 3.0
Pressure field data
ASCII
DATASET POLYDATA
POINTS 1000 float
1.234567890e-01 2.345678901e-02 3.456789012e-03
...
VERTICES 1000 2000
1 0
1 1
...
POINT_DATA 1000
SCALARS p_pred float 1
LOOKUP_TABLE default
1.000000000e+02
...
```

これらの VTK ファイルは、ParaView で直接読み込んで3次元可視化できます。

## 技術的詳細

### CSR 行列演算

係数行列 A はメモリ効率のため CSR 形式で保持し、`matvec_csr_torch` 関数で行列-ベクトル積を計算します。

### 複数 rank の統合学習

- 各 rank のグラフは独立したサンプルとして扱われます
- 正規化統計量（平均・標準偏差）は全 rank のデータから計算されます
- 各エポックで全ての (time, rank) ケースを用いて損失を計算し、モデルを更新します

## 変更履歴

### 2025-12-23: PDE損失正規化の修正と可視化機能の追加

#### PDE損失正規化の修正
- **`PDE_LOSS_NORMALIZATION` 設定を追加**: 3つの正規化方式から選択可能
  - `"relative"` (推奨): 相対残差ノルム `||r||²/||b||²`
  - `"row_diag"`: 対角成分による行ごと正規化
  - `"none"`: 複合正規化 `||r||²/(||Ax||²+||b||²)`
- **`LAMBDA_PDE` を 1.0 から 0.01 に変更**: 新しい正規化方式に合わせて調整
- **問題の修正**: 従来の行ごと正規化（r/diag）では PDE_loss が 1e-10〜1e-12 と極端に小さくなり、実質的に PDE 損失から学習が行われない問題を修正

#### 学習モードの柔軟化
- **完全な教師あり学習（LAMBDA_PDE = 0）に対応**: データ損失のみで学習可能
- **完全な教師なし学習（LAMBDA_DATA = 0）に対応**: PDE損失のみで学習可能
- **学習開始時にモード表示**: ハイブリッド/教師あり/教師なしを明示
- **Optuna スクリプトに lambda 範囲オプション追加**:
  - `--lambda_data_min`, `--lambda_data_max`
  - `--lambda_pde_min`, `--lambda_pde_max`
  - 0 を指定して固定値として使用可能

#### 可視化機能の追加
- **圧力場比較プロット**: 2D断面での真値・予測値・差分の3パネル表示
- **散布図**: 真値 vs 予測値の散布図（45度線と回帰直線付き）
- **誤差ヒストグラム**: 予測誤差の分布を可視化
- **相関係数・RMSE・相対誤差を図中に表示**

### 2025-12-22: PDE損失の改善とモデルアーキテクチャ更新

#### バグ修正
- **val評価での対角スケーリング不整合を修正**: Final diagnostics (val) でも train と同じ対角スケーリングを適用するように修正

#### パフォーマンス改善
- **apply_diagonal_scaling_csr() のベクトル化**: Pythonループを NumPy のベクトル化演算に置き換え（10-100倍高速化）
- **matvec_csr_numpy() の高速化**: scipy.sparse.csr_matrix を使用（数十倍高速化）
- **validation 評価コードの重複削除**: `evaluate_validation_cases()` 共通関数を作成（約140行削減）
- **不要な empty_cache() 呼び出しを削除**: 学習ループ内での呼び出しを削除

#### PDE 損失の安定化
- **安定した残差正規化**: `||r||² / (||Ax||² + ||b||² + eps)` または行ごと正規化
- **行ごとスケーリング（Jacobi的前処理）**: 対角成分で残差を正規化し、勾配のバランスを改善
- **EPS_RES を 1e-12 から 1e-8 に増加**: 数値安定性を向上

#### ゲージ正則化の改善
- **セル体積による重み付け**: 物理的に意味のある体積加重平均を使用

#### モデルアーキテクチャの改善
- **入力射影層の追加**: 残差接続のため入力次元を隠れ次元に射影
- **LayerNorm の追加**: 各畳み込み層の後に正規化を適用
- **残差接続（Skip connections）の追加**: 勾配の流れを改善

#### 学習の安定化
- **勾配クリッピング**: `USE_GRAD_CLIP = True` で勾配ノルムをクリップ
- **学習率ウォームアップ**: `USE_LR_WARMUP = True` で学習初期に学習率を徐々に増加

#### デフォルト値の調整
- **LAMBDA_PDE**: 0.0001 → 1.0（教師なし学習でのPDE損失の重要性を反映）
- **Optuna 探索範囲**: lambda_pde の範囲を 0.01〜10.0 に拡大

#### 教師なし学習の改善
- **出力スケールの推定**: `b` と対角成分から `x_std` を推定（ダミー値 1.0 の代わり）

#### API 更新
- **非推奨 PyTorch AMP API の更新**: `torch.cuda.amp.*` → `torch.amp.*`

## ライセンス

このプロジェクトのライセンスについては、リポジトリの所有者にお問い合わせください。

## 参考文献

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [GraphSAGE: Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)
- [OpenFOAM](https://www.openfoam.com/)
