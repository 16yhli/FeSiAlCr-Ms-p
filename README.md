# FeSiAlCrMs

**Data-driven design of Fe–Si–Al–Cr soft magnetic alloys with high saturation magnetization (Ms).**

A Gaussian process regression (GPR) surrogate is coupled with Bayesian optimization and active
learning to search the Fe–Si–Al–Cr composition space, and the AI-recommended alloys are validated
experimentally. The repository contains the complete analysis pipeline: feature engineering,
surrogate model selection, uncertainty-aware optimization, iterative active-learning rounds, and
the resulting fitted models.

---

## Method

| Stage | What it does | Script |
|---|---|---|
| Feature engineering | Composition (at% Fe/Si/Al/Cr) is augmented with two physics descriptors: **valence electron concentration** (`vec`) and **atomic radius mismatch** (`atom`) | inline in every script, derived from elemental valence / radius / weight |
| Surrogate model selection | Six GPR kernel strategies are compared under 5-fold cross-validation, with hyperparameters tuned by Optuna (TPE sampler, 60 trials) | `FeSiALCrMsModel.py` |
| Bayesian optimization | GPR surrogate + Expected Improvement (EI) acquisition over the composition space | `0-BOxunyou.py`, `1-GPRxunyou.py` |
| Active learning | Four iterative rounds; each round recommends new alloys, which are synthesised, measured, and fed back into the model | `1-GPRqueding.py`, `1-0GPRtuijian.py` … `1-3GPRtuijian.py` |
| Pareto recommendation | EI vs. density Pareto front to balance exploitation and exploration, producing the final recommended compositions | `Ms-P pareto.py` |

### Surrogate model

```python
# exactly as implemented in 1-GPRqueding.py and 1-3GPRtuijian.py
scaler = StandardScaler()                                  # X standardised before fitting
kernel = ConstantKernel(100.0) * Matern(length_scale=10.0, nu=0.5)

GaussianProcessRegressor(
    kernel=kernel,
    alpha=alpha_weights,          # per-source observation noise (see below)
    normalize_y=False,            # keep the true Ms variance mapping
    n_restarts_optimizer=10,
    random_state=42,
).fit(scaler.fit_transform(X), y)
```

- **Features** (`X`): `Fe_at, Si_at, Al_at, Cr_at, atom, vec`
- **Target** (`y`): `Ms`
- **Data-source-aware noise**: every sample carries a `source` label, and the GP noise term is
  scaled per source — `alpha = source.map({0: 225.0, 1: 49.0, 2: 9.0})`. Less reliable data are
  assigned larger assumed noise, so the fit is dominated by the most trustworthy measurements.
- **Saved artefacts**: each `.pkl` is a dict `{'scaler': ..., 'model': ..., 'features': [...]}` —
  the fitted `StandardScaler` is stored alongside the model so that new compositions are scaled
  identically.
- **Reproducibility**: the random seed is fixed to `42` throughout (`random`, `numpy`, Optuna
  TPESampler, and the GPR itself).

---

## Pipeline and execution order

The scripts form a **chained model-iteration pipeline**, not a set of independent tools.
Each step loads the model produced by the previous step, so the order matters:

```
combined_training_data.csv
        │
        ▼
① 1-GPRqueding.py ──────────────► ms_gpr_matern05_model.pkl        (V1)
        │
        ▼
② 1-0GPRtuijian.py ─────────────► round-1 recommendations
        │
        ▼
   1-FeSiAlCr.xlsx  (measured Ms of the round-1 alloys)
        │
        ▼
③ 1-1GPRtuijian.py ─────────────► ms_gpr_matern05_model_v2.pkl     (V2)
        │
        ▼
   2-FeSiAlCr.xlsx  (measured Ms of the round-2 alloys)
        │
        ▼
④ 1-2GPRtuijian.py ─────────────► ms_gpr_matern05_model_v3.pkl     (V3)
        │
        ▼
   3-FeSiAlCr.xlsx  (measured Ms of the round-3 alloys)
        │
        ▼
⑤ 1-3GPRtuijian.py ─────────────► ms_gpr_matern05_model_v4.pkl     (V4)
        │
        ▼
⑥ Ms-P pareto.py ───────────────► EI–density Pareto front, final recommendations
```

**Run them in this order:** `1-GPRqueding` → `1-0GPRtuijian` → `1-1GPRtuijian` → `1-2GPRtuijian`
→ `1-3GPRtuijian` → `Ms-P pareto`

> `Ms-P pareto.py` is the **end** of the released chain — it loads the **V4** model. `Ms-P` stands
> for "Ms Pareto", not for a step named "Ms-P".
>
> V5 (V4 further updated with two validation experiments) is **not released**; see the note at the
> end of the Results section.

### Independent analysis scripts

These four do **not** sit on the chain; they only need `combined_training_data.csv`:

| Script | Purpose |
|---|---|
| `FeSiALCrMsModel.py` | Kernel-strategy benchmark: 6 GPR kernels × Optuna tuning × 5-fold CV |
| `0-BOxunyou.py` | Bayesian optimization search (writes `BO_Optimization_History_Square.pdf`) |
| `1-GPRxunyou.py` | GPR-based design-space search |
| `FeSiAlCrtuijian.py` | Alloy recommendation |

---

## Repository contents

```
FeSiAlCrMs/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── 1-GPRqueding.py               # ① train the initial model (V1)
├── 1-0GPRtuijian.py              # ② active-learning round 1
├── 1-1GPRtuijian.py              # ③ round 2  -> V2
├── 1-2GPRtuijian.py              # ④ round 3  -> V3
├── 1-3GPRtuijian.py              # ⑤ round 4  -> V4
├── Ms-P pareto.py                # ⑥ EI–density Pareto recommendation (loads V4)
│
├── FeSiALCrMsModel.py            # kernel-strategy benchmark (independent)
├── 0-BOxunyou.py                 # Bayesian optimization (independent)
├── 1-GPRxunyou.py                # GPR search (independent)
├── FeSiAlCrtuijian.py            # alloy recommendation (independent)
│
├── combined_training_data.csv    # training set, 119 samples
├── FeSiAlCrMs.xlsx               # source dataset (sheets: FeCr, FeSiAl2)
├── 1-FeSiAlCr.xlsx               # round-1 alloys + measured Ms
├── 2-FeSiAlCr.xlsx               # round-2 alloys + measured Ms
├── 3-FeSiAlCr.xlsx               # round-3 alloys + measured Ms
│
├── ms_gpr_matern05_model.pkl     # V1
├── ms_gpr_matern05_model_v2.pkl  # V2
├── ms_gpr_matern05_model_v3.pkl  # V3
└── ms_gpr_matern05_model_v4.pkl  # V4  (final model; used by Ms-P pareto.py)
```

---

## Data

### `combined_training_data.csv` — 119 training samples

| Column | Description |
|---|---|
| `Fe_at`, `Si_at`, `Al_at`, `Cr_at` | composition in atomic percent |
| `atom` | atomic radius mismatch |
| `vec` | valence electron concentration |
| `Ms` | saturation magnetization (emu/g) — the regression target |
| `source` | data provenance (see below) |

### Data provenance (`source`)

| `source` | Meaning | Samples | GP noise weight |
|---|---|---|---|
| `0` | **Theory** — computed/theoretical data | 51 | 225.0 |
| `1` | **Literature** — experimental values reported in the literature | 68 | 49.0 |
| `2` | **Experiment** — this work's own measurements (three successive batches) | 3 per batch | 9.0 |

The noise weight decreases with increasing trustworthiness, which is exactly what
`alpha_weights = source.map({0: 225.0, 1: 49.0, 2: 9.0})` encodes.

### Other data files

| File | Role |
|---|---|
| `FeSiAlCrMs.xlsx` | Source dataset. Two sheets — `FeCr` (51 rows) and `FeSiAl2` (70 rows) — with columns `Fe_at, Si_at, Al_at, Cr_at, avg_valence_electrons, radius_diff_percent, Ms`. `combined_training_data.csv` is the merged, renamed version of these sheets (`avg_valence_electrons` → `vec`, `radius_diff_percent` → `atom`), so the raw data is fully traceable. |
| `1-FeSiAlCr.xlsx`, `2-FeSiAlCr.xlsx`, `3-FeSiAlCr.xlsx` | Compositions recommended in each active-learning round, together with their measured `Ms` (`source = 2`). These are consumed by `1-1` / `1-2` / `1-3GPRtuijian.py`. |

> No script reads `FeSiAlCrMs.xlsx` directly — it is provided as the raw, traceable source of the
> training set.

---

## Installation

```bash
git clone https://github.com/16yhli/FeSiAlCr-Ms-p.git
cd FeSiAlCrMs
pip install -r requirements.txt
```

Requires Python 3.8+. Dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`,
`optuna`, `joblib`, `openpyxl`.

---

## Usage

Every script resolves its file paths relative to its own location
(`os.path.dirname(os.path.abspath(__file__))`), so it can be launched from any working directory
with no path configuration:

```bash
python FeSiALCrMsModel.py        # reproduce the kernel-strategy benchmark
python 1-3GPRtuijian.py          # run an active-learning round
python "Ms-P pareto.py"          # EI–density Pareto recommendation
```

> Note the space in `Ms-P pareto.py` — quote the filename when calling it from a shell.

---

## Reproducing the models

The four bundled `*.pkl` files (V1–V4) are the **exact models produced by the active-learning
pipeline**, archived for reference. They were serialised with an earlier `numpy` / `scikit-learn`
version, so unpickling may fail in a
different environment. This is a known limitation, and the affected scripts raise an explicit
error (e.g. `1-0GPRtuijian.py`: *"模型文件 ms_gpr_matern05_model.pkl 未找到"*).

Two options:

1. **Load directly** — install the same `numpy` / `scikit-learn` versions that produced them.
2. **Retrain (recommended)** — run `1-GPRqueding.py` to regenerate V1, then follow the pipeline
   order above. The configuration is fully specified (kernel, features, per-source noise, seed),
   so retraining reproduces the models up to floating-point variation.

---

## Results

- The kernel-strategy benchmark in `FeSiALCrMsModel.py` identifies the best-performing surrogate
  among six candidates (Linear+Matérn(ARD), Matérn(ARD), Linear+RQ, RQ, Linear+RBF, RBF).
- Four active-learning rounds were executed, recommending three alloys each. Subsequent rounds
  progressively reduce the predictive uncertainty across the design space (visualised by
  `1-3GPRtuijian.py`, which compares the V3 and V4 uncertainty fields).
- The final EI–density Pareto front from `Ms-P pareto.py` provides the recommended compositions
  that balance high predicted `Ms` against the density objective.

The compositions and measured `Ms` of the recommended alloys are provided in
`1-FeSiAlCr.xlsx`, `2-FeSiAlCr.xlsx` and `3-FeSiAlCr.xlsx`.

> **Not included in this repository**: the figure-generating scripts and rendered figures
> (`KMeans_9alloys_schematic.py`, `mark_recommended_points.py`, and the associated PDF/PNG outputs).
> Two further validation experiments and the corresponding model (`ms_gpr_matern05_model_v5.pkl`)
> are held back pending publication. The released pipeline is complete and reproducible up to the
> final model **V4**, which is the model used by `Ms-P pareto.py`; the withheld material is
> available from the authors on reasonable request.

---

## Notes and limitations

- The `.pkl` models are binary artefacts and are environment-sensitive (see above). For long-term
  reproducibility, prefer retraining from the published configuration.
- `Ms-P pareto.py` contains a space in its filename; it is kept for consistency with the analysis
  history and must be quoted in shell commands.
- The scripts are research code: they print diagnostics and save figures to the working directory
  rather than exposing a packaged CLI.

---

## 中文说明

本仓库为 **Fe–Si–Al–Cr 软磁合金高饱和磁化强度（Ms）的机器学习设计代码**，方法为
GPR 代理模型 + 贝叶斯优化 + 主动学习，并经实验验证。

**执行顺序**（这是一条链，上一步的模型喂给下一步，顺序不能乱）：

```
① 1-GPRqueding.py     训练初始模型 V1
② 1-0GPRtuijian.py    第 1 轮推荐
③ 1-1GPRtuijian.py    第 2 轮（加入 1-FeSiAlCr.xlsx 实测数据）→ V2
④ 1-2GPRtuijian.py    第 3 轮（加入 2-FeSiAlCr.xlsx）→ V3
⑤ 1-3GPRtuijian.py    第 4 轮（加入 3-FeSiAlCr.xlsx）→ V4
⑥ Ms-P pareto.py      加载 V4，输出 EI–密度帕累托前沿推荐
```

`FeSiALCrMsModel.py`、`0-BOxunyou.py`、`1-GPRxunyou.py`、`FeSiAlCrtuijian.py`
是独立分析脚本，只依赖 `combined_training_data.csv`。

**数据来源标记（`source` 列）**：`0` = 理论/计算数据（51 条，噪声权重 225）、
`1` = 文献实验数据（68 条，噪声权重 49）、`2` = 本工作自有实验数据（噪声权重 9）。
权重越小代表越可信，因此自有实验数据主导模型拟合。

**模型配置**：核函数 `ConstantKernel(100.0) × Matern(length_scale=10.0, nu=0.5)`，
特征为成分 + 价电子浓度 `vec` + 原子半径差 `atom`，先做 `StandardScaler` 标准化，
随机种子固定为 42。`.pkl` 是字典格式 `{'scaler', 'model', 'features'}`，
**scaler 与模型必须一起保存**，否则新配方无法正确缩放。

⚠️ 仓库中的 `.pkl` 为旧版 numpy/scikit-learn 序列化，换环境可能无法加载，
建议直接重跑 `1-GPRqueding.py` 重新生成。

**关于模型版本**：仓库发布了 **V1–V4** 四个模型，链条终点是 V4（`Ms-P pareto.py` 加载的就是
V4）。V5 是在 V4 基础上加入两组验证实验数据后训练的，因该部分数据尚未发表，
**暂未公开**（生成 V5 的 `Fe79Fe81_validation.py` 脚本亦未包含）。

---

## License

Released under the MIT License — see [LICENSE](LICENSE).
