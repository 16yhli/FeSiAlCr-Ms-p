import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import optuna
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, RBF, ConstantKernel as C, DotProduct
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.decomposition import PCA

# --- [0. 全局设置] ---
SET_SEED = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 16

set_seed(SET_SEED)

# --- [1. 数据加载] ---
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "combined_training_data.csv")
df = pd.read_csv(data_path)
feature_cols = ['Fe_at', 'Si_at', 'Al_at', 'Cr_at', 'atom', 'vec']
X, y = df[feature_cols], df['Ms']
source = df['source']
alpha_weights = source.map({0: 225.0, 1: 49.0, 2: 9.0}).values

# --- [2. 实验核心逻辑] ---
def get_kernel(mode, p):
    c_amp = p['c_amplitude']
    if 'mat' in mode or 'rbf' in mode:
        ls_ard = np.array([p[f'ls_{i}'] for i in range(len(feature_cols))])
        base_kernel = C(c_amp) * (Matern(length_scale=ls_ard, nu=2.5) if 'mat' in mode else RBF(length_scale=ls_ard))
    elif 'rq' in mode:
        base_kernel = C(c_amp) * RationalQuadratic(length_scale=p['ls_single'], alpha=p['alpha_rq'])
    
    if 'lin' in mode:
        return C(c_amp) * DotProduct(sigma_0=p['sigma0']) + base_kernel
    else:
        return base_kernel

def run_comparison(mode):
    def objective(trial):
        p = {
            'c_amplitude': trial.suggest_float('c_amplitude', 0.1, 100, log=True),
            'noise_level': trial.suggest_float('noise_level', 0.001, 0.5, log=True)
        }
        if 'mat' in mode or 'rbf' in mode:
            for i in range(len(feature_cols)):
                p[f'ls_{i}'] = trial.suggest_float(f'ls_{i}', 0.5, 50.0, log=True)
        elif 'rq' in mode:
            p['ls_single'] = trial.suggest_float('ls_single', 0.5, 50.0, log=True)
            p['alpha_rq'] = trial.suggest_float('alpha_rq', 0.01, 100.0, log=True)
        if 'lin' in mode:
            p['sigma0'] = trial.suggest_float('sigma0', 0.1, 100.0, log=True)

        cv = KFold(n_splits=5, shuffle=True, random_state=SET_SEED)
        scores = []
        for train_idx, test_idx in cv.split(X):
            current_alpha = alpha_weights[train_idx] * p['noise_level']
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X.iloc[train_idx])
            X_test_scaled = scaler.transform(X.iloc[test_idx])
            gpr = GaussianProcessRegressor(kernel=get_kernel(mode, p), alpha=current_alpha, 
                                           optimizer=None, normalize_y=True, random_state=SET_SEED)
            gpr.fit(X_train_scaled, y.iloc[train_idx])
            scores.append(r2_score(y.iloc[test_idx], gpr.predict(X_test_scaled)))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SET_SEED))
    study.optimize(objective, n_trials=60)
    return study

# --- [3. 执行实验] ---
modes = ['lin_mat', 'mat_only', 'lin_rq', 'rq_only', 'lin_rbf', 'rbf_only']
titles = ['Linear + Matern (ARD)', 'Matern Only (ARD)', 'Linear + RQ (Iso)', 'RQ Only (Iso)', 'Linear + RBF (ARD)', 'RBF Only (ARD)']
studies = {mode: run_comparison(mode) for mode, title in zip(modes, titles)}

# --- [4. 绘图与结果展示] ---
fig, axes = plt.subplots(2, 3, figsize=(22, 12), facecolor='white')
axes = axes.flatten()
summary = []

for ax, mode, title in zip(axes, modes, titles):
    study = studies[mode]
    bp = study.best_params
    cv = KFold(n_splits=5, shuffle=True, random_state=SET_SEED)
    all_preds, all_true = [], []
    for train_idx, test_idx in cv.split(X):
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X.iloc[train_idx])
        X_test_s = scaler.transform(X.iloc[test_idx])
        gpr = GaussianProcessRegressor(kernel=get_kernel(mode, bp), alpha=alpha_weights[train_idx]*bp['noise_level'], 
                                       optimizer=None, normalize_y=True, random_state=SET_SEED)
        gpr.fit(X_train_s, y.iloc[train_idx])
        all_preds.extend(gpr.predict(X_test_s))
        all_true.extend(y.iloc[test_idx])
    
    res_r2, res_mae = r2_score(all_true, all_preds), mean_absolute_error(all_true, all_preds)
    summary.append((title, mode, res_r2, res_mae))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=1, random_state=SET_SEED)
    X_pca = pca.fit_transform(X_scaled)
    final_gpr = GaussianProcessRegressor(kernel=get_kernel(mode, bp), alpha=alpha_weights*bp['noise_level'], 
                                          optimizer=None, normalize_y=True, random_state=SET_SEED)
    final_gpr.fit(X_scaled, y)
    x_grid_pca = np.linspace(X_pca.min()-1.2, X_pca.max()+1.2, 500).reshape(-1, 1)
    X_grid_raw = scaler.inverse_transform(pca.inverse_transform(x_grid_pca))
    y_mean, y_std = final_gpr.predict(scaler.transform(X_grid_raw), return_std=True)

    ax.fill_between(x_grid_pca.flatten(), y_mean - 1.96*y_std, y_mean + 1.96*y_std, color='gray', alpha=0.15)
    ax.plot(x_grid_pca.flatten(), y_mean, color='#1b4965', lw=2.5)
    colors = {0: 'red', 1: 'green', 2: 'blue'}
    for s_val in [0, 1, 2]:
        mask = (source == s_val)
        ax.scatter(X_pca[mask], y[mask], c=colors[s_val], s=35, edgecolors='k', alpha=0.6, zorder=5)
    ax.set_title(f"{title}\n$R^2$: {res_r2:.3f} | MAE: {res_mae:.2f}")

plt.tight_layout()
plt.show()

# --- [5. 最终总结与参数输出] ---
print("\n" + "="*80)
print(f"{'Kernel Strategy':<25} | {'R2':<8} | {'MAE':<8}")
print("-" * 80)
summary.sort(key=lambda x: x[2], reverse=True)
for t, m, r, mae in summary:
    print(f"{t:<25} | {r:<8.4f} | {mae:<8.3f}")

print("\n" + "="*80)
print(" 各策略最佳参数组合详情:")
print("="*80)
for t, m, r, mae in summary:
    print(f"\n>>> [{t}]")
    best_params = studies[m].best_params
    for param_name, param_value in best_params.items():
        if isinstance(param_value, float):
            print(f"  - {param_name:15}: {param_value:.6f}")
        else:
            print(f"  - {param_name:15}: {param_value}")
print("="*80)