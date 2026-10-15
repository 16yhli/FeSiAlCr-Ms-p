import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C, DotProduct
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# --- [0. 加载数据] ---
# 动态获取当前脚本所在目录，并读取 combined_training_data.csv
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "combined_training_data.csv")
df = pd.read_csv(data_path)

# 提取特征、目标值和数据来源标识
X = df[['Fe_at', 'Si_at', 'Al_at', 'Cr_at', 'atom', 'vec']]
y = df['Ms']
source = df['source']

# 按照您的要求映射可靠性权重 (注意: 这里对应的是高斯过程的观测噪声方差)
alpha_weights = source.map({0: 225.0, 1: 49.0, 2: 9.0}).values
SET_SEED = 42

# --- [1. 定义 6 种不同的核函数组合] ---
# C() 代表常数核，用于拟合目标量 Ms 的尺度（方差）
kernels_dict = {
    "RBF": C(100.0) * RBF(length_scale=10.0),
    "Matern (\u03BD=0.5)": C(100.0) * Matern(length_scale=10.0, nu=0.5), # 允许突变 (适合相变区)
    "Matern (\u03BD=1.5)": C(100.0) * Matern(length_scale=10.0, nu=1.5), # 中等平滑度 (推荐)
    "Matern (\u03BD=2.5)": C(100.0) * Matern(length_scale=10.0, nu=2.5), # 较平滑
    "Rational Quadratic": C(100.0) * RationalQuadratic(length_scale=1.0, alpha=0.1), # 多尺度特征混合
    "DotProduct + Matern": DotProduct(sigma_0=1.0) + C(100.0) * Matern(length_scale=10.0, nu=1.5) # 引入线性趋势
}

titles = list(kernels_dict.keys())
modes = list(kernels_dict.values())

# --- [2. & 3. 交叉验证评估与全量训练绘图] ---
fig, axes = plt.subplots(2, 3, figsize=(22, 12), facecolor='white')
axes = axes.flatten()
summary =[]

print("正在执行 5 折交叉验证并训练最终模型，请稍候...")

for ax, kernel, title in zip(axes, modes, titles):
    cv = KFold(n_splits=5, shuffle=True, random_state=SET_SEED)
    all_preds, all_true = [],[]
    
    # -------- 5折交叉验证计算严谨的泛化性能 --------
    for train_idx, test_idx in cv.split(X):
        scaler = StandardScaler()
        # 严格防泄露：仅在训练集上 fit
        X_train_s = scaler.fit_transform(X.iloc[train_idx])
        X_test_s = scaler.transform(X.iloc[test_idx])
        
        gpr_cv = GaussianProcessRegressor(
            kernel=kernel, 
            alpha=alpha_weights[train_idx], # 赋予对应的权重
            normalize_y=False,              # 必须为False，保持 Ms 的真实尺度以匹配 225/49 的权重
            n_restarts_optimizer=5,         # 增加重启次数避免局部最优
            random_state=SET_SEED
        )
        gpr_cv.fit(X_train_s, y.iloc[train_idx])
        all_preds.extend(gpr_cv.predict(X_test_s))
        all_true.extend(y.iloc[test_idx])
    
    res_r2 = r2_score(all_true, all_preds)
    res_mae = mean_absolute_error(all_true, all_preds)

    # -------- 使用全量数据训练最终模型以绘制 PCA 面板 --------
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    
    # 提取第1主成分用于1D展示
    pca = PCA(n_components=1, random_state=SET_SEED)
    X_pca = pca.fit_transform(X_scaled)
    
    final_gpr = GaussianProcessRegressor(
        kernel=kernel, 
        alpha=alpha_weights, 
        normalize_y=False, 
        n_restarts_optimizer=10, 
        random_state=SET_SEED
    )
    final_gpr.fit(X_scaled, y)
    
    # 记录该核函数的最优结果及拟合出的超参数
    summary.append((title, res_r2, res_mae, final_gpr.kernel_))

    # -------- 构建绘图用的连续 PCA 网格 --------
    x_grid_pca = np.linspace(X_pca.min() - 1.2, X_pca.max() + 1.2, 500).reshape(-1, 1)
    # 将 1D 的 PCA 网格反投影回 6D 的标准化特征空间，以便模型预测
    X_grid_scaled = pca.inverse_transform(x_grid_pca)
    
    y_mean, y_std = final_gpr.predict(X_grid_scaled, return_std=True)

    # -------- 绘图逻辑 --------
    ax.fill_between(x_grid_pca.flatten(), y_mean - 1.96*y_std, y_mean + 1.96*y_std, color='gray', alpha=0.15, label='95% CI')
    ax.plot(x_grid_pca.flatten(), y_mean, color='#1b4965', lw=2.5, label='GPR Mean')
    
    colors = {0: '#d62828', 1: '#2a9d8f', 2: '#003049'} # 采用高级且对比较强烈的学术配色
    labels = {0: 'Source 0 (Theory)', 1: 'Source 1 (Literature)', 2: 'Source 2'}
    
    for s_val in [0, 1, 2]:
        mask = (source == s_val)
        if mask.sum() > 0: # 数据集中存在该source才画
            ax.scatter(X_pca[mask], y[mask], c=colors[s_val], s=45, edgecolors='white', linewidths=0.5, alpha=0.8, zorder=5, label=labels[s_val])
            
    ax.set_title(f"{title}\n$R^2$: {res_r2:.3f} | MAE: {res_mae:.2f}", fontsize=14, pad=10)
    ax.set_xlabel("Principal Component 1", fontsize=11)
    ax.set_ylabel("Saturation Magnetization (Ms)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    if ax == axes[0]: # 仅在第一个子图显示图例保持整洁
        ax.legend(loc='best', fontsize=9)

plt.tight_layout()
plt.show()

# ---[4. 最终总结与参数输出] ---
print("\n" + "="*80)
print(f"{'Kernel Strategy':<25} | {'R2':<8} | {'MAE':<8}")
print("-" * 80)
# 按 R2 降序排序
summary.sort(key=lambda x: x[1], reverse=True)
for t, r, mae, _ in summary:
    print(f"{t:<25} | {r:<8.4f} | {mae:<8.3f}")

print("\n" + "="*80)
print(" 各策略拟合后的最佳超参数组合详情 (Optima):")
print("="*80)
for t, r, mae, fitted_kernel in summary:
    print(f"\n>>> [{t}]")
    # get_params() 会将优化好的 length_scale, constant_value 等打印出来
    best_params = fitted_kernel.get_params()
    for param_name, param_value in best_params.items():
        # 过滤掉不需要展示的 bounds 参数，保持界面清爽
        if "bound" not in param_name and param_name != "kernel":
            if isinstance(param_value, float):
                print(f"  - {param_name:25}: {param_value:.4f}")
            else:
                print(f"  - {param_name:25}: {param_value}")
print("="*80)