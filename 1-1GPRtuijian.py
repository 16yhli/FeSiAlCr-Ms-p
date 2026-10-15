import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

# --- [1. 基础物理特性配置] ---
ELEMENT_DATA = pd.DataFrame({
    'Fe':[8, 1.241, 55.845],
    'Si':[4, 1.153, 28.0855],
    'Al':[3, 1.432, 26.9815],
    'Cr':[6, 1.249, 51.9961]
}, index=['valence', 'radius', 'weight']).T

def compute_physics_features(df):
    """计算原子尺寸差(atom)和价电子浓度(vec)"""
    elements =['Fe', 'Si', 'Al', 'Cr']
    at_cols =[f'{e}_at' for e in elements]
    fractions = df[at_cols].values / 100.0
    
    df['vec'] = np.dot(fractions, ELEMENT_DATA['valence'].values)
    
    radius_values = ELEMENT_DATA['radius'].values
    presence_mask = (df[at_cols].values > 0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        row_radius = np.where(presence_mask, radius_values, np.nan)
        df['atom'] = (np.nanmax(row_radius, axis=1) / np.nanmin(row_radius, axis=1) - 1) * 100
        
    df['atom'] = df['atom'].fillna(0)
    return df

# ---[2. 顶级学术期刊风格绘图函数] ---
# 修改：分离 model 和 scaler，以适应 .pkl 包的调用方式
def plot_academic_sigma_analysis(df_pool, df_old, df_new_exp, model_new, scaler_new, mpsd_improvement):
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(9, 6), dpi=120) 
    
    color_bg_old, color_bg_new = "#A8A8A8", "#5CB5F0"
    color_src0, color_src1, color_src2, color_new = '#E74C3C', '#27AE60', '#1F618D', '#F1C40F'

    # 1. 绘制背景云团 (初始三角，更新圆点)
    ax.scatter(df_pool['predicted_Ms'], df_pool['old_std'], color=color_bg_old, 
               alpha=0.15, s=9, marker='^', label=f'Initial Field (Avg $\sigma$: {np.mean(df_pool["old_std"]):.1f})', zorder=1)
    ax.scatter(df_pool['predicted_Ms'], df_pool['updated_std'], color=color_bg_new, 
               alpha=0.35, s=9, marker='o', label=f'Updated Field (Avg $\sigma$: {np.mean(df_pool["updated_std"]):.1f})', zorder=2)

    # 2. 预测训练点的真实 Sigma
    feature_cols =['Fe_at', 'Si_at', 'Al_at', 'Cr_at', 'atom', 'vec']
    def get_real_sigma(df_subset):
        X_scaled = scaler_new.transform(df_subset[feature_cols])
        _, sig = model_new.predict(X_scaled, return_std=True)
        return sig

    # 绘制实测点并标记其真实 Sigma 数值在图例中
    if not (s0 := df_old[df_old['source'] == 0]).empty:
        s0_sig = get_real_sigma(s0)
        ax.scatter(s0['Ms'], s0_sig, color=color_src0, marker='o', s=45, alpha=0.6, edgecolors='k', label=f'Theory ($\sigma_{{avg}}$: {np.mean(s0_sig):.1f})', zorder=10)
    if not (s1 := df_old[df_old['source'] == 1]).empty:
        s1_sig = get_real_sigma(s1)
        ax.scatter(s1['Ms'], s1_sig, color=color_src1, marker='s', s=45, alpha=0.7, edgecolors='k', label=f'Literature ($\sigma_{{avg}}$: {np.mean(s1_sig):.1f})', zorder=11)
    if not (s2 := df_old[df_old['source'] == 2]).empty:
        s2_sig = get_real_sigma(s2)
        ax.scatter(s2['Ms'], s2_sig, color=color_src2, marker='^', s=65, alpha=0.9, edgecolors='k', label=f'Prior Exp. ($\sigma_{{avg}}$: {np.mean(s2_sig):.1f})', zorder=12)
    if not df_new_exp.empty:
        sn_sig = get_real_sigma(df_new_exp)
        ax.scatter(df_new_exp['Ms'], sn_sig, color=color_new, marker='*', s=180, alpha=1.0, edgecolors='k', label=f'Exp1 ($\sigma_{{avg}}$: {np.mean(sn_sig):.1f})', zorder=13)

    # --- [全封闭坐标轴设置] ---
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)

    ax.tick_params(axis='both', which='both', direction='in', labelsize=16, 
                   bottom=True, left=True, top=False, right=False, width=1.5, length=6)
    
    ax.set_xlabel('Magnetization $M_s$ (emu/g)', fontsize=20, fontweight='bold', labelpad=12)
    ax.set_ylabel('Uncertainty $\sigma$ (emu/g)', fontsize=20, fontweight='bold', labelpad=12)
    
    y_limit = max(np.percentile(df_pool['old_std'], 95), 20)
    ax.set_ylim(0, y_limit)
    
    ax.text(0.05, 0.90, f'MPSD Reduced: {mpsd_improvement:.1f}%', transform=ax.transAxes, 
            fontsize=16, fontweight='bold', color='#1A5276', bbox=dict(boxstyle='round', fc='white', ec='#1A5276', alpha=0.2))
    
    ax.legend(loc='upper right', fontsize=16, frameon=False)
    
    plt.tight_layout()
    return fig

# --- [3. 核心主程序] ---
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir, "combined_training_data.csv")
    excel_path = os.path.join(current_dir, "1-FeSiAlCr.xlsx")
    
    # 模型路径
    old_model_path = os.path.join(current_dir, "ms_gpr_matern05_model.pkl")
    new_model_path = os.path.join(current_dir, "ms_gpr_matern05_model_v2.pkl") # 另存为V2版本，保留迭代历史

    if not os.path.exists(train_path) or not os.path.exists(excel_path) or not os.path.exists(old_model_path):
        print("错误：训练数据、新实验数据或上代模型文件 (.pkl) 缺失，请检查当前目录。")
        return

    # 1. 加载旧数据与旧模型（用于评估 MPSD 降低幅度）
    print("正在加载第一代模型评估旧不确定度...")
    old_package = joblib.load(old_model_path)
    scaler_old = old_package['scaler']
    model_old = old_package['model']
    feature_cols = old_package['features']

    # 2. 读取并合并数据
    df_old = pd.read_csv(train_path)
    df_new_exp = pd.read_excel(excel_path)
    df_new_exp['source'] = 2  # 新测结果赋予最高可靠性标识
    df_new_exp = compute_physics_features(df_new_exp)
    
    df_combined = pd.concat([df_old, df_new_exp], ignore_index=True)

    # 3. 训练进化版的新模型
    print("正在将最新实验数据注入模型，重新训练第二代 Matern (v=0.5) 模型...")
    scaler_new = StandardScaler()
    X_combined_scaled = scaler_new.fit_transform(df_combined[feature_cols])
    y_combined = df_combined['Ms'].values
    
    # 按照设定应用异方差权重
    alpha_weights_new = df_combined['source'].map({0: 225.0, 1: 49.0, 2: 9.0}).values
    
    kernel_new = C(100.0) * Matern(length_scale=10.0, nu=0.5)
    model_new = GaussianProcessRegressor(
        kernel=kernel_new, 
        alpha=alpha_weights_new, 
        normalize_y=False, 
        n_restarts_optimizer=10, 
        random_state=42
    )
    model_new.fit(X_combined_scaled, y_combined)
    
    # 4. 保存新代模型包
    new_package = {
        'scaler': scaler_new,
        'model': model_new,
        'features': feature_cols
    }
    joblib.dump(new_package, new_model_path)
    print(f"✅ 模型已成功进化并保存为: {new_model_path}\n")

    # --- 开始成分空间扫描与推荐 ---
    FE_MIN_LIMIT = 70.0
    MS_THRESHOLD = 100.0
    s_min, s_max = max(1.0, df_old['Si_at'].min()), df_old['Si_at'].max()
    a_min, a_max = max(1.0, df_old['Al_at'].min()), df_old['Al_at'].max()
    c_min, c_max = max(1.0, df_old['Cr_at'].min()), df_old['Cr_at'].max()

    # 1. 绘图池 (步长2.0降低计算量)
    candidates_plot =[]
    for si in np.arange(0, s_max + 2, 2.0):
        for al in np.arange(0, a_max + 2, 2.0):
            for cr in np.arange(0, c_max + 2, 2.0):
                fe = 100.0 - (si + al + cr)
                if 60 <= fe <= 100: candidates_plot.append([fe, si, al, cr])
    df_plot_pool = pd.DataFrame(candidates_plot, columns=['Fe_at', 'Si_at', 'Al_at', 'Cr_at'])
    df_plot_pool = compute_physics_features(df_plot_pool)

    # 2. 推荐池 (细致步长1.0)
    candidates_rec =[]
    for si in np.arange(s_min, s_max + 1, 1.0): 
        for al in np.arange(a_min, a_max + 1, 1.0): 
            for cr in np.arange(c_min, c_max + 1, 1.0):
                fe = 100.0 - (si + al + cr)
                if fe >= FE_MIN_LIMIT: candidates_rec.append([fe, si, al, cr])
    df_rec_pool = pd.DataFrame(candidates_rec, columns=['Fe_at', 'Si_at', 'Al_at', 'Cr_at'])
    df_rec_pool = compute_physics_features(df_rec_pool)

    # 计算预测均值和方差
    X_plot_old = scaler_old.transform(df_plot_pool[feature_cols])
    X_plot_new = scaler_new.transform(df_plot_pool[feature_cols])
    
    _, sigma_old = model_old.predict(X_plot_old, return_std=True)
    mu_new, sigma_new = model_new.predict(X_plot_new, return_std=True)
    
    # 物理下限强制控制
    mu_new = np.maximum(mu_new, 0.0)

    df_plot_pool['old_std'] = sigma_old
    df_plot_pool['updated_std'] = sigma_new
    df_plot_pool['predicted_Ms'] = mu_new
    
    improvement = (np.mean(sigma_old) - np.mean(sigma_new)) / np.mean(sigma_old) * 100

    # 绘图输出
    fig = plot_academic_sigma_analysis(df_plot_pool, df_old, df_new_exp, model_new, scaler_new, improvement)
    plt.show()

    # 第二轮实验推荐详情计算
    target_candidates = [[73.0, 25.0, 1.0, 1.0],[96.0, 2.0, 1.0, 1.0],[94.0, 4.0, 1.0, 1.0]    ]
    df_target = pd.DataFrame(target_candidates, columns=['Fe_at', 'Si_at', 'Al_at', 'Cr_at'])
    
    # 1. 补全物理特征以满足模型输入要求
    df_target = compute_physics_features(df_target)
    
    # 2. 提取特征并使用 V2 模型的 scaler_new 进行标准化
    X_target_new = scaler_new.transform(df_target[feature_cols])
    
    # 3. 预测 Ms 与不确定性 (使用 model_new)
    mu_target, sigma_target = model_new.predict(X_target_new, return_std=True)
    df_target['predicted_Ms'] = np.maximum(mu_target, 0.0) # 物理常识保护
    df_target['updated_std'] = sigma_target

    print("\n" + "="*80)
    print(f"【推荐成分预测及不确定性评估 (基于 V2 模型)】")
    print(" 以下为指定的 3 个配方的预测结果：")
    print("="*80)
    
    # 4. 打印结果（自动换算 wt.%）
    for i, (idx, row) in enumerate(df_target.iterrows(), 1):
        # 换算 wt%
        at_vals = row[['Fe_at', 'Si_at', 'Al_at', 'Cr_at']].values
        w_mult = ELEMENT_DATA.loc[['Fe', 'Si', 'Al', 'Cr'], 'weight'].values
        masses = at_vals * w_mult
        wt_pcts = masses / masses.sum() * 100
        
        print(f"推荐配方 {i} [\u03C3={row['updated_std']:.2f}]: Fe={row['Fe_at']:.1f}, Si={row['Si_at']:.1f}, Al={row['Al_at']:.1f}, Cr={row['Cr_at']:.1f} (at.%)")
        print(f"             Fe={wt_pcts[0]:.1f}, Si={wt_pcts[1]:.1f}, Al={wt_pcts[2]:.1f}, Cr={wt_pcts[3]:.1f} (wt.%) | 预测 Ms={row['predicted_Ms']:.1f}")
        print("-" * 75)

if __name__ == "__main__":
    main()