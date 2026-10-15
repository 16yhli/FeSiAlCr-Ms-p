import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
def plot_academic_sigma_analysis(df_pool, df_csv, df_exp1, df_exp2, df_exp3, model_new, scaler_new, mpsd_improvement):
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=120) 
    
    color_bg_old, color_bg_new = "#A8A8A8", "#5CB5F0"
    color_src0, color_src1 = '#E74C3C', '#27AE60'  # 理论与文献
    
    # 【核心修改】：保持视觉一致性！Exp1黄星，Exp2蓝三角，新增 Exp3 为紫色菱形
    color_exp1, color_exp2, color_exp3 = '#F1C40F', '#1F618D', '#9B59B6'  

    # 1. 绘制背景云团 (初始三角代表V3，更新圆点代表V4)
    ax.scatter(df_pool['predicted_Ms'], df_pool['old_std'], color=color_bg_old, 
               alpha=0.15, s=9, marker='^', label=f'V3 Field (Avg $\sigma$: {np.mean(df_pool["old_std"]):.1f})', zorder=1)
    ax.scatter(df_pool['predicted_Ms'], df_pool['updated_std'], color=color_bg_new, 
               alpha=0.35, s=9, marker='o', label=f'V4 Field (Avg $\sigma$: {np.mean(df_pool["updated_std"]):.1f})', zorder=2)

    # 2. 预测训练点的真实 Sigma
    feature_cols =['Fe_at', 'Si_at', 'Al_at', 'Cr_at', 'atom', 'vec']
    def get_real_sigma(df_subset):
        X_scaled = scaler_new.transform(df_subset[feature_cols])
        _, sig = model_new.predict(X_scaled, return_std=True)
        return sig

    # 绘制基础数据(理论与文献)
    if not (s0 := df_csv[df_csv['source'] == 0]).empty:
        s0_sig = get_real_sigma(s0)
        ax.scatter(s0['Ms'], s0_sig, color=color_src0, marker='o', s=45, alpha=0.5, edgecolors='k', label=f'Theory ($\sigma_{{avg}}$: {np.mean(s0_sig):.1f})', zorder=10)
    if not (s1 := df_csv[df_csv['source'] == 1]).empty:
        s1_sig = get_real_sigma(s1)
        ax.scatter(s1['Ms'], s1_sig, color=color_src1, marker='s', s=45, alpha=0.6, edgecolors='k', label=f'Literature ($\sigma_{{avg}}$: {np.mean(s1_sig):.1f})', zorder=11)
        
    # 绘制第一批实验点 (Exp1)
    if not df_exp1.empty:
        e1_sig = get_real_sigma(df_exp1)
        ax.scatter(df_exp1['Ms'], e1_sig, color=color_exp1, marker='*', s=220, alpha=0.9, edgecolors='k', label=f'Exp 1 ($\sigma_{{avg}}$: {np.mean(e1_sig):.1f})', zorder=12)
        
    # 绘制第二批新实验点 (Exp2)
    if not df_exp2.empty:
        e2_sig = get_real_sigma(df_exp2)
        ax.scatter(df_exp2['Ms'], e2_sig, color=color_exp2, marker='^', s=150, alpha=1.0, edgecolors='k', label=f'Exp 2 ($\sigma_{{avg}}$: {np.mean(e2_sig):.1f})', zorder=13)

    # 【新增】：绘制第三批新实验点 (Exp3) - 紫色菱形 (D)
    if not df_exp3.empty:
        e3_sig = get_real_sigma(df_exp3)
        ax.scatter(df_exp3['Ms'], e3_sig, color=color_exp3, marker='D', s=120, alpha=1.0, edgecolors='k', label=f'Exp 3 ($\sigma_{{avg}}$: {np.mean(e3_sig):.1f})', zorder=14)

    # ---[全封闭坐标轴设置] ---
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
    
    # 动态适应 Y 轴高度
    y_limit = max(np.percentile(df_pool['old_std'], 95), 12)
    ax.set_ylim(0, y_limit)
    
    ax.text(0.05, 0.90, f'MPSD Reduced: {mpsd_improvement:.1f}%', transform=ax.transAxes, 
            fontsize=16, fontweight='bold', color='#1A5276', bbox=dict(boxstyle='round', fc='white', ec='#1A5276', alpha=0.2))
    
    ax.legend(loc='upper right', fontsize=12, frameon=False, ncol=2) # 调整为两列以免图例过长遮挡数据
    
    plt.tight_layout()
    return fig

# --- [3. 核心主程序] ---
def main():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd() 
        
    # 数据文件路径
    train_path = os.path.join(current_dir, "combined_training_data.csv")
    exp1_path = os.path.join(current_dir, "1-FeSiAlCr.xlsx")
    exp2_path = os.path.join(current_dir, "2-FeSiAlCr.xlsx")
    exp3_path = os.path.join(current_dir, "3-FeSiAlCr.xlsx") # 新增第三批数据
    
    # 模型路径
    old_model_path = os.path.join(current_dir, "ms_gpr_matern05_model_v3.pkl")
    new_model_path = os.path.join(current_dir, "ms_gpr_matern05_model_v4.pkl") # 演化至 V4

    # 自动诊断文件缺失
    files_to_check =[train_path, exp1_path, exp2_path, exp3_path, old_model_path]
    missing =[os.path.basename(p) for p in files_to_check if not os.path.exists(p)]
    if missing:
        print("="*60)
        print(f"❌ 文件未找到！Python 当前工作目录是：\n{current_dir}")
        print(f"缺失以下文件: {', '.join(missing)}")
        print("💡 请检查文件是否放在了正确目录，或者名称是否拼写无误。")
        print("="*60)
        return

    # 1. 加载 V3 模型评估背景基准
    print("正在加载 V3 模型用于评估背景基准...")
    old_package = joblib.load(old_model_path)
    scaler_old = old_package['scaler']
    model_old = old_package['model']
    feature_cols = old_package['features']

    # 2. 读取并合并所有数据
    df_csv = pd.read_csv(train_path)
    
    df_exp1 = pd.read_excel(exp1_path)
    df_exp1['source'] = 2  
    df_exp1 = compute_physics_features(df_exp1)
    
    df_exp2 = pd.read_excel(exp2_path)
    df_exp2['source'] = 2  
    df_exp2 = compute_physics_features(df_exp2)
    
    df_exp3 = pd.read_excel(exp3_path)
    df_exp3['source'] = 2  
    df_exp3 = compute_physics_features(df_exp3)
    
    # df_old 是 V3 模型当年训练用的数据组合 (csv + exp1 + exp2)
    df_old = pd.concat([df_csv, df_exp1, df_exp2], ignore_index=True)
    # df_combined 是将用来训练 V4 模型的全部数据 (csv + exp1 + exp2 + exp3)
    df_combined = pd.concat([df_old, df_exp3], ignore_index=True)

    # 3. 训练进化版的新模型 (V4)
    print("正在将第三批最新实验数据 (Exp3) 注入，训练第四代 Matern (v=0.5) 模型...")
    scaler_new = StandardScaler()
    X_combined_scaled = scaler_new.fit_transform(df_combined[feature_cols])
    y_combined = df_combined['Ms'].values
    
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
    
    # 4. 保存 V4 模型包
    new_package = {
        'scaler': scaler_new,
        'model': model_new,
        'features': feature_cols
    }
    joblib.dump(new_package, new_model_path)
    print(f"✅ 模型已成功进化并保存为: {os.path.basename(new_model_path)}\n")

    # --- 开始成分空间扫描与推荐 ---
    FE_MIN_LIMIT = 70.0
    MS_THRESHOLD = 100.0
    s_min, s_max = max(1.0, df_csv['Si_at'].min()), df_csv['Si_at'].max()
    a_min, a_max = max(1.0, df_csv['Al_at'].min()), df_csv['Al_at'].max()
    c_min, c_max = max(1.0, df_csv['Cr_at'].min()), df_csv['Cr_at'].max()

    # 1. 绘图池 (步长2.0保持较低计算量以快速出图)
    candidates_plot =[]
    for si in np.arange(0, s_max + 2, 2.0):
        for al in np.arange(0, a_max + 2, 2.0):
            for cr in np.arange(0, c_max + 2, 2.0):
                fe = 100.0 - (si + al + cr)
                if 60 <= fe <= 100: candidates_plot.append([fe, si, al, cr])
    df_plot_pool = pd.DataFrame(candidates_plot, columns=['Fe_at', 'Si_at', 'Al_at', 'Cr_at'])
    df_plot_pool = compute_physics_features(df_plot_pool)

    # 2. 推荐池 (精细步长 1)
    print("正在执行高精度全域网格扫描 (步长: 1)...")
    candidates_rec =[]
    for si in np.arange(s_min, s_max + 0.1, 1.0): 
        for al in np.arange(a_min, a_max + 0.1, 1.0): 
            for cr in np.arange(c_min, c_max + 0.1, 1.0):
                fe = 100.0 - (si + al + cr)
                if fe >= FE_MIN_LIMIT: candidates_rec.append([fe, si, al, cr])
    df_rec_pool = pd.DataFrame(candidates_rec, columns=['Fe_at', 'Si_at', 'Al_at', 'Cr_at'])
    df_rec_pool = compute_physics_features(df_rec_pool)

    # 计算绘图池的不确定性
    X_plot_old = scaler_old.transform(df_plot_pool[feature_cols])
    X_plot_new = scaler_new.transform(df_plot_pool[feature_cols])
    
    _, sigma_old = model_old.predict(X_plot_old, return_std=True)
    mu_new, sigma_new = model_new.predict(X_plot_new, return_std=True)
    mu_new = np.maximum(mu_new, 0.0)

    df_plot_pool['old_std'] = sigma_old
    df_plot_pool['updated_std'] = sigma_new
    df_plot_pool['predicted_Ms'] = mu_new
    
    improvement = (np.mean(sigma_old) - np.mean(sigma_new)) / np.mean(sigma_old) * 100

    # 输出绘图
    fig = plot_academic_sigma_analysis(df_plot_pool, df_csv, df_exp1, df_exp2, df_exp3, model_new, scaler_new, improvement)
    plt.show()

    # 第四轮实验推荐详情计算
    X_rec_new = scaler_new.transform(df_rec_pool[feature_cols])
    mu_rec, sigma_rec = model_new.predict(X_rec_new, return_std=True)
    mu_rec = np.maximum(mu_rec, 0.0) 
    
    df_rec_pool['updated_std'] = sigma_rec
    df_rec_pool['predicted_Ms'] = mu_rec

    # ---------------------------------------------------------
    # ---[第四轮推荐策略：基于 V4 模型的 K-Means 多样性批量主动学习] ---
    # ---------------------------------------------------------
    print("\n" + "="*85)
    print("【第四轮成分盲区探索结果：高精度网格 基于 V4 模型 + K-Means 多样性聚类】")
    print(" 策略：避免实验点扎堆，在四元成分空间中寻找3个相互独立的高不确定性区域")
    print("="*85)
    
    # 1. 筛选出满足 Ms 阈值要求的成分池
    df_target = df_rec_pool[df_rec_pool['predicted_Ms'] >= MS_THRESHOLD].copy()
    
    # 2. 截取不确定性最高的前 150 个点作为“高危盲区候选池”
    top_candidates = df_target.sort_values(by='updated_std', ascending=False).head(150).copy()
    
    # 3. 对这 150 个成分点进行特征标准化（确保各元素权重一致）
    comp_cols =['Fe_at', 'Si_at', 'Al_at', 'Cr_at']
    X_comp_scaled = StandardScaler().fit_transform(top_candidates[comp_cols])
    
    # 4. 使用 KMeans 将这些盲区强制划分为 3 个物理距离较远的“孤岛” (n_init 避免版本警告)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    top_candidates['cluster'] = kmeans.fit_predict(X_comp_scaled)
    
    # 5. 从每个孤岛（Cluster）中，挑出那个不确定性最高的最优代表点
    diverse_recommendations =[]
    for c in range(3):
        cluster_data = top_candidates[top_candidates['cluster'] == c]
        best_in_cluster = cluster_data.loc[cluster_data['updated_std'].idxmax()]
        diverse_recommendations.append(best_in_cluster)
    
    # 6. 将选出的 3 个代表点按照不确定性从高到低排序输出
    diverse_df = pd.DataFrame(diverse_recommendations).sort_values(by='updated_std', ascending=False)
    
    for i, (idx, row) in enumerate(diverse_df.iterrows(), 1):
        at_vals = row[['Fe_at', 'Si_at', 'Al_at', 'Cr_at']].values
        w_mult = ELEMENT_DATA.loc[['Fe', 'Si', 'Al', 'Cr'], 'weight'].values
        masses = at_vals * w_mult
        wt_pcts = masses / masses.sum() * 100
        
        print(f"🌍 独立盲区 {i}[该区域最大 σ={row['updated_std']:.2f}]: ")
        print(f"  ▶[at.%]: Fe={row['Fe_at']:.1f}, Si={row['Si_at']:.1f}, Al={row['Al_at']:.1f}, Cr={row['Cr_at']:.1f}")
        print(f"  ▶ [wt.%]: Fe={wt_pcts[0]:.1f}, Si={wt_pcts[1]:.1f}, Al={wt_pcts[2]:.1f}, Cr={wt_pcts[3]:.1f}")
        print(f"  ▶ 预测 Ms={row['predicted_Ms']:.1f} emu/g")
        print("-" * 80)

if __name__ == "__main__":
    main()