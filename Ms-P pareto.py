import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from scipy.stats import norm

# ---[1. 基础物理特性与转换] ---
ELEMENT_DATA = pd.DataFrame({
    'Fe':[8, 1.241, 55.845],
    'Si':[4, 1.153, 28.0855],
    'Al':[3, 1.432, 26.9815],
    'Cr':[6, 1.249, 51.9961]
}, index=['valence', 'radius', 'weight']).T

def compute_physics_features(df):
    """计算模型所需的物理特征 (基于 at%)"""
    elements =['Fe', 'Si', 'Al', 'Cr']
    at_cols = [f'{e}_at' for e in elements]
    fractions = df[at_cols].values / 100.0
    df['vec'] = np.dot(fractions, ELEMENT_DATA['valence'].values)
    
    radius_values = ELEMENT_DATA['radius'].values
    presence_mask = (df[at_cols].values > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        row_radius = np.where(presence_mask, radius_values, np.nan)
        df['atom'] = (np.nanmax(row_radius, axis=1) / np.nanmin(row_radius, axis=1) - 1) * 100
    df['atom'] = df['atom'].fillna(0)
    return df

def at_to_wt(df):
    """原子百分比(at%)转质量百分比(wt%)，用于计算电阻率"""
    weights = ELEMENT_DATA['weight']
    mass_fe = df['Fe_at'] * weights['Fe']
    mass_si = df['Si_at'] * weights['Si']
    mass_al = df['Al_at'] * weights['Al']
    mass_cr = df['Cr_at'] * weights['Cr']
    total_mass = mass_fe + mass_si + mass_al + mass_cr
    
    df['Fe_wt'] = (mass_fe / total_mass) * 100
    df['Si_wt'] = (mass_si / total_mass) * 100
    df['Al_wt'] = (mass_al / total_mass) * 100
    df['Cr_wt'] = (mass_cr / total_mass) * 100
    return df

def calculate_resistivity(df):
    """经验公式计算电阻率 (基于 wt%)"""
    df['rho'] = 10 + 13.75 * df['Si_wt'] + 11.6 * df['Al_wt'] + 4 * df['Cr_wt']
    return df

def calculate_ei(mu, sigma, f_star):
    """
    计算预期改进 (Expected Improvement, EI)
    f_star: 目标基准值 (此处为 Sendust 的 124)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = (mu - f_star) / sigma
        ei = (mu - f_star) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = np.maximum(mu[sigma == 0.0] - f_star, 0.0)
    return ei

def find_pareto_front(df, col1, col2, maximize_both=True):
    """提取两个最大化目标的帕累托前沿"""
    pts = df[[col1, col2]].values
    is_pareto = np.ones(pts.shape[0], dtype=bool)
    for i, pt in enumerate(pts):
        if is_pareto[i]:
            # 判断是否被其他点严格支配
            is_pareto[is_pareto] = np.any(pts[is_pareto] > pt, axis=1) | np.all(pts[is_pareto] == pt, axis=1)
            is_pareto[i] = True 
    return df[is_pareto].sort_values(by=col1)

# ---[2. 核心主程序] ---
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "ms_gpr_matern05_model_v4.pkl")
    
    if not os.path.exists(model_path):
        print(f"❌ 未找到 V4 模型 ({model_path})，请确认路径。")
        return

    # 加载已冻结的 V4 模型
    package = joblib.load(model_path)
    scaler = package['scaler']
    model = package['model']
    feature_cols = package['features']

    # --- 1. 细密构建整个成分空间进行“穷举” (步长 0.5) ---
    print("正在构建精细成分空间 (步长 0.5)...")
    candidates =[]
    # 限制 Fe>=70 剔除明显非磁性/无法成型的区域
    for si in np.arange(1, 20.5, 0.5):
        for al in np.arange(1, 15.5, 0.5):
            for cr in np.arange(1, 10.5, 0.5):
                fe = 100.0 - (si + al + cr)
                if 70 <= fe <= 98:
                    candidates.append([fe, si, al, cr])
                    
    df_pool = pd.DataFrame(candidates, columns=['Fe_at', 'Si_at', 'Al_at', 'Cr_at'])
    df_pool = compute_physics_features(df_pool)
    df_pool = at_to_wt(df_pool)
    df_pool = calculate_resistivity(df_pool)
    
    # --- 2. 预测 Ms 分布并计算 EI ---
    print("正在使用 V4 模型执行全域推理与 EI 评估...")
    X_pool_scaled = scaler.transform(df_pool[feature_cols])
    mu, sigma = model.predict(X_pool_scaled, return_std=True)
    
    df_pool['mu'] = np.maximum(mu, 0.0) # 预测均值
    df_pool['sigma'] = sigma            # 预测不确定度
    
    # 【核心：以仙台斯特合金的 124 emu/g 作为基准 f*】
    TARGET_F_STAR = 124.0 
    df_pool['EI'] = calculate_ei(df_pool['mu'].values, df_pool['sigma'].values, TARGET_F_STAR)
    
    # --- 3. 计算双目标帕累托前沿 (目标：EI 最大化 & ρ 最大化) ---
    print("正在计算 EI vs. ρ 的帕累托前沿...")
    
    # 为了避免无意义的运算，过滤掉 EI < 0.01 的死寂区域（毫无打破 124 的希望）
    df_promising = df_pool[df_pool['EI'] > 0.01].copy()
    
    if df_promising.empty:
        print("⚠️ 模型认为在当前搜索空间内，几乎没有可能打破 124 emu/g 的配方。")
        return

    pareto_df = find_pareto_front(df_promising, 'EI', 'rho', maximize_both=True)

    # ---[4. 顶级学术可视化] ---
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    
    # 绘制背景候选池
    sc = ax.scatter(df_promising['rho'], df_promising['EI'], 
                    c=df_promising['mu'], cmap='viridis', 
                    alpha=0.4, s=15, label='Design Space (Promising)')
    
    # 绘制帕累托前沿连线与强调点
    ax.plot(pareto_df['rho'], pareto_df['EI'], 
            color='#E74C3C', linewidth=2.5, zorder=5, label='Pareto Front (EI vs $\\rho$)')
    ax.scatter(pareto_df['rho'], pareto_df['EI'], 
               color='#F1C40F', edgecolor='k', s=80, marker='*', zorder=6)
    
    # 颜色条表示预测的均值 μ
    cbar = plt.colorbar(sc)
    cbar.set_label('Predicted $M_s$ Mean ($\mu$, emu/g)', fontsize=14, fontweight='bold')
    
    ax.set_xlabel(r'Calculated Resistivity $\rho$ ($\mu\Omega \cdot cm$)', fontsize=16, fontweight='bold')
    ax.set_ylabel(r'Expected Improvement (EI) over Sendust (124 emu/g)', fontsize=16, fontweight='bold')
    ax.set_title(r'Discovering High-$\mu$ / High-$\rho$ FeSiAlCr Alloys via EI-guided Pareto Front', fontsize=16)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=14)
    plt.tight_layout()
    plt.show()

# ---[5. 打印帕累托前沿选点详情] ---
    
    # 【修复】：强制按 EI 降序排列，让 EI 最大的排在最前面作为推荐点1
    pareto_df = pareto_df.sort_values(by='EI', ascending=False)
    
    # 【修复】：去除了所有带斜杠的 \rho, \mu, \sigma 隐患，直接使用物理字符
    print("\n" + "="*95)
    print(f"【终极突破推荐：基于 Sendust (Ms=124) 的 EI-ρ 帕累托前沿】")
    print(" 以下配方是平衡了“极高打破记录的概率 (EI)”与“高频绝缘抗涡流需求 (ρ)”的绝对物理极值点：")
    print("="*95)
    
    num_recommendations = min(5, len(pareto_df))
    indices = np.linspace(0, len(pareto_df)-1, num_recommendations, dtype=int)
    
    for i, idx in enumerate(indices, 1):
        row = pareto_df.iloc[idx]
        
        if i == 1: 
            strategy = "最高打破 124 的概率 (此时电阻率相对垫底)"
        elif i == num_recommendations: 
            strategy = "最高的电阻率绝缘性 (保留一丝打破 124 的希望)"
        else: 
            strategy = "综合平衡型 (高阻抗 + 高突破率)"
            
        print(f"🌟 推荐点 {i} [{strategy}]:")
        print(f"[AI 评估指标]  EI 潜力值 = {row['EI']:.3f}  |  理论电阻率 ρ = {row['rho']:.1f} μΩ·cm")
        print(f"  [预测详细值]  Ms 均值 μ = {row['mu']:.1f} emu/g  |  模型不确定度 σ = {row['sigma']:.2f}")
        print(f"  [称重配方 wt%] Fe={row['Fe_wt']:.2f}, Si={row['Si_wt']:.2f}, Al={row['Al_wt']:.2f}, Cr={row['Cr_wt']:.2f}")
        print(f"  [原子配方 at%] Fe={row['Fe_at']:.1f}, Si={row['Si_at']:.1f}, Al={row['Al_at']:.1f}, Cr={row['Cr_at']:.1f}")
        print("-" * 95)

if __name__ == "__main__":
    main()