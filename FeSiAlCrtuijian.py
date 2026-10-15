import numpy as np
import pandas as pd
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, DotProduct
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

# --- [1. 基础物理特性配置] ---
ELEMENT_DATA = pd.DataFrame({
    'Fe': [8, 1.241, 55.845],
    'Si': [4, 1.153, 28.0855],
    'Al': [3, 1.432, 26.9815],
    'Cr': [6, 1.249, 51.9961]
}, index=['valence', 'radius', 'weight']).T

def compute_physics_features(df):
    """计算原子尺寸差(atom)和价电子浓度(vec)"""
    elements = ['Fe', 'Si', 'Al', 'Cr']
    at_cols = [f'{e}_at' for e in elements]
    fractions = df[at_cols].values / 100.0
    
    # 1. 计算 vec (价电子浓度)
    df['vec'] = np.dot(fractions, ELEMENT_DATA['valence'].values)
    
    # 2. 计算 atom (原子尺寸差)
    radius_values = ELEMENT_DATA['radius'].values
    presence_mask = (df[at_cols].values > 0)
    row_radius = np.where(presence_mask, radius_values, np.nan)
    # 使用简单尺寸差百分比
    df['atom'] = (np.nanmax(row_radius, axis=1) / np.nanmin(row_radius, axis=1) - 1) * 100
    return df

def at_to_wt(df):
    """原子百分比转重量百分比（称料参考）"""
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

# --- [2. 核心主程序] ---
def main():
    SET_SEED = 42
    
    # 数据加载
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "combined_training_data.csv")
    
    if not os.path.exists(data_path):
        print("错误：未找到训练数据文件。")
        return

    df_train = pd.read_csv(data_path)
    # 特征：4个组分 + 2个物理特征
    feature_cols = ['Fe_at', 'Si_at', 'Al_at', 'Cr_at', 'atom', 'vec']
    X_train = df_train[feature_cols]
    y_train = df_train['Ms']
    
    # 异方差 Alpha 设置 (0:理论, 1:文献, 2:自测)
    # 按照弹性化逻辑，这里先定义基础权重系数
    alpha_weights = df_train['source'].map({0: 225.0, 1: 49.0, 2: 9.0}).values

    # --- [关键：注入最佳模型参数 (基于 Linear + RBF ARD)] ---
    # 注意：这里的 noise_level 将作为 alpha 的缩放因子
    best_p = {
        'c_amplitude': 0.852410,   # 示例最佳参数，请替换为你运行筛选代码得到的最新值
        'noise_level': 0.05120,   # 缩放因子
        'sigma0': 55.321045,
        'ls': [7.21, 15.43, 0.85, 3.12, 1.05, 1.22] # 对应 6 个特征的长度尺度
    }

    # 构建弹性 Alpha
    final_alpha = alpha_weights * best_p['noise_level']

    kernel = (
        C(best_p['c_amplitude']) * DotProduct(sigma_0=best_p['sigma0']) + 
        C(best_p['c_amplitude']) * RBF(length_scale=np.array(best_p['ls']))
    )

    # 训练模型
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('gpr', GaussianProcessRegressor(kernel=kernel, alpha=final_alpha, optimizer=None, normalize_y=True, random_state=SET_SEED))
    ])
    model.fit(X_train, y_train)

    # --- [3. 空间搜索与不确定度评估] ---
    print("正在构建四元合金成分空间 (步长 1.0)...")
    s_min, s_max = df_train['Si_at'].min(), df_train['Si_at'].max()
    a_min, a_max = df_train['Al_at'].min(), df_train['Al_at'].max()
    c_min, c_max = df_train['Cr_at'].min(), df_train['Cr_at'].max()
    fe_min, fe_max = df_train['Fe_at'].min(), df_train['Fe_at'].max()

    candidates = []
    for si in np.arange(s_min, s_max + 1, 1.0):
        for al in np.arange(a_min, a_max + 1, 1.0):
            for cr in np.arange(c_min, c_max + 1, 1.0):
                fe = 100.0 - (si + al + cr)
                if fe_min <= fe <= fe_max:
                    candidates.append([fe, si, al, cr])
    
    df_pool = pd.DataFrame(candidates, columns=['Fe_at', 'Si_at', 'Al_at', 'Cr_at'])
    df_pool = compute_physics_features(df_pool)
    
    print(f"池大小: {len(df_pool)}。正在执行全域不确定度扫描...")
    mu, sigma = model.predict(df_pool[feature_cols], return_std=True)
    df_pool['predicted_Ms'] = mu
    df_pool['std_dev'] = sigma

    # --- [4. 全局不确定度评价指标] ---
    mpsd = np.mean(sigma)
    max_sigma = np.max(sigma)
    
    print("\n" + "="*50)
    print("      成分空间预测不确定度评估")
    print("="*50)
    print(f" 平均预测标准差 (MPSD): {mpsd:.4f}")
    print(f" 最大预测标准差 (Max-σ): {max_sigma:.4f}")
    print(f" 状态说明: MPSD越小，模型对空间的整体掌握越准。")
    print("="*50)

    # --- [5. 主动学习推荐：纯探索 (Max-Sigma)] ---
    # 推荐 5 个模型目前最“心虚”的成分
    top_exploration = df_pool.sort_values(by='std_dev', ascending=False).head(5).copy()
    top_exploration = at_to_wt(top_exploration)
 
    print("\n【主动学习推荐方案：降低成分空间盲区】")
    print("以下配方是模型当前最不确定的区域，进行实验可最大化提升模型全域精度：")
    print("-" * 85)
    for i, (idx, row) in enumerate(top_exploration.iterrows(), 1):
        print(f"建议 {i} [盲区风险值 σ={row['std_dev']:.4f}]:")
        print(f"  [at.%]: Fe={row['Fe_at']:.1f}, Si={row['Si_at']:.1f}, Al={row['Al_at']:.1f}, Cr={row['Cr_at']:.1f}")
        print(f"  [wt.%]: Fe={row['Fe_wt']:.1f}, Si={row['Si_wt']:.1f}, Al={row['Al_wt']:.1f}, Cr={row['Cr_wt']:.1f}")
        print(f"  当前预测 Ms: {row['predicted_Ms']:.2f} emu/g")
        print(f"  探索理由：该区域缺乏实验数据支撑，实测后将显著带动周围成分空间的预测精度。")
        print("-" * 85)

if __name__ == "__main__":
    main()