import numpy as np
import pandas as pd
import os
import joblib

# --- [1. 基础物理特性配置] ---
ELEMENT_DATA = pd.DataFrame({
    'Fe': [8, 1.241, 55.845],
    'Si': [4, 1.153, 28.0855],
    'Al': [3, 1.432, 26.9815],
    'Cr': [6, 1.249, 51.9961]
}, index=['valence', 'radius', 'weight']).T

def compute_physics_features(df):
    """计算原子尺寸差(atom)和价电子浓度(vec)"""
    elements =['Fe', 'Si', 'Al', 'Cr']
    at_cols =[f'{e}_at' for e in elements]
    fractions = df[at_cols].values / 100.0
    
    # 1. 计算 vec (价电子浓度)
    df['vec'] = np.dot(fractions, ELEMENT_DATA['valence'].values)
    
    # 2. 计算 atom (原子尺寸差)
    radius_values = ELEMENT_DATA['radius'].values
    presence_mask = (df[at_cols].values > 0)
    
    # 忽略除以零或全是 NaN 的警告
    with np.errstate(divide='ignore', invalid='ignore'):
        row_radius = np.where(presence_mask, radius_values, np.nan)
        # 使用简单尺寸差百分比
        df['atom'] = (np.nanmax(row_radius, axis=1) / np.nanmin(row_radius, axis=1) - 1) * 100
    
    # 将完全没有成分导致的 NaN 补为 0
    df['atom'] = df['atom'].fillna(0)
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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 加载训练数据获取成分搜索边界
    data_path = os.path.join(current_dir, "combined_training_data.csv")
    if not os.path.exists(data_path):
        print("错误：未找到训练数据文件(需要它来确定搜索边界)。")
        return
    df_train = pd.read_csv(data_path)
    
    # 2. 加载预训练的物理认知模型包 (Matern v=0.5)
    model_path = os.path.join(current_dir, "ms_gpr_matern05_model.pkl")
    if not os.path.exists(model_path):
        print("错误：未找到模型文件 ms_gpr_matern05_model.pkl，请先运行保存模型的代码。")
        return
        
    print("正在加载预训练的 GPR 模型及 Scaler...")
    loaded_package = joblib.load(model_path)
    scaler = loaded_package['scaler']
    model = loaded_package['model']
    feature_cols = loaded_package['features'] # 确保列顺序与训练时绝对一致

    # ---[3. 空间搜索与特征构建] ---
    print("正在构建四元合金成分空间 (步长 0.5)...")
    s_min, s_max = df_train['Si_at'].min(), df_train['Si_at'].max()
    a_min, a_max = df_train['Al_at'].min(), df_train['Al_at'].max()
    c_min, c_max = df_train['Cr_at'].min(), df_train['Cr_at'].max()
    fe_min, fe_max = df_train['Fe_at'].min(), df_train['Fe_at'].max()

    candidates =[]
    for si in np.arange(s_min, s_max + 0.5, 0.5):
        for al in np.arange(a_min, a_max + 0.5, 0.5):
            for cr in np.arange(c_min, c_max + 0.5, 0.5):
                fe = 100.0 - (si + al + cr)
                if fe_min <= fe <= fe_max:
                    candidates.append([fe, si, al, cr])
    
    df_pool = pd.DataFrame(candidates, columns=['Fe_at', 'Si_at', 'Al_at', 'Cr_at'])
    df_pool = compute_physics_features(df_pool)
    
    # 严格按照训练时的特征顺序排列
    X_pool = df_pool[feature_cols]
    
    # --- [4. 预测与不确定度评估] ---
    print(f"池大小: {len(df_pool)}。正在执行全域不确定度扫描...")
    
    # 使用训练好的 scaler 对候选池进行标准化
    X_pool_scaled = scaler.transform(X_pool)
    
    # 执行预测并返回均值和标准差
    mu, sigma = model.predict(X_pool_scaled, return_std=True)
    
    # 【物理常识后处理】：确保即使是最边缘的非磁性成分，Ms也不会小于 0
    mu = np.maximum(mu, 0.0)
    
    df_pool['predicted_Ms'] = mu
    df_pool['std_dev'] = sigma

    # 全局不确定度评价指标
    mpsd = np.mean(sigma)
    max_sigma = np.max(sigma)
    
    print("\n" + "="*50)
    print("      成分空间预测不确定度评估")
    print("="*50)
    print(f" 平均预测标准差 (MPSD): {mpsd:.4f}")
    print(f" 最大预测标准差 (Max-\u03C3): {max_sigma:.4f}")
    print(f" 状态说明: MPSD越小，模型对空间的整体掌握越准。")
    print("="*50)

    # ---[5. 主动学习推荐：纯探索 (Max-Sigma)] ---
    target_candidates = [[80.0, 1.0, 9.0,  10.0],[79.0, 1.0, 10.0, 10.0],[78.0, 1.0, 11.0, 10.0]]
    top_exploration = pd.DataFrame(target_candidates, columns=['Fe_at', 'Si_at', 'Al_at', 'Cr_at'])
    
    # 1. 补全物理特征 (vec, atom) 以满足模型输入要求
    top_exploration = compute_physics_features(top_exploration)
    
    # 2. 提取对应特征并使用加载的 scaler 进行标准化
    X_target = top_exploration[feature_cols]
    X_target_scaled = scaler.transform(X_target)
    
    # 3. 预测 Ms 与不确定性 (std_dev)
    mu_target, sigma_target = model.predict(X_target_scaled, return_std=True)
    top_exploration['predicted_Ms'] = np.maximum(mu_target, 0.0) # 物理约束：不小于0
    top_exploration['std_dev'] = sigma_target
    
    # 4. 转换为 wt.% 以供实验称料参考
    top_exploration = at_to_wt(top_exploration)
 
    print("\n【成分预测及不确定性评估】")
    print("以下为推荐的 3 个配方的预测结果：")
    print("-" * 85)
    for i, (idx, row) in enumerate(top_exploration.iterrows(), 1):
        print(f"推荐配方 {i} [预测不确定度 \u03C3 = {row['std_dev']:.4f}]:")
        print(f"  [at.%]: Fe={row['Fe_at']:.1f}, Si={row['Si_at']:.1f}, Al={row['Al_at']:.1f}, Cr={row['Cr_at']:.1f}")
        print(f"[wt.%]: Fe={row['Fe_wt']:.1f}, Si={row['Si_wt']:.1f}, Al={row['Al_wt']:.1f}, Cr={row['Cr_wt']:.1f}")
        print(f"  当前预测 Ms: {row['predicted_Ms']:.2f} A·m²/kg")
        print("-" * 85)

if __name__ == "__main__":
    main()