import os
import joblib
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

# --- [1. 加载数据] ---
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "combined_training_data.csv")
df = pd.read_csv(data_path)

X = df[['Fe_at', 'Si_at', 'Al_at', 'Cr_at', 'atom', 'vec']]
y = df['Ms']

# 获取数据来源权重
source = df['source']
alpha_weights = source.map({0: 225.0, 1: 49.0, 2: 9.0}).values

# --- [2. 特征标准化] ---
# 必须保存这个scaler，以便将来对新配方进行相同的缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---[3. 构建并训练符合物理认知的 Matern (v=0.5) 模型] ---
print("正在使用全量数据训练 Matern (\u03BD=0.5) 模型...")
kernel = C(100.0) * Matern(length_scale=10.0, nu=0.5)

final_gpr = GaussianProcessRegressor(
    kernel=kernel, 
    alpha=alpha_weights, 
    normalize_y=False,       # 保持真实方差映射
    n_restarts_optimizer=10, # 多次重启以找到全局最优超参数
    random_state=42
)

final_gpr.fit(X_scaled, y)

# --- [4. 连同 Scaler 一起打包保存] ---
# 使用字典将模型和缩放器组合在一起
model_package = {
    'scaler': scaler,
    'model': final_gpr,
    'features':['Fe_at', 'Si_at', 'Al_at', 'Cr_at', 'atom', 'vec'] # 记录特征顺序防错
}

save_path = os.path.join(current_dir, "ms_gpr_matern05_model.pkl")
joblib.dump(model_package, save_path)

print(f"✅ 模型和预处理管道已成功保存至: {save_path}")
print(f"优化后的物理核参数: {final_gpr.kernel_}")