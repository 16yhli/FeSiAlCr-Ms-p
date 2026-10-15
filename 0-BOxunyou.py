import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from scipy.interpolate import PchipInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

# ---[0. 加载数据] ---
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "combined_training_data.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError("未找到数据文件，请检查路径。")

df = pd.read_csv(data_path)

X = df[['Fe_at', 'Si_at', 'Al_at', 'Cr_at', 'atom', 'vec']]
y = df['Ms']
source = df['source']
alpha_weights = source.map({0: 225.0, 1: 49.0, 2: 9.0}).values

SET_SEED = 123  

# ---[1. 定义 Optuna 贝叶斯优化的目标函数] ---
def objective(trial):
    c_val = trial.suggest_float("C", 0.1, 2000.0, log=True)
    l_scale = trial.suggest_float("length_scale", 0.01, 200.0, log=True)
    
    kernel = C(c_val) * Matern(length_scale=l_scale, nu=0.5)
    
    cv = KFold(n_splits=5, shuffle=True, random_state=SET_SEED)
    all_preds, all_true = [],[]
    
    for train_idx, test_idx in cv.split(X):
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X.iloc[train_idx])
        X_test_s = scaler.transform(X.iloc[test_idx])
        
        gpr = GaussianProcessRegressor(
            kernel=kernel, 
            alpha=alpha_weights[train_idx], 
            normalize_y=False, 
            optimizer=None,  
            random_state=SET_SEED
        )
        
        try:
            gpr.fit(X_train_s, y.iloc[train_idx])
            preds = gpr.predict(X_test_s)
        except:
            return -999.0 
            
        all_preds.extend(preds)
        all_true.extend(y.iloc[test_idx])
        
    r2 = r2_score(all_true, all_preds)
    return r2

# --- [2. 执行贝叶斯优化] ---
print("正在执行 Matern(v=0.5) 模型的贝叶斯超参数寻优 (BO)...")
optuna.logging.set_verbosity(optuna.logging.WARNING) 
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SET_SEED))

study.optimize(objective, n_trials=80)

# ---[3. 提取历史数据与平滑处理] ---
trials = study.trials
trial_numbers =[t.number for t in trials if t.value is not None and t.value > 0]
obj_values =[t.value for t in trials if t.value is not None and t.value > 0]

running_max = np.maximum.accumulate(obj_values)

x_smooth = np.linspace(min(trial_numbers), max(trial_numbers), 500)
pchip = PchipInterpolator(trial_numbers, running_max)
y_smooth = pchip(x_smooth)

# ---[4. 绘制终极顶刊比例图表] ---
plt.rcParams['font.family'] = 'Arial' 

# 【核心修改 1】：绝对正方形紧凑比例，3.2 x 3.2 英寸，完美适配单栏
fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=600) 

color_scatter = '#5DADE2' 
color_line = '#C0392B'    

# 【核心修改 2】：散点缩至 8
ax.scatter(trial_numbers, obj_values, color=color_scatter, s=8, alpha=0.5, 
           linewidths=0, label='Objective value ($R^2$)', zorder=2)

# 【核心修改 3】：主曲线厚度降至 1.2
ax.plot(x_smooth, y_smooth, color=color_line, linewidth=1.2, 
        alpha=0.95, label='Best value ($R^2$)', zorder=3)

ax.set_facecolor('white') 
ax.grid(False) 

# 【核心修改 4】：边框线宽降至 0.8 pt
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color('black')

# 【核心修改 5】：刻度线缩短变细，数字降至 7 pt
ax.tick_params(axis='both', direction='in', labelsize=7, width=0.8, length=3, pad=3, top=True, right=True) 

y_min = max(0.40, min(obj_values) - 0.05)
ax.set_ylim(y_min, 1.05) 
ax.set_xlim(min(trial_numbers) - 2, max(trial_numbers) + 2)

# 【核心修改 6】：轴标签降至 8 pt，间距 labelpad 压缩
ax.set_xlabel("Number of trials", fontsize=8, labelpad=4)
ax.set_ylabel("Cross-validation $R^2$ score", fontsize=8, labelpad=4)

# 【核心修改 7】：图例降至 7 pt，框线极细 (0.6 pt)
leg = ax.legend(loc='lower right', fontsize=7, frameon=True, edgecolor='black', framealpha=1.0, fancybox=False, borderpad=0.4, handlelength=1.5)
leg.get_frame().set_linewidth(0.6)

plt.tight_layout(pad=0.5)

save_path = os.path.join(current_dir, "BO_Optimization_History_Square.pdf")
plt.savefig(save_path, format='pdf', bbox_inches='tight')
plt.show()

print(f"严苛正方形顶刊版已保存至: {save_path}")