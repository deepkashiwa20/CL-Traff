import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import os

METRICS = ["RMSE", "MAE", "MAPE"]

TIMESTEPS = [3, 6, 12]

LAMBDA_RANGE = [0, 0.01, 0.05, 0.1, 1.0]  # 0 means original baselines

TEMP_RANGE = [0, 0.1, 0.3, 0.5, 0.7, 1.0]

TOPK_RANGE = [0, 1, 2, 5, 10, 20]

ORIGINAL_METRICS = np.array([[[5.55, 2.86, 7.55], [6.57, 3.25, 8.99], [7.56, 3.68, 10.46]]]) #* Baseline

SCHEMA_3_FUSION1_METRICS = np.array([[[5.43, 2.84, 7.63], [6.72, 3.35, 9.59], [8.55, 4.22, 12.76]]])  # 1 layer fusion is worse

SCHEMA_3_FUSION2_METRICS = np.array([[[5.41, 2.82, 7.39], [6.60, 3.27, 9.12], [7.93, 3.84, 11.34]]])  # 2 layer fusion is worse

NO_SPATIAL_NEG = np.array([[[5.30, 2.78, 7.21], [6.40, 3.22, 8.87], [7.70, 3.80, 11.10]]])

SCHEMA_4_X_AUG_METRICS = np.array([[[5.35, 2.80, 7.25], [6.46, 3.23, 8.88], [7.74, 3.80, 11.00]]])

SCHEMA2_METRICS = np.array([[[5.28, 2.75, 7.14], [6.42, 3.18, 8.87], [7.77, 3.75, 11.12]]])  # temp=0.5 is not better than schema 1 

SCHEMA2_METRICS = np.array([[[5.31, 2.77, 7.18], [6.46, 3.22, 8.85], [7.83, 3.82, 11.12]]])  # temp=0.1 is not better than schema 1 

SCHEMA0_CONNECT_METRICS = np.array([[[5.46, 2.83, 7.52], [6.64, 3.27, 9.25], [7.95, 3.84, 11.46]]])  # connect and distance matrix is worse than adaptive matrix
SCHEMA0_DISTANCE_METRICS = np.array([[[5.50, 2.85, 7.45], [6.70, 3.31, 9.19], [8.11, 3.96, 11.59]]])

SCHEMA1_CONNECT_METRICS = np.array([[[5.59, 2.89, 7.64], [6.80, 3.38, 9.48], [8.27, 4.07, 12.05]]])
SCHEMA1_DISTANCE_METRICS = np.array([[[5.56, 2.88, 7.70], [6.77, 3.38, 9.62], [8.24, 4.06, 12.29]]])

SCHEMA1_CONNECT_DENOMINATOR_METRICS = np.array([[[5.56, 2.86, 7.53], [6.71, 3.33, 9.25], [8.04, 3.95, 11.58]]])
SCHEMA1_DISTANCE_DENOMINATOR_METRICS = np.array([[[5.49, 2.85, 7.56], [6.67, 3.31, 9.34], [8.06, 3.96, 11.83]]])
SCHEMA1_ADAPTIVE_DENOMINATOR_METRICS = np.array([[[5.31, 2.78, 7.20], [6.43, 3.22, 8.89], [7.75, 3.81, 11.21]]])  # whether constrative denominators is less impact

# 6.187012, 3.378840, 8.599727 7.720099, 4.039131, 10.803047, 9.867319, 5.175108, 14.482716 learning_rate=0.03 is worse
# 

# temp = 0.1 & top_k = 10, varying the lam from {0.01, 0.05, 0.1, 1.0}
SCHEMA_1_LAM_METRICS = np.array([[[5.29, 2.77, 7.18], [6.42, 3.21, 8.90], [7.77, 3.79, 11.19]],
               [[5.28, 2.76, 7.16], [6.40, 3.21, 8.71], [7.75, 3.79, 10.78]],  # better
               [[5.35, 2.81, 7.34], [6.48, 3.25, 9.04], [7.78, 3.84, 11.23]],
               [[5.53, 2.89, 7.65], [6.65, 3.33, 9.33], [7.80, 3.96, 11.64]]
            ])

# lam = 0.05 & top_k = 10, varying the temp from {0.1, 0.3, 0.5, 0.7 1.0}
SCHEMA_1_TEMP_METRICS = np.array([[[5.28, 2.76, 7.16], [6.40, 3.21, 8.71], [7.75, 3.79, 10.78]],  # better
                [[5.35, 2.79, 7.31], [6.56, 3.25, 9.18], [7.89, 3.83, 11.55]],
                [[5.29, 2.77, 7.23], [6.36, 3.19, 8.84], [7.59, 3.73, 10.98]],
                [[5.31, 2.79, 7.20], [6.48, 3.23, 8.88], [7.83, 3.83, 11.08]],
                [[5.31, 2.79, 7.19], [6.43, 3.24, 8.92], [7.77, 3.82, 11.25]]
            ])

# lam = 0.05 & temp = 0.1, varying the top_k from {1, 2, 5, 10, 20}
SCHEMA_1_TOPK_METRICS = np.array([[[5.36, 2.80, 7.40], [6.49, 3.24, 9.13], [7.73, 3.79, 11.25]],
                [[5.35, 2.82, 7.35], [6.48, 3.29, 9.11], [7.83, 3.93, 11.43]],
                [[5.37, 2.81, 7.41], [6.54, 3.26, 9.14], [7.86, 3.84, 11.40]],
                [[5.28, 2.76, 7.16], [6.40, 3.21, 8.71], [7.75, 3.79, 10.78]],  # better
                [[5.33, 2.79, 7.31], [6.48, 3.23, 8.94], [7.82, 3.81, 11.02]]
            ])



dir_name = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(dir_name, 'Results')
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
# Apply the default theme
sns.set_theme(style="darkgrid")
plt.figure()

lambda_data = np.concatenate([ORIGINAL_METRICS, SCHEMA_1_LAM_METRICS], axis=0)
lambda_range = np.array(LAMBDA_RANGE)[:, np.newaxis]
temp_data = np.concatenate([ORIGINAL_METRICS, SCHEMA_1_TEMP_METRICS], axis=0)
temp_range = np.array(TEMP_RANGE)[:, np.newaxis]
topk_data = np.concatenate([ORIGINAL_METRICS, SCHEMA_1_TOPK_METRICS], axis=0)
topk_range = np.array(TOPK_RANGE)[:, np.newaxis]

for i, t in enumerate(TIMESTEPS):
    data = lambda_data[:, i, :]  # (5, 3)
    for m, metric in enumerate(METRICS):
        data_metric = data[:, [m]]  # (5, )
        data_metric_with_parameter = np.concatenate([lambda_range, data_metric], axis=-1)  # (5, 2)
        df = pd.DataFrame(data_metric_with_parameter, columns=['Lambda', metric])
        sns.relplot(data=df, x="Lambda", y=metric, kind="line", errorbar=None)
        plt.savefig(os.path.join(save_dir, 'lambda_comp_{}_{}_timestamp.pdf').format(metric, t))
        # plt.show()

for i, t in enumerate(TIMESTEPS):
    data = temp_data[:, i, :]  # (5, 3)
    for m, metric in enumerate(METRICS):
        data_metric = data[:, [m]]  # (5, )
        data_metric_with_parameter = np.concatenate([temp_range, data_metric], axis=-1)  # (5, 2)
        df = pd.DataFrame(data_metric_with_parameter, columns=['Temp', metric])
        sns.relplot(data=df, x="Temp", y=metric, kind="line", errorbar=None)
        plt.savefig(os.path.join(save_dir, 'temp_comp_{}_{}_timestamp.pdf').format(metric, t))
        # plt.show()

for i, t in enumerate(TIMESTEPS):
    data = topk_data[:, i, :]  # (5, 3)
    for m, metric in enumerate(METRICS):
        data_metric = data[:, [m]]  # (5, )
        data_metric_with_parameter = np.concatenate([topk_range, data_metric], axis=-1)  # (5, 2)
        df = pd.DataFrame(data_metric_with_parameter, columns=['Topk', metric])
        sns.relplot(data=df, x="Topk", y=metric, kind="line", errorbar=None)
        plt.savefig(os.path.join(save_dir, 'topk_comp_{}_{}_timestamp.pdf').format(metric, t))
        # plt.show()
