import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os 

results_path = "/media/rp/Elements1/abhay_ws/RCC_modeling/CNN_model/checkpoints/ablation/20251101_132517/ablation_results.csv"

# read results
df = pd.read_csv(results_path)

# print columns 
print(df.columns.tolist())


# input metrics
input_metrics = ['fraction']

# output metrics 
output_metrics = ['best_val_loss', 'best_mae_mean', 'best_rmse_mean', 'best_val_class_mean_acc']

# plot all output metrics vs input metrics
for out_metric in output_metrics: 
    plt.figure() 
    plt.plot(df['fraction'], df[out_metric], marker='o') 
    plt.xlabel('Input Data Fraction') 
    plt.ylabel(out_metric) 
    plt.title(f'{out_metric} vs Input Data Fraction') 
    plt.legend() 
    plt.grid() 
    if out_metric == 'best_val_class_mean_acc':
        plt.ylim(0, 1)
    out_path = os.path.join(os.path.dirname(results_path), f'ablation_{out_metric}_vs_data_fraction.png')
    plt.savefig(out_path) 
    plt.close()


# fit a log curve to best_val_class_mean_acc vs fraction and project up to best_val_class_mean_acc=1 and coplot with a dashed line 
from scipy.optimize import curve_fit
def log_func(x, a, b):
    return a * np.log(x) + b
x_data = df['fraction'].values
y_data = df['best_val_class_mean_acc'].values
# fit curve
popt, pcov = curve_fit(log_func, x_data, y_data)
# determine x where y reaches 1.0 and build projection domain that stops at that y
a, b = popt
x_target = np.nan
if abs(a) > 1e-12:
    x_target = np.exp((1.0 - b) / a)
# define safe domain bounds
_eps = 1e-4
x_min_data = float(np.nanmin(x_data))
x_max_data = float(np.nanmax(x_data))
# default start/end
x_start = max(_eps, min(x_min_data, 0.01))
x_end = max(1.0, x_max_data)
# adjust to ensure projection stops exactly at y=1.0 when feasible in (0, +inf)
if np.isfinite(x_target) and x_target > 0:
    if a > 0:
        # y increases with x; stop at x_target
        x_end = x_target
    elif a < 0:
        # y decreases with x; start at x_target so left boundary is y=1
        x_start = max(_eps, x_target)
# ensure valid increasing range
if not np.isfinite(x_start) or not np.isfinite(x_end) or x_end <= x_start:
    x_start = max(_eps, min(x_min_data, 0.01))
    x_end = max(1.0, x_max_data)
# generate x values for projection
x_proj = np.linspace(x_start, x_end, 200)
y_proj = log_func(x_proj, *popt)
# plot
plt.figure()
plt.plot(x_data, y_data, 'o', label='Data')
plt.plot(x_proj, y_proj, '-', label='Log Fit Projection (to y=1.0)')
# find x value where y = 1 and annotate
if np.isfinite(x_target) and x_target > 0:
    plt.axvline(x=x_target, color='r', linestyle='--', label=f'Projected frac for 100% Acc: {x_target:.3f}')
# reference line at y=1.0
plt.axhline(y=1.0, color='k', linestyle=':', linewidth=1)
plt.xlabel('Input Data Fraction')
plt.ylabel('Best Validation Classification Mean Accuracy')
plt.title('Best Validation Classification Mean Accuracy vs Input Data Fraction')
plt.legend()
plt.grid()
plt.ylim(0, 1.05)
out_path = os.path.join(os.path.dirname(results_path), f'ablation_best_val_class_mean_acc_with_projection.png')
plt.savefig(out_path)
plt.close()