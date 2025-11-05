import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt

eval_data_dir = "/media/rp/Elements1/abhay_ws/RCC_modeling/CNN_model/checkpoints/"
file_index = 3000
chunk_index = 726
filename = f"val_predictions_regression_epoch_{file_index}.csv"
file_path = os.path.join(eval_data_dir, filename)
df_eval = pd.read_csv(file_path)
df_eval = df_eval[df_eval['chunk'] == chunk_index]
print(df_eval.describe())



# plot predicted vs true for all 6 outputs
output_cols = ['X_norm', 'Y_norm', 'Z_norm', 'A_norm', 'B_norm', 'C_norm']
for col in output_cols:
    plt.figure()
    plt.scatter(df_eval[f'gt_{col}'], df_eval[f'pred_{col}'], alpha=0.1)
    plt.xlabel(f'True {col}')
    plt.ylabel(f'Predicted {col}')
    plt.title(f'Predicted vs True for {col} (Epoch {file_index})')
    plt.plot([-3, 3], [-3, 3], 'r--')  # unity line
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    out_path = os.path.join(eval_data_dir, f'pred_vs_true_{col}_epoch_{file_index}.png')
    plt.savefig(out_path)
    plt.close()

# plot timeseries of true vs predicted in a 2x3 plot 
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
time_steps = np.arange(len(df_eval))
for i, col in enumerate(output_cols):
    row = i // 3
    col_idx = i % 3
    axs[row, col_idx].plot(time_steps, df_eval[f'gt_{col}'], label='True', alpha=0.7)
    axs[row, col_idx].plot(time_steps, df_eval[f'pred_{col}'], label='Predicted', alpha=0.7)
    axs[row, col_idx].set_title(f'True vs Predicted for {col} (Epoch {file_index})')
    axs[row, col_idx].set_xlabel('Time Step')
    axs[row, col_idx].set_ylabel(col)
    axs[row, col_idx].legend()
    axs[row, col_idx].grid()
    axs[row, col_idx].set_ylim([-1.5,1.5])
plt.tight_layout()
out_path = os.path.join(eval_data_dir, f'timeseries_pred_vs_true_epoch_{file_index}.png')
plt.savefig(out_path)
plt.show()
plt.close() 