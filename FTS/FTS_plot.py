import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

data_path = "/home/rp/abhay_ws/RCC_modeling/FTS/data/ati_192-168-10-133_2025-11-04_10-40-09.csv" 
df = pd.read_csv(data_path)
time = df['t_epoch'].values - df['t_epoch'].values[0]  # relative time in seconds
fx = df['Fx'].values
fy = df['Fy'].values
fz = df['Fz'].values
tx = df['Tx'].values
ty = df['Ty'].values
tz = df['Tz'].values

# plot figure of 2x3 subplots for forces and torques
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
labels = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
data = [fx, fy, fz, tx, ty, tz]
for i in range(6):
    row = i // 3
    col = i % 3
    axs[row, col].plot(time, data[i])
    axs[row, col].set_title(labels[i])
    axs[row, col].set_xlabel('Time (s)')
    axs[row, col].set_ylabel(labels[i] + ' (units)')
plt.tight_layout()
plt.show()