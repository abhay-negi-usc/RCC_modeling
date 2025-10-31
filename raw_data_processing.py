import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation as R 
import os 

import cicp.cicp_util as CU
from utils import z_rotation_matrix, kuka_fk, kuka_fk_batch, transform_wrench_ref_to_tgt

# open log data 
# data_path = "./data/RCC_mounted_data_2013-01-08_23-21-43.csv" # isolated z rotation trial 
# data_path = "./data/RCC_mounted_data_2013-01-08_11-04-47.csv" # full motion trial
data_path = "./data/RCC_combined_14.csv" # 14 combined trials 
df = pd.read_csv(data_path) 
df = df.rename(columns={
    'axisQMsr_LBR_iiwa_14_R820_1[0]': 'J0',
    'axisQMsr_LBR_iiwa_14_R820_1[1]': 'J1',
    'axisQMsr_LBR_iiwa_14_R820_1[2]': 'J2',
    'axisQMsr_LBR_iiwa_14_R820_1[3]': 'J3',
    'axisQMsr_LBR_iiwa_14_R820_1[4]': 'J4',
    'axisQMsr_LBR_iiwa_14_R820_1[5]': 'J5',
    'axisQMsr_LBR_iiwa_14_R820_1[6]': 'J6',    
})

# run forward kinematics to get position 
joint_positions = df[['J0', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6']].values.astype(np.float32) * np.pi / 180.0
tf_B_E = kuka_fk_batch(joint_positions).astype(float) # units: mm 
ee_positions = tf_B_E[:,:3,3]
ee_rotations = R.from_matrix(tf_B_E[:,:3,:3]).as_euler('xyz', degrees=True)
# ee_rotations = R.from_matrix(tf_ee[:,:3,:3]).as_quat()  # x,y,z,w
# concatenate positions and rotations
c,b,a = ee_rotations[:,0], ee_rotations[:,1], ee_rotations[:,2]
ee_rotations_abc = np.stack((a,b,c), axis=1)  
ee_poses = np.concatenate((ee_positions, ee_rotations_abc), axis=1)
ee_poses.shape
df['FK_X'] = ee_poses[:,0] 
df['FK_Y'] = ee_poses[:,1] 
df['FK_Z'] = ee_poses[:,2] 
df['FK_A'] = ee_poses[:,3] 
df['FK_B'] = ee_poses[:,4] 
df['FK_C'] = ee_poses[:,5] 

# define frames 
tf_E_RE = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 76],  
    [0, 0, 0, 1]], dtype=float).reshape(1,4,4)

tf_E_RP = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 114],  
    [0, 0, 0, 1]], dtype=float).reshape(1,4,4)

# compute poses with respect to zero frame 
tf_B_RE = CU.batch_compose_homogenous_transforms(tf_B_E, np.repeat(tf_E_RE, tf_B_E.shape[0], axis=0))
tf_B_RE0 = tf_B_RE[0,:,:].reshape(1,4,4)
tf_RE0_B = np.linalg.inv(tf_B_RE0)
tf_RE0_RE = CU.batch_compose_homogenous_transforms(np.repeat(tf_RE0_B, tf_B_RE.shape[0], axis=0), tf_B_RE)
pose_RE0_RE = CU.batch_matrices_to_poses_xyzabc(tf_RE0_RE)

# read wrench 
df_wrench_E = df[['cartForce1_X', 'cartForce1_Y', 'cartForce1_Z', 'cartTorque1_TauX',  'cartTorque1_TauY', 'cartTorque1_TauZ', ]]
wrench_E = df_wrench_E.astype(float)
tf_E_RE_meters = tf_E_RE.copy()
tf_E_RE_meters[0, :3, 3] /= 1000.  # convert to meters
wrench_RE = transform_wrench_ref_to_tgt(wrench_E, tf_E_RE_meters[0,:,:])
df_wrench_RE = pd.DataFrame(wrench_RE, columns=['FX', 'FY', 'FZ', 'TX', 'TY', 'TZ'])

# # coplot wrench and wrench_var 
# labels = ['Force_X', 'Force_Y', 'Force_Z', 'Torque_X', 'Torque_Y', 'Torque_Z']
# fig, axs = plt.subplots(2, 3, figsize=(15, 10))
# for i in range(6):
#     row = i // 3
#     col = i % 3
#     axs[row, col].plot(df_wrench_E.iloc[:,i], label='Wrench End Effector Frame')
#     axs[row, col].plot(df_wrench_RE.iloc[:,i], label='Wrench RCC-End Effector Side Frame')
#     axs[row, col].set_title(labels[i])
#     axs[row, col].set_xlabel('Time step')
#     axs[row, col].set_ylabel(labels[i])
#     axs[row, col].legend()
# plt.tight_layout()
# plt.show()

# # plot 2x3 subplots for positions and orientations of pose_RE0_RE
# labels = ['X', 'Y', 'Z', 'A', 'B', 'C']
# fig, axs = plt.subplots(2, 3, figsize=(15, 10))
# for i in range(6):
#     row = i // 3
#     col = i % 3
#     axs[row, col].plot(pose_RE0_RE[:,i])
#     axs[row, col].set_title(labels[i])
#     axs[row, col].set_xlabel('Time step')
#     axs[row, col].set_ylabel(labels[i])
# plt.tight_layout()
# plt.show()

# scatterplot wrench vs pose components
labels = ['X', 'Y', 'Z', 'A', 'B', 'C']
wrench_labels = ['FX', 'FY', 'FZ', 'TX', 'TY', 'TZ']
fig, axs = plt.subplots(6, 6, figsize=(20, 20))
for i in range(6):
    for j in range(6):
        axs[i, j].scatter(pose_RE0_RE[:,i], df_wrench_RE.iloc[:,j], s=1, alpha=0.1)
        axs[i, j].set_xlabel(labels[i])
        axs[i, j].set_ylabel(wrench_labels[j])
plt.tight_layout()
plt.show()

# 4x3 timeseries plot of pose vs time and wrench vs time with synced x-axis
fig, axs = plt.subplots(4, 3, figsize=(15, 15), sharex=True)
for i in range(6):
    row = i // 3
    col = i % 3
    axs[row, col].plot(pose_RE0_RE[:,i], label='Pose RE0 to RE')
    axs[row, col].set_title(labels[i])
    axs[row, col].set_ylabel(labels[i])
    axs[row, col].legend()
for i in range(6):
    row = (i + 6) // 3
    col = (i + 6) % 3
    axs[row, col].plot(df_wrench_RE.iloc[:,i], label='Wrench RCC-End Effector Side Frame', color='orange')
    axs[row, col].set_title(wrench_labels[i])
    axs[row, col].set_ylabel(wrench_labels[i])
    axs[row, col].legend()

# Add x-axis label only to the bottom row
for col in range(3):
    axs[3, col].set_xlabel('Time step')

plt.tight_layout()
plt.show()

# coplot timeseries TZ vs A 
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
axs[0].plot(pose_RE0_RE[:,5], label='Orientation C (Z)')
axs[0].set_title('Orientation C (Z)')
axs[0].set_ylabel('Degrees')
axs[0].legend()
axs[1].plot(df_wrench_RE['TZ'], label='Torque Z', color='orange')
axs[1].set_title('Torque Z')
axs[1].set_xlabel('Time step')
axs[1].set_ylabel('Nm')
axs[1].legend()
plt.tight_layout()
plt.show()

# make a new df of pose_RE0_RE and wrench_RE and save to csv 
df_processed = pd.DataFrame(pose_RE0_RE, columns=['X', 'Y', 'Z', 'A', 'B', 'C'])
df_processed[['FX', 'FY', 'FZ', 'TX', 'TY', 'TZ']] = df_wrench_RE[['FX', 'FY', 'FZ', 'TX', 'TY', 'TZ']]

# data labeling 
# compute normalized pose values ranging from -1 to +1 based on standard deviation 
# where +/-1 corresponds to +/-2 standard deviations from the mean
for i, label in enumerate(['X', 'Y', 'Z', 'A', 'B', 'C']):
    mean_val = df_processed[label].mean()
    std_val = df_processed[label].std()
    # Normalize such that ±2 std dev maps to ±1
    df_processed[label + '_norm'] = (df_processed[label] - mean_val) / (2 * std_val)   

# # histogram of normalized pose values
# fig, axs = plt.subplots(2, 3, figsize=(15, 10))
# for i, label in enumerate(['X', 'Y', 'Z', 'A', 'B', 'C']):
#     row = i // 3
#     col = i % 3
#     axs[row, col].hist(df_processed[label + '_norm'], bins=50, alpha=0.7)
#     axs[row, col].set_title(f'Normalized {label} Histogram')
#     axs[row, col].set_xlabel('Normalized Value')
#     axs[row, col].set_ylabel('Count')
# plt.tight_layout()
# plt.show()

# define class labels with sign consideration
# Note: with std dev normalization, ±1 = ±2σ, so ±0.5 ≈ ±1σ
normalized_class_boundaries = [0.5, 0.75] # normalized pose values for class boundaries (0.5 = 1σ, 0.75 = 1.5σ)
for i, label in enumerate(['X', 'Y', 'Z', 'A', 'B', 'C']):
    norm_values = df_processed[label + '_norm'].values
    
    # Class mapping with sign consideration (ordered from negative to positive):
    # Class 0: value <= -0.75 (beyond -1.5σ, extreme negative)
    # Class 1: -0.75 < value <= -0.5 (-1.5σ to -1σ, moderate negative)
    # Class 2: -0.5 < value < 0.5 (-1σ to +1σ, center/neutral region)
    # Class 3: 0.5 <= value < 0.75 (+1σ to +1.5σ, moderate positive)
    # Class 4: value >= 0.75 (beyond +1.5σ, extreme positive)
    
    # Default is class 2 (center region)
    class_labels = np.full_like(norm_values, 2, dtype=int)
    
    # Negative values
    class_labels[norm_values <= -normalized_class_boundaries[1]] = 0  # extreme negative
    class_labels[(norm_values <= -normalized_class_boundaries[0]) & (norm_values > -normalized_class_boundaries[1])] = 1  # moderate negative
    
    # Positive values  
    class_labels[(norm_values >= normalized_class_boundaries[0]) & (norm_values < normalized_class_boundaries[1])] = 3  # moderate positive
    class_labels[norm_values >= normalized_class_boundaries[1]] = 4  # extreme positive
    
    df_processed[label + '_class'] = class_labels 

# print summary statistics on class distribution
total_samples = len(df_processed)
class_descriptions = {
    0: "Extreme Negative (≤ -0.75, beyond -1.5σ)",
    1: "Moderate Negative (-0.75 to -0.5, -1.5σ to -1σ)",
    2: "Center/Neutral (-0.5 to 0.5, -1σ to +1σ)",
    3: "Moderate Positive (0.5 to 0.75, +1σ to +1.5σ)",
    4: "Extreme Positive (≥ 0.75, beyond +1.5σ)"
}

for label in ['X', 'Y', 'Z', 'A', 'B', 'C']:
    class_counts = df_processed[label + '_class'].value_counts().sort_index()
    print(f"Class distribution for {label}:")
    for class_label, count in class_counts.items():
        description = class_descriptions.get(class_label, f"Class {class_label}")
        print(f"  Class {class_label} ({description}): {count} samples")

        # print percentage of samples in each class
        percentage = (count / total_samples) * 100
        print(f"    ({percentage:.2f}%)")
        

# save processed data to csv
output_path = os.path.splitext(data_path)[0] + "_processed.csv"
df_processed.to_csv(output_path, index=False)   

import pdb; pdb.set_trace()