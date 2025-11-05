import os 
import pdb 
import numpy as np 


data_dir = "/home/rp/Downloads/RCC/" 
files = [
    "/home/rp/abhay_ws/RCC_modeling/FTS/data/kuka/RCC_new_compliance_data_collection_2013-01-01_09-18-06.csv",
    "/home/rp/abhay_ws/RCC_modeling/FTS/data/kuka/RCC_new_compliance_data_collection_2013-01-01_09-31-48.csv",
    "/home/rp/abhay_ws/RCC_modeling/FTS/data/kuka/RCC_new_compliance_data_collection_2013-01-01_09-52-23.csv",
    "/home/rp/abhay_ws/RCC_modeling/FTS/data/kuka/RCC_new_compliance_data_collection_2013-01-01_10-05-06.csv",
    "/home/rp/abhay_ws/RCC_modeling/FTS/data/kuka/RCC_new_compliance_data_collection_2013-01-01_10-17-31.csv",
    "/home/rp/abhay_ws/RCC_modeling/FTS/data/kuka/RCC_new_compliance_data_collection_2013-01-01_10-44-06.csv",
    "/home/rp/abhay_ws/RCC_modeling/FTS/data/kuka/RCC_new_compliance_data_collection_2013-01-01_10-54-48.csv",
    "/home/rp/abhay_ws/RCC_modeling/FTS/data/kuka/RCC_new_compliance_data_collection_2013-01-01_11-07-27.csv",
    "/home/rp/abhay_ws/RCC_modeling/FTS/data/kuka/RCC_new_compliance_data_collection_2013-01-01_11-30-27.csv",
    "/home/rp/abhay_ws/RCC_modeling/FTS/data/kuka/RCC_new_compliance_data_collection_2013-01-01_11-40-27.csv",
    "/home/rp/abhay_ws/RCC_modeling/FTS/data/kuka/RCC_new_compliance_data_collection_2013-01-01_12-20-16.csv",
    "/home/rp/abhay_ws/RCC_modeling/FTS/data/kuka/RCC_new_compliance_data_collection_2013-01-01_12-37-30.csv",
    "/home/rp/abhay_ws/RCC_modeling/FTS/data/kuka/RCC_new_compliance_data_collection_2013-01-01_12-46-32.csv",
    "/home/rp/abhay_ws/RCC_modeling/FTS/data/kuka/RCC_new_compliance_data_collection_2013-01-01_13-02-28.csv",
    "/home/rp/abhay_ws/RCC_modeling/FTS/data/kuka/RCC_new_compliance_data_collection_2013-01-01_13-09-17.csv",
]

# read all csv files and concatenate them in order 
data_list = []
for i, file in enumerate(files, start=1):
    file_path = file if os.path.isabs(file) else os.path.join(data_dir, file)
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    # Append trial number column (1-based index corresponding to file order)
    trial_col = np.full((data.shape[0], 1), i, dtype=int)
    data = np.hstack([data, trial_col])
    data_list.append(data)
combined = np.concatenate(data_list, axis=0)

# save as csv with header from first file
# read header from first file
header = []
first_file = files[0]
first_path = first_file if os.path.isabs(first_file) else os.path.join(data_dir, first_file)
with open(first_path, 'r') as f:
    header = f.readline().strip()
# Append trial to header if not already present
if 'trial' not in [h.strip() for h in header.split(',')]:
    header = header + ',trial'
np.savetxt("./data/RCC_kuka_15_trials.csv", combined, delimiter=',', header=header, comments='')
