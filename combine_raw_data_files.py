import os 
import pdb 
import numpy as np 


data_dir = "/home/rp/Downloads/RCC/" 
files = [
    "RCC_mounted_data_collection_2013-01-11_01-09-56.csv", 
    "RCC_mounted_data_collection_2013-01-11_01-01-16.csv", 
    "RCC_mounted_data_collection_2013-01-11_00-51-41.csv", 
    "RCC_mounted_data_collection_2013-01-11_00-40-17.csv",
    "RCC_mounted_data_collection_2013-01-02_01-33-57.csv",
    "RCC_mounted_data_collection_2013-01-02_01-48-50.csv",
    "RCC_mounted_data_collection_2013-01-02_02-02-27.csv",
    "RCC_mounted_data_collection_2013-01-02_02-14-42.csv",
    "RCC_mounted_data_collection_2013-01-02_02-26-51.csv",
    "RCC_mounted_data_collection_2013-01-02_02-41-18.csv",
    "RCC_mounted_data_collection_2013-01-02_02-49-36.csv",
    "RCC_mounted_data_collection_2013-01-02_02-57-13.csv",
    "RCC_mounted_data_collection_2013-01-02_03-05-51.csv",
    "RCC_mounted_data_collection_2013-01-02_03-17-10.csv",
]

# read all csv files and concatenate them in order 
data_list = []
for file in files:
    file_path = os.path.join(data_dir, file)
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    data_list.append(data)  
data = np.concatenate(data_list, axis=0)

# save as csv with header from first file
# read header from first file
header = []
with open(os.path.join(data_dir, files[0]), 'r') as f:
    header = f.readline().strip()
np.savetxt("./data/RCC_combined_14.csv", data, delimiter=',', header=header, comments='')
