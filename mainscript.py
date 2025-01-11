import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def read_file(file_name):

    df = pd.DataFrame(pd.read_excel(file_name))

    return df

df = read_file("400212752_2024-09-03_CakeDetection.xls")

# print(df.columns)
# print(df.head())
# #print("---")
# indexes_with_step_11 = df[df["StepIndx"] ==11].index
# df = df.drop(indexes_with_step_11)
# print(df["StepIndx"].unique())
df = df[382:469]
selection_of_df = df.filter(["imb", "StepIndx", "Current", "PeelerMoveT", "MachStat", "density", "flow", "PlantActStepTime", "PlantSetStepTime", "Feedtime", "FeedPause", "FeedPulse", "PulseTimeFeedValve", "PauseTimeFeedValve"])
# print(selection_of_df.head())
size = len(selection_of_df)
start = 0
time = []
for i in range(size):
    if i ==0:
        time.append(i)
    else:
        time.append(time[i-1] + 5)
#
selection_of_df["time"] = time

selection_np = selection_of_df.to_numpy()
# selection_of_df.plot(x="time", y= ["imb", "StepIndx", "Current", "PeelerMoveT", "MachStat", "density", "flow", "PlantActStepTime", "PlantSetStepTime", "Feedtime", "FeedPause", "FeedPulse", "PulseTimeFeedValve", "PauseTimeFeedValve"], kind="line", figsize=(10,10), label=['Imbalance of batch 1', 'StepIndx of batch 1'])

scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(selection_of_df), columns=selection_of_df.columns)

# Preview the standardized data
print("Standardized-------")
# print(df_standardized["MachStat"])

potentially_zero = df_standardized["MachStat"].to_numpy()
print("!!!!!!")
print(np.count_nonzero(potentially_zero))

corr_matrix = df_standardized.corr()

# # Plot the heatmap
# plt.figure(figsize=(10, 8))  # Adjust figure size as needed
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Heatmap of Standardized Variables')
# plt.show()


# plt.grid()
# plt.show()

#
# #print("---<<")
# #print(df["imb"])
#
# #print("---<<---<<<")
#
# #print(selection_of_df["Current"])
#
# #print("---<<---!")
# selection_of_df = selection_of_df.drop(0)
# #print(selection_of_df)
#

#
# #print("---<<---!!!!!")
# #print(selection_np)
# #print(selection_np.shape)
#
# current = selection_np[:,0]
# imbalance = selection_np[:,1]
step_index = selection_np[:,2]
# # print(current)
# # print(imbalance)
# # print(step_index)
#
# shape_of_step_index = step_index.shape
# print("Length of step index is: ", shape_of_step_index[0])
#

def find_batch_end(index_array):
    endpoints = []
    for i in range(index_array.shape[0]):
        if index_array[i]  == 1 and index_array[i-1]  == 8:
            endpoints.append(i)
    return endpoints

endpoints_of_step_index = find_batch_end(step_index)
# print(endpoints_of_step_index)










