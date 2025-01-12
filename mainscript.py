import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def read_file(file_name):
    df = pd.DataFrame(pd.read_excel(file_name))
    return df

df = read_file("400212752_2024-09-03_CakeDetection.xls")
print("raw df:")
print(df)

# print(df.columns)
# print(df.head())
# #print("---")
# indexes_with_step_11 = df[df["StepIndx"] ==11].index
# df = df.drop(indexes_with_step_11)
# print(df["StepIndx"].unique())

# remove row 0 in dataframe = Row 2 in Excel with Text Descriptions
df = df[1:]

testdf = df[1:3]
print("testdf")
print(testdf)

# remove step 11 at beginning of excel list
indexes_with_step_11 = df[df["StepIndx"] ==11].index
df = df.drop(indexes_with_step_11)

# select parameters
selection_of_df = df.filter(["imb", "StepIndx", "Current", "PeelerMoveT", "MachStat", "density", "flow", "PlantActStepTime", "PlantSetStepTime", "Feedtime", "FeedPause", "FeedPulse", "PulseTimeFeedValve", "PauseTimeFeedValve"])
# print(selection_of_df.head())

# time in 5sec steps
size = len(selection_of_df)
start = 0
time = []
for i in range(size):
    if i ==0:
        time.append(i)
    else:
        time.append(time[i-1] + 5)

# add column time to dataframe
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
print("selection_np shape:")
print(selection_np.shape)
#
# current = selection_np[:,0]
# imbalance = selection_np[:,1]
step_index = selection_np[:,1]
# # print(current)
# # print(imbalance)
# # print(step_index)
#
# shape_of_step_index = step_index.shape
# print("Length of step index is: ", shape_of_step_index[0])
#

# switch from state 8->1 new batch starts
def find_batch_end(index_array):
    startpoints = []
    # i = 0, 1, 2, ...
    for i in range(index_array.shape[0]):
        if i > 0:
            if index_array[i]  == 1 and index_array[i-1]  == 8:
                # i+1 -> +1 numpy array indexes start at 0
                startpoints.append(i+1)
    return startpoints

startpoints_of_step_index = find_batch_end(step_index)
print("startpoints of batch (0), 1, 2, 3,..(can be compared to (Excel_row-381):")
print(startpoints_of_step_index)

# first batch j=0
# second batch j=1
def defineBatches(start, df, j):
    if j==0:
        batch = df[0:start[j]-1]
    else:
        # start[j-1] row a not included in selection df[a:b] -> -1
        batch = df[start[j-1]-1:start[j]-1]
    return batch


batch_number = 0
batch = defineBatches(startpoints_of_step_index, selection_of_df, batch_number)
print(f"Batch {batch_number}\n", batch)

batch_number = 1
batch = defineBatches(startpoints_of_step_index, selection_of_df, batch_number)
print(f"Batch {batch_number}\n", batch)

batch_number = 2
batch = defineBatches(startpoints_of_step_index, selection_of_df, batch_number)
print(f"Batch {batch_number}\n", batch)










