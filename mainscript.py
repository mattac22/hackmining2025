# Hackathon HackMining2025 Team Members: Christian Sinn, Srishti Mishra, Chuxin Yuan, Zhishan Wang

# potential error in feedtimes (line 113-125) -> peaks (at batch with rinsing(step 10)) in final plot should not occur
# because the duration of step 2 is counted.

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

# remove row 0 in dataframe = Row 2 in Excel with Text Descriptions
df = df[1:]

# remove step 11 at beginning of excel list
indexes_with_step_11 = df[df["StepIndx"] == 11].index
# print("indexes_with_step_11")
# print(indexes_with_step_11)
# print("indexes_with_step_11 length")
# print(len(indexes_with_step_11))
df = df.drop(indexes_with_step_11)

# select parameters
selection_of_df = df.filter(["imb", "StepIndx", "Current", "PeelerMoveT", "MachStat", "density", "flow", "PlantActStepTime", "PlantSetStepTime", "Feedtime", "FeedPause", "FeedPulse", "PulseTimeFeedValve", "PauseTimeFeedValve"])
# print(selection_of_df.head())

# time series in 5sec steps
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

#scaling for corr heat map
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

print("selection_np shape:")
print(selection_np.shape)

# step column
step_index = selection_np[:,1]
# shape_of_step_index = step_index.shape

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

# calculate total time of a step in a specific batch(batch is a dataframe)
def calculateStepTime(df, stepNr):
    indexes = df[df["StepIndx"] == stepNr].index
    time_of_step = 5*len(indexes)
    return time_of_step


#calculate trend of feedtime over multiple batches
feedtimes = []
num_batches = len(startpoints_of_step_index)+1

for batch_nr in range(num_batches-1):
    batch = defineBatches(startpoints_of_step_index, selection_of_df, batch_nr)
    #print(f"Batch {batch_nr}\n", batch)

    # calculate steptime of step 2 filling
    steptime_st2 = calculateStepTime(batch, 2)
    feedtimes.append(steptime_st2)

print("feedtimes")
print(feedtimes)

print(len(feedtimes))
print(range(1, len(feedtimes)+1, 1))

#plot of feedtimes (blue curve with circles)
x_ax = range(1, len(feedtimes)+1)
y_ax = feedtimes
plt.plot(x_ax, y_ax, marker='o', linestyle='-')
# fig, ax = plt.plot()
# ax.plot(x_ax, y_ax)
#plt.show()

# calculate intervals with multiple batches in a row seperated by cleaning step(-1)
modified_feedtimes = feedtimes
for j in range(len(feedtimes)):
    if feedtimes[j] > 120:
        modified_feedtimes[j] = -1

print("----   ------     -----     -----")
#interval_storage = pd.DataFrame()

interval_start = []
interval_start.append(0)
interval_end = []
found_peak = 0
col = 0
for k in range(len(modified_feedtimes)):
   if modified_feedtimes[k] == -1:
       found_peak += 1
       # k not included in [_:k]
       y_int = modified_feedtimes[interval_start[found_peak-1]:k]
       print(y_int)
       #interval_storage[f"{col}"] = y_int
       #col += 1
       x_int = range(interval_start[found_peak-1]+1,k+1)
       print(x_int)
       #interval_storage[f"{col}"] = x_int
       #col += 1
       interval_end.append(k-1)
       if k < len(modified_feedtimes)-1:
           interval_start.append(k+1)
interval_end.append(len(modified_feedtimes)-1)

#print(interval_storage)
print("start")
print(interval_start)
print("end")
print(interval_end)

# for every interval (with multiple batches) calculate a linear approximation of points
for intvl in range(7):
    x_plt = x_ax[interval_start[intvl]:interval_end[intvl]+1]
    print("x")
    print(x_plt)
    y_plt = modified_feedtimes[interval_start[intvl]:interval_end[intvl]+1]
    print("y")
    print(y_plt)

    # calculate linear(1) approximation
    z_plt = np.polyfit(x_plt, y_plt, 1)
    f_plt = np.poly1d(z_plt)

    x_plot = x_plt
    y_plot = f_plt(x_plot)
    plt.title("Feed time trend")
    plt.xlabel("Batch number")
    plt.ylabel("Duration of Feed step 2")
    plt.plot(x_plot, y_plot, linewidth=3)

# show batch 11 on command line
batch = defineBatches(startpoints_of_step_index, selection_of_df, 11-1)
print("batch 11:")
print(batch)

plt.show()




