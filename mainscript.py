import numpy as np
import pandas as pd


df = pd.DataFrame(pd.read_excel("400212752_2024-09-03_CakeDetection.xls"))
#print(df)

#eliminate first row
df = df.drop(0)

# df = df.drop(["Timestamp"], axis=1)
# def histogram_intersection(a, b):
#     v = np.minimum(a, b).sum().round(decimals=1)
#     return v
# print("<<<<<<<<<<<<<<<<<<<")
# print(df.corr(method=histogram_intersection))
# print("<<<<<<<<<<<<<<<<<<<")

# eliminate step 11
indexes_with_step_11 = df[df["StepIndx"] == 11].index
df = df.drop(indexes_with_step_11)
print(df.head())

#create time steps in seconds
shape_of_df = df.shape
print(shape_of_df)
time = []
for x in range(shape_of_df[0]):
    time.append(5*x)
#print(time)


print("---")
selection_of_df = df.filter(["Current", "imb", "StepIndx"])
print("selection of df:")
print(selection_of_df)

selection_np = selection_of_df.to_numpy()

print("---<<---!!!!!")
print(selection_np)
print(selection_np.shape)

current = selection_np[:,0]
imbalance = selection_np[:,1]
step_index = selection_np[:,2]
print(current)
print(imbalance)
print(step_index)

shape_of_step_index = step_index.shape
print(shape_of_step_index[0])

def find_batch_end(index_array):
    endpoints = []
    for i in range(index_array.shape[0]):
        if index_array[i]  == 1 and index_array[i-1]  == 8:
            endpoints.append(i)
    return endpoints

endpoints_of_step_index = find_batch_end(step_index)
print(endpoints_of_step_index)
print("len endpoints")
print(len(endpoints_of_step_index))

print("----")

def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v
print("<<<<<<<<<<<<<<<<<<<")
print(selection_of_df.corr(method=histogram_intersection))
print("<<<<<<<<<<<<<<<<<<<")

