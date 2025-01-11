import numpy as np
import pandas as pd

df = pd.DataFrame(pd.read_excel("400212752_2024-09-03_CakeDetection.xls"))
#print(df)

print("---")
selection_of_df = df.filter(["Current", "imb", "StepIndx"])
print("selection of df:")
print(selection_of_df)

#print("---<<")
#print(df["imb"])

print("---<<---<<<")

#print(selection_of_df["Current"])

print("---<<---!")
selection_of_df = selection_of_df.drop(0)
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

