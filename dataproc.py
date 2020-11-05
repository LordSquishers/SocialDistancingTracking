#!/usr/bin/env python
# coding: utf-8

# In[9]:


import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_data():
    file = open("inference/output/results.txt", "r") 
    locations = defaultdict(list)
    
    for line in file: # loads results.txt into list by ID with entries (frame, bounding box coords)
        splits = line.split()
        locations[splits[1]].append((int(splits[0]), int(splits[2]), int(splits[3]), int(splits[4]), int(splits[5])))
    
    return locations


def calculate_centerpoint(x1, y1, x2, y2): # does what it says on the tin
    return ((x1 + x2) / 2, (y1 + y2) / 2)

 
def calculate_bottom_center(x1, y1, x2, y2):
    return ((x1 + x2) / 2, y2)


def process_centers(ID): # gets individual ID box data. converts to np array [frame, x, y]
    single_tracking_data = data[str(ID)]
    points = np.zeros((len(single_tracking_data), 3))
    
    idx = 0
    for frame_data in single_tracking_data:
        # hey! this variable is named center, but may not actually be the center.
        # I recommend checking what ctr is equal to instead.
        ctr = calculate_bottom_center(frame_data[1], frame_data[2], frame_data[3], frame_data[4])
        points[idx][0] = frame_data[0]
        points[idx][1] = ctr[0]
        points[idx][2] = ctr[1]
        idx += 1
    
    return points


def plot_centerpoints(ID, show_graphs, res): 
    # takes [frame x, y] which should have been formatted correctly and formats them correctly for pyplot.
    # ignores frame time.
    points = process_centers(ID)
    xs = np.zeros(points.shape[0])
    ys = np.zeros(points.shape[0])
    x_res, y_res = resolution
    
    idx = 0
    for point in points:
        xs[idx] = point[1]
        ys[idx] = y_res - point[2]
        idx += 1
    
    # webcam res (0, x, 0, y)
    res_x, res_y = res
    plt.axis([0, res_x, 0, res_y])
    plt.plot(xs, ys, label=ID)
    plt.legend()
    
    if show_graphs:
        plt.show() # creates diff. graphs
    
    return (xs, ys)


def unique_ids(): # iterates through data file for all dict keys (IDs)
    all_ids_present = np.zeros(len(data))
    idx = 0
    for key, value in data.items():
        all_ids_present[idx] = key
        idx += 1
        if idx >= len(data):
            break
    
    # np.unique outputs float array, so turn to ints for indexes.
    return np.array(np.unique(all_ids_present), dtype=np.int16)


def plot_ids(IDs, show_graphs, res): # plots an array of IDs.
    all_id_data = list()
    for ID in IDs:
        all_id_data.append(plot_centerpoints(ID, show_graphs, res))
    
    return all_id_data


# In[10]:
import sys

if len(sys.argv) != 4:
    print('usage: dataproc.py   res_x [int]   res_y [int]   show_graphs [bool]')
    sys.exit(2)

show_graphs = sys.argv[3].lower() == "true"
resolution = (int(sys.argv[1]), int(sys.argv[2]))

data = load_data() # loads data
results = plot_ids(unique_ids(), show_graphs, resolution) # finds all unique tracking IDs and plots them. True = Seperate Graphs, False = Single Graph

current_ID = 1
for result in results:
    xs, ys = result
    saved_arr = np.vstack((xs, ys))
    # print(saved_arr)
    filename = str(current_ID) + '.csv'
    np.savetxt('results/' + filename, saved_arr, delimiter=',')
    print('Saved ' + filename + ' to results/' + filename)
    current_ID += 1

# In[ ]:




