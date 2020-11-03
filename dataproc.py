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


def process_centers(ID): # gets individual ID box data. converts to np array [frame, x, y]
    single_tracking_data = data[str(ID)]
    points = np.zeros((len(single_tracking_data), 3))
    
    idx = 0
    for frame_data in single_tracking_data:
        ctr = calculate_centerpoint(frame_data[1], frame_data[2], frame_data[3], frame_data[4])
        points[idx][0] = frame_data[0]
        points[idx][1] = ctr[0]
        points[idx][2] = ctr[1]
        idx += 1
    
    return points


def plot_centerpoints(ID, seperate_graph, res): 
    # takes [frame x, y] which should have been formatted correctly and formats them correctly for pyplot.
    # ignores frame time.
    points = process_centers(ID)
    xs = np.zeros(points.shape[0])
    ys = np.zeros(points.shape[0])
    
    idx = 0
    for point in points:
        xs[idx] = point[1]
        ys[idx] = point[2]
        idx += 1
    
    # webcam res (0, x, 0, y)
    res_x, res_y = res
    plt.axis([0, res_x, 0, res_y])
    plt.plot(xs, ys, label=ID)
    plt.legend()
    
    if seperate_graph:
        plt.show() # creates diff. graphs


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


def plot_ids(IDs, seperate, res): # plots an array of IDs.
    for ID in IDs:
        plot_centerpoints(ID, seperate, res)
    
    if not seperate:
        plt.show()


# In[10]:
import sys

if len(sys.argv) != 4:
    print('usage: dataproc.py   res_x [int]   res_y [int]   unique_graphs [bool]')
    sys.exit(2)

unique_graphs = sys.argv[3].lower() == "true"
resolution = (int(sys.argv[1]), int(sys.argv[2]))

data = load_data() # loads data
plot_ids(unique_ids(), unique_graphs, resolution) # finds all unique tracking IDs and plots them. True = Seperate Graphs, False = Single Graph


# In[ ]:




