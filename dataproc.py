#!/usr/bin/env python
# coding: utf-8

# In[9]:


import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
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
    
    if show_graphs:
        # webcam res (0, x, 0, y)
        res_x, res_y = res
        plt.axis([0, res_x, 0, res_y])
        plt.scatter(xs, ys)#,label=ID)
        #plt.legend()
    
        plt.show() # creates diff. graphs
    
    return (xs, ys, ID)


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


def map_to_plane(x, y):
    ### CALIBRATION ###
    # provide points from image 1
    pts_src = np.float32([154, 174, 702, 349, 702, 572,1, 572]).reshape(4, 1, -1)
    # corresponding points from image 2 (i.e. (154, 174) matches (212, 80))
    pts_dst = np.float32([212, 80,489, 80,505, 180,367, 235]).reshape(4, 1, -1)
    
    # calculate matrix H
    h = cv2.findHomography(pts_src, pts_dst)[0]
    ### END CALIBRATION ###

    # provide a point you wish to map from image 1 to image 2
    # print(x, y)
    a = np.array([[[x, y]]], dtype='float32')
    # print(a.shape)
    
    # finally, get the mapping
    pointsOut = cv2.perspectiveTransform(a, h)
    # print(pointsOut)
    return pointsOut[0][0][0], pointsOut[0][0][1]

def calculate_heatmap(resulting_data):
    print('calculating heatmap')

    total_entries = 0
    for entry in resulting_data:
        x_set, y_set, set_id = entry # a set of xs, ys, and ID for each person in a frame
        total_entries += len(x_set)

    print(str(total_entries) + ' entries')
    x_data = np.zeros((total_entries))
    y_data = np.zeros((total_entries))

    # print(x_data.shape)

    i = 0
    for result in resulting_data:
        x_points, y_points, person_id = result
        
        j = 0
        for x in x_points:
            #x_mapped, y_mapped = map_to_plane(x, y_points[j])
            x_mapped = x
            y_mapped = int(sys.argv[3]) - y_points[j] # FLIP Y
            # print(x_mapped)
            # print(y_mapped)
            x_data[i] = x_mapped
            y_data[i] = y_mapped
            i += 1
            j += 1

    #print(x_data)
    # call the kernel density estimator function
    #ax = sns.kdeplot(x_data, y_data, cmap="Blues", shade=True, shade_lowest=False)
    # the function has additional parameters you can play around with to fine-tune your heatmap, e.g.:
    ax = sns.kdeplot(x_data, y_data, kernel="gau", bw = 25, cmap="Reds", n_levels = 50, shade=True, shade_lowest=False, gridsize=100)
    
    # plot your KDE
    ax.set_frame_on(False)
    plt.xlim(0, 704)
    plt.ylim(576, 0)
    plt.axis('off')
    #ax.get_legend().remove()
    plt.show()
    print('showing')
    
    # save your KDE to disk
    fig = ax.get_figure()
    fig.savefig('kde.png', transparent=True, bbox_inches='tight', pad_inches=0)


# In[10]:
import sys
import seaborn as sns
from scipy import stats, integrate

heatmap = False
if len(sys.argv) == 5:
    if(sys.argv[1]) == 'heatmap':
        heatmap = True
else:
    print('usage: dataproc.py  heatmap? [heatmap]    res_x [int]   res_y [int]   graphs [bool]')
    sys.exit(2)

unique_graphs = sys.argv[4].lower() == "true"
resolution = (int(sys.argv[2]), int(sys.argv[3]))

data = load_data() # loads data
results = plot_ids(unique_ids(), unique_graphs, resolution) # finds all unique tracking IDs and plots them. True = Seperate Graphs, False = Single Graph

calculate_heatmap(results)

if not heatmap :
    for result in results:
        xs, ys, ID = result
        saved_arr = np.vstack((xs, ys))
        # print(saved_arr)
        filename = str(ID) + '.csv'
        np.savetxt('results/' + filename, saved_arr, delimiter=',')
        print('Saved ' + filename + ' to results/' + filename)
