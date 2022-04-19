import random
import numpy as np
from scipy.spatial import distance

def FuzzyCMeans(data, k):
    num_datapoints = len(data)
    num_dimensions = len(data[0])
    f = 4 # fuzzy (how hard or soft clustering is)
    m = 1
    weights = np.random.dirichlet(np.ones(k), num_datapoints) #dirichlet --> make random nums that add to 1 basically
    centers = np.zeros((k, num_dimensions)) # initalize centers -> 0

    # Calculate centroids
    for j in range(k):
        # Sum of all membership values (from datapoints) for a cluster j
        mem_sum = sum(np.power(weights[:,j], m)) 
        data_mem_sum = 0
        for i in range(num_datapoints):
            # Multiplying the membership values of the datapoint by the datapoint's x, y values
            dp_sum = np.multiply(np.power(weights[i, j], m), data[i, :])
            data_mem_sum += dp_sum
        centroid_pos = data_mem_sum / mem_sum
        centers[j] = np.reshape(centroid_pos, num_dimensions) # Update centers positions
    
    # Recalculate the membership values
    for i in range(num_datapoints):
        # Calculate the total distance to ALL clusters (using Euclidean distance)
        total_dist = 0
        for j in range(k):
            total_dist += np.power(1/distance.euclidean(centers[j, 0:num_dimensions], data[i, 0:num_dimensions]), 1/(f-1))
        # New membership value is equal to the euclidean distance from a datapoint i to cluster j
        # divided by the total distance to all clusters
        for j in range(k):
            new_weight = np.power((1/distance.euclidean(centers[j, 0:num_dimensions], data[i, 0:num_dimensions])), 1/(f-1)) / total_dist
            weights[i,j] = new_weight
        
    # Decide on a datapoint's primary cluster based on these updated values
    addZeros = np.zeros((num_datapoints, 1))
    data = np.append(data, addZeros, axis=1)
    for i in range(num_datapoints):
        cluster_num = np.where(weights[i] == np.amax(weights[i]))
        data[i, num_dimensions] = cluster_num[0]
    
    return data
    


            
        

    


        
    



k = 2
data = np.random.randint(100, size=(4,2)) # size=(m, n) => m samples of n size are pulled 
data = np.array([
        [1,1,2,1], 
        [2,1,2,3], 
        [2,2,4,5], 
        [50,42,2,83],
        [51,43,1,82],
        [51,44,3,89],
        [53,40,8,80]])
new_data = FuzzyCMeans(data, k)
print(new_data)

