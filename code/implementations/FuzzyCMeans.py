import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
import cv2


def FuzzyCMeans(data, k=2, f=2):
    '''
        Fuzzy C Means Clustering Algorithm
        Original source: https://www.youtube.com/watch?v=FA-hJBu5Bkc by The Academician

        Inputs:
            - data: numpy array of dimension n * d
                - n: number of datapoints
                - d: number of dimensions of a datapoint (x, y, z, ...)
            - k: (int, default=2) number of clusters
            - f: (int, default=2) "fuzziness" or softness of clustering
                - As f -> infinity, more cluster "sharing" (datapoints being in many clusters) occurs

        Outputs:
            - centers (numpy array): positions of the clusters
            - datapoint_clusters: datapoints and their corresponding clusters
                - numpy array of dimension n * (d+1) as the extra dimension holds the number of the cluster
    '''

    num_datapoints = len(data)
    num_dimensions = len(data[0]) # x, y, z, ...
    mem_values = np.random.dirichlet(np.ones(k), num_datapoints) #dirichlet --> make random mem_values that add to 1 basically
    centers = np.zeros((k, num_dimensions)) # initalize k centers with same dimensions as datapoints, all 0s

    # Calculate centroids
    for j in range(k):
        # Sum of all membership values (from datapoints) for a cluster j
        mem_sum = sum(np.power(mem_values[:,j], f)) 
        data_mem_sum = 0
        for i in range(num_datapoints):
            # Multiplying the membership values of the datapoint by the datapoint's x, y values
            dp_sum = np.multiply(np.power(mem_values[i, j], f), data[i, :])
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
        # divided by the total distance to all clusters from the same datapoint
        for j in range(k):
            new_weight = np.power((1/distance.euclidean(centers[j, 0:num_dimensions], data[i, 0:num_dimensions])), 1/(f-1)) / total_dist
            mem_values[i,j] = new_weight
    # Decide on a datapoint's primary cluster based on these updated values
    addZeros = np.zeros((num_datapoints, 1))
    datapoint_clusters = np.append(data, addZeros, axis=1)
    for i in range(num_datapoints):
        cluster_num = np.where(mem_values[i] == np.amax(mem_values[i]))
        datapoint_clusters[i, num_dimensions-1] = cluster_num[0]
    return centers, datapoint_clusters
            


    


        
    



# k = 4
# data = np.random.randint(100, size=(100,4)) # size=(m, n) => m samples of n size are pulled 
# # data = np.array([
# #         [1,1,2,1], 
# #         [2,1,2,3], 
# #         [2,2,4,5], 
# #         [50,42,2,83],
# #         [51,43,1,82],
# #         [51,44,3,89],
# #         [53,40,8,80]])
# # data = np.array([[1, 2], [3, 4]])
# # new_data = FuzzyCMeans(data, k)
# img = cv2.imread("corgi-tree.jpeg", cv2.IMREAD_UNCHANGED)
# reshaped_img = img.reshape((-1,3))
# centers, new_data, img_data = FuzzyCMeans(reshaped_img, k)
# centers = np.uint8(centers)
# print(centers)
# # print(centers.flatten())
# print(np.uint(new_data[:, 3]).flatten())
# segmented_image = centers[np.uint8(new_data[:, 3]).flatten()]
# print(segmented_image)
# segmented_image = segmented_image.reshape((img.shape))
# COLOR_TABLE = {0:'r', 1:'b', 2:'p'}
# plt.imshow(segmented_image)
# plt.pause(100)


