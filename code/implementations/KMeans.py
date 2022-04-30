import numpy as np
import pandas as pd

def KMeans(data, k=2):
    '''
        Originally curated by Khushijain

        Link: https://medium.com/nerd-for-tech/k-means-python-implementation-from-scratch-8400f30b8e5c

        Parameters:
            - data: a numpy 2D array of points
                - Unfortunately, as the euclidean distance is calculated using x^2 + y^2, this will only work for two dimensions.
            - k (int, default=2): number of clusters, to separate data into
        
        Outputs:
            - centroids: a 2D array of X-Y coordinates of centers of clusters
            - clusters: an array where each ith element (i.e datapoint) corresponds
                        to a specific cluster (0, 1, ...)
    '''
    # This will store what datapoints correspond to what cluster.
    clusters = np.zeros(data.shape[0]) # data.shape[0] = the # of dimensions (x, y) of a datapoint (data[0])

    # Select the random K centroids (of clusters) from the data.
    # Not in the tutorial, but converted to dataframe for useful sample function to do this.
    centroids = pd.DataFrame(data).sample(n=k).values

    # While the cluster center's position have NOT changed
    diff = 1
    while diff:
        minimum_centroid_dist = float('inf')
        for i, datapoint in enumerate(data):
            # Calculate the distance to each centroid using Euclidean distance
            for j, centroid in enumerate(centroids):
                # dist = sqrt(x^2 + y^2) 
                # x = centroid.x - this.x
                # y = centroid.y - this.y
                dist_to_centroid = np.sqrt((centroid[0] - datapoint[0]) ** 2 + (centroid[1] - datapoint[1]) ** 2)
                if minimum_centroid_dist > dist_to_centroid:
                    minimum_centroid_dist = dist_to_centroid
                    clusters[i] = j # The closest cluster for each datapoint is stored in this array.
        # Compute the new clusters by computing the mean Xs and Ys of each subsequent data point in that cluster
        new_centroids = pd.DataFrame(data).groupby(by=clusters).mean().values
        if np.count_nonzero(centroids - new_centroids) == 0:
            # The cluster positions haven't changed 
            diff = 0
        else:
            centroids = new_centroids
    return centroids, clusters

