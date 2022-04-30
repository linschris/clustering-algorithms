from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import skfuzzy as fuzz
import numpy as np

def cluster_blobs(num_datapoints=100, num_clusters=3):
    # make_blobs returns the generated samples (numpy array of 2D points) as well as the clusters they're in (int array)
    centers = [[1000,1000], [-1000, -1000], [0, 0]] # Predefined clusters

    data, clusters = make_blobs(n_samples=num_datapoints, centers=centers, cluster_std=10)
    
    # Plot the data
    plt.scatter(data[:,0],data[:,1]) # ",0" means slicing the data by datapoint and grabbing the 0th column (or x).
    plt.xlabel('x'),plt.ylabel('y')
    plt.title('Datapoints before clustering')
    plt.show()

    # TODO: Clustering with the implemented algorithm of KMeans.py with k = 2 clusters
    # centroids, clusters = KMeans(data, k=num_clusters)

    # Clustering with sklearn's algorithm
    kmeans = KMeans(n_clusters=num_clusters, random_state = 0).fit(data)
    # Cluster labels for each datapoint stored in labels_
    # print(data)
    # print(kmeans.labels_)
    clusters = [[] for i in range(num_clusters)]
    # Append datapoints to each cluster
    for index, datapoint in enumerate(data):
        clusters[kmeans.labels_[index]].append(list(datapoint))
    
    plot_data(clusters, num_clusters)


def plot_data(datapoint_clusters, num_clusters):
    for i in range(num_clusters):
        datapoint_clusters[i] = np.array(datapoint_clusters[i])
        plt.scatter(datapoint_clusters[i][:,0],datapoint_clusters[i][:,1])
        datapoint_clusters[i] = datapoint_clusters[i].tolist() 
    plt.xlabel('x'),plt.ylabel('y')
    plt.title('Datapoints after K-Means clustering')
    plt.show()










if __name__ == "__main__":
    import sys    
    # print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)
    print(dir())
    print(__name__)
    cluster_blobs(num_datapoints=10000)
