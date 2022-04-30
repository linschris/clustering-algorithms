from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as sKMeans
from matplotlib import pyplot as plt
import numpy as np
# from implementations.FuzzyCMeans import FuzzyCMeans
from implementations.KMeans import KMeans

def cluster_blobs(num_datapoints=100, num_clusters=4):
    '''Step 1: Create the (blobs of) datapoints '''
    # make_blobs returns the generated samples (numpy array of 2D points) as well as the clusters they're in (int array)
    centers = [[1000,1000], [-1000, -1000], [0, 0]] # Predefined clusters

    data, clusters = make_blobs(n_samples=num_datapoints, centers=centers, cluster_std=2)
    
    # Plot the data
    plt.scatter(data[:,0],data[:,1]) # ",0" means slicing the data by datapoint and grabbing the 0th column (or x).
    plt.xlabel('x'),plt.ylabel('y')
    plt.title('Datapoints before clustering')
    # plt.show()

    '''Step 2: Cluster them with a given clustering algorithm: sKMeans works best.'''
    '''If you want a better understanding of KMeans, navigate to code/examples/implementations/KMeans.py'''

    # Clustering with the implemented algorithm of KMeans.py
    # Current issue is it won't on > 2 clusters, as the groupby() method fails

    # centroids, clusters = KMeans(data, k=num_clusters)
    # clusters = clusters.astype(int)


    # datapoint_clusters =[[] for i in range(num_clusters)]
    # for index, datapoint in enumerate(data):
    #     # Store the data points of each cluster
    #     datapoint_clusters[clusters[index]].append(datapoint)
    

    # plot_data(datapoint_clusters, num_clusters)

    # Clustering with sklearn's algorithm
    kmeans = sKMeans(n_clusters=num_clusters, random_state = 0).fit(data)

    '''Step 3: Create an 3-D array of k (num_clusters) rows, with each row containing all the datapoints in that cluster.'''
    # Cluster labels for each datapoint stored in labels_
    clusters = [[] for i in range(num_clusters)]
    # Append datapoints to each cluster
    for index, datapoint in enumerate(data):
        clusters[kmeans.labels_[index]].append(list(datapoint))
    
    '''Step 4: Plot the datapoints '''

    plot_data(clusters, num_clusters)

def plot_data(datapoint_clusters, num_clusters):
    '''For each row in our 3-D array, it plots them on a X-Y scatter plot of one color.'''
    for i in range(num_clusters):
        datapoint_clusters[i] = np.array(datapoint_clusters[i])
        plt.scatter(datapoint_clusters[i][:,0],datapoint_clusters[i][:,1]) # X-Y scatter plot
        datapoint_clusters[i] = datapoint_clusters[i].tolist() 
    plt.xlabel('x'),plt.ylabel('y')
    plt.title('Datapoints after K-Means clustering')
    plt.show()


if __name__ == "__main__":
    cluster_blobs(num_datapoints=10, num_clusters=2)
