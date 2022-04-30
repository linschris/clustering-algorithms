from unicodedata import digit
from sklearn.datasets import load_iris, load_digits
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt


def cluster_digits(num_clusters=10, show_clusters=True):
    ''' Load the digits dataset '''
    digits = load_digits()
    data = digits.data 
    data = 255-data # Invert the colors to make it easier to cluster.

    kmeans = KMeans(n_clusters=num_clusters, init="random")
    kmeans.fit(data)
    Z = kmeans.predict(data) # Predicts closest cluster

    ''' Plots and show the predicted cluster for the digits '''
    if show_clusters:
        for i in range(0,num_clusters):
            row = np.where(Z==i)[0]  # row in Z for elements of cluster i
            num = row.shape[0]       #  number of elements for each cluster
            r = np.floor(num/10.)    # number of rows in the figure of the cluster 

            print("cluster "+str(i))
            print(str(num)+" elements")

            plt.figure(figsize=(10,10))
            for k in range(0, num):
                # print(int(r+1, k+1)
                plt.subplot(int(r+1), 10, k+1)
                image = data[row[k], ]
                image = image.reshape(8, 8)
                plt.imshow(image, cmap='gray')
                plt.axis('off')
                plt.show()
    
    return data, kmeans.labels_ # These are returned to be used for evaluation in evaluating_clusters.py

def cluster_iris(num_clusters = 3, show_clusters=True):
    ''' Load the digits dataset '''
    data = load_iris()
    data = data.data

    kmeans = KMeans(n_clusters=num_clusters, init="random")
    kmeans.fit(data)
    Z = kmeans.predict(data)

    ''' Plots and show the predicted cluster for the digits '''
    if show_clusters:
        for i in range(0,num_clusters):
            row = np.where(Z==i)[0]  # row in Z for elements of cluster i
            num = row.shape[0]       #  number of elements for each cluster
            r = np.floor(num/10.)    # number of rows in the figure of the cluster 

            print("cluster "+str(i))
            print(str(num)+" elements")

            plt.figure(figsize=(10,10))
            for k in range(0, num):
                # print(int(r+1, k+1)
                plt.subplot(int(r+1), 10, k+1)
                image = data[row[k], ]
                image = image.reshape(2, 2) # The size of the images is 4
                plt.imshow(image, cmap='rainbow') # See the colors clearly (unlike in digits)
                plt.axis('off')
                plt.show()
    
    return data, kmeans.labels_ # These are returned to be used for evaluation in evaluating_clusters.py






    


if __name__ == "__main__":
    # cluster_digits()
    cluster_iris()