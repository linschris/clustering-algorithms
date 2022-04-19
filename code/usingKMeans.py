'''
    KMeans Code
    Originated from Aktas's Tutorial: https://towardsdatascience.com/image-segmentation-with-clustering-b4bbc98f2ee6
'''

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
from matplotlib import pyplot as plt
import cv2
import pandas as pd

''' Understanding make_blobs, and how data is formatted at the moment'''

# Can utilize handcoded data
data = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])
data = np.array([[1, 5], [3, 1], [10, 3], [10, 2], [10, 1], [1, 0], [2, 15], [0.5, 4.9], [5, 3], [7, 13], [18, 18], [1.9, 0.5]])

# Or, for simplicity utilize numpy's randint which allows to create 2D samples (for x and y coordinates in this case)
data = np.random.randint(100, size=(10,2)) # size=(m, n) => m samples of n size are pulled

# data = pd.read_csv('clustering.csv')
# data.head() # Gets first 5 rows?

# data = data.loc[:, ['ApplicantIncome', 'LoanAmount']]
# data.head(2)

# print(data.sample(n=2).values)

# X = data.values
# data = X


# Starting centers of clusters (hardwritten)
centers = [[1, 100], [-1, -100], [1, -1]]

# Can utilize making "blobs" of datapoints with a Gaussian distribution (and being used at the moment)
# data, _ = make_blobs(n_samples=100, centers=centers, cluster_std=100)

# Plot the data
# plt.scatter(data[:,0],data[:,1]) # ",0" means slicing the data by datapoint and grabbing the 0th column (or x).
# plt.xlabel('x'),plt.ylabel('y')
# plt.show()

''' K-Means Operation (Visualized for each iteration) '''
''' Create an empty list for each cluster, k is the cluster number '''

k = 2  # Number of clusters we want to find in data

# Initialize all clusters to be [0,0] at start
clusters = [[[0 for _ in range(2)] for _ in range(1)] for _ in range(k)]

for i in range(k):
    clusters[i].pop() # Remove the [0,0] from its home, leaving empty 2D arrays [[], []]
    # From what I can tell, this allows the KMeans algorithm to put it its own.

''' Original Code '''
kmeans = KMeans(n_clusters=k, random_state = 0).fit(data)

for index,data_point in enumerate(data):
    clusters[kmeans.labels_[index]].append(list(data_point))
    print(clusters)
    
for i in range(k):
    clusters[i] = np.array(clusters[i])
    plt.scatter(clusters[i][:,0],clusters[i][:,1])
    clusters[i] = clusters[i].tolist() 
plt.show()

# """ Image Segmentation """

# img = cv2.imread("youtube_thumbnail.jpeg", cv2.IMREAD_UNCHANGED)
# vectorized = img.reshape((-1,3))
# kmeans = KMeans(n_clusters=100, random_state = 0, n_init=5).fit(vectorized)
# centers = np.uint8(kmeans.cluster_centers_)
# segmented_data = centers[kmeans.labels_.flatten()]
 
# segmented_image = segmented_data.reshape((img.shape))
# plt.imshow(segmented_image)
# plt.pause(100)
