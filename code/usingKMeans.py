'''
    KMeans Code
    Originated from Aktas's Tutorial: https://towardsdatascience.com/image-segmentation-with-clustering-b4bbc98f2ee6
'''
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, make_blobs, load_digits
import numpy as np
from matplotlib import pyplot as plt
import cv2
import pandas as pd
import os

''' Understanding make_blobs, and how data is formatted at the moment '''

# Can utilize handcoded data
data = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])
data = np.array([[1, 5], [3, 1], [10, 3], [10, 2], [10, 1], [1, 0], [2, 15], [0.5, 4.9], [5, 3], [7, 13], [18, 18], [1.9, 0.5]])

# Or, for simplicity utilize numpy's randint which allows to create 2D samples (for x and y coordinates in this case)
data = np.random.randint(100, size=(10,2)) # size=(m, n) => m samples of n size are pulled
data = np.array([
        [1,1,2,1], 
        [2,1,2,3], 
        [2,2,4,5], 
        [50,42,2,83],
        [51,43,1,82],
        [51,44,3,89],
        [53,40,8,80]])

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

k = 5  # Number of clusters we want to find in data

# Initialize all clusters to be [0,0] at start
clusters = [[[0 for _ in range(2)] for _ in range(1)] for _ in range(k)]

for i in range(k):
    clusters[i].pop() # Remove the [0,0] from its home, leaving empty 2D arrays [[], []]
    # From what I can tell, this allows the KMeans algorithm to put it its own.

''' Original Code '''
# kmeans = KMeans(n_clusters=k, random_state = 0).fit(data)

# for index,data_point in enumerate(data):
#     clusters[kmeans.labels_[index]].append(list(data_point))
#     print(clusters)
    
# for i in range(k):
#     clusters[i] = np.array(clusters[i])
#     plt.scatter(clusters[i][:,0],clusters[i][:,1])
#     clusters[i] = clusters[i].tolist() 
# plt.show()

# """ Image Segmentation """

img = cv2.imread("corgi-white.jpeg", cv2.IMREAD_COLOR)
print(img.shape)
vectorized = img.reshape((-1,3))
print(vectorized)
kmeans = KMeans(n_clusters=10, random_state = 0, n_init=5).fit(vectorized)
centers = np.uint8(kmeans.cluster_centers_)
segmented_data = centers[kmeans.labels_.flatten()]
 
segmented_image = segmented_data.reshape((img.shape))

plt.imshow(segmented_image)
plt.pause(100)

def elbowMethod(data, max_iterations=10):
    wcss = []
    for i in range(1, max_iterations+1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

# print(vectorized.shape)

plt.plot(range(1, 11), elbowMethod(vectorized))
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()


def IoU():
    pass

def openImage(file_path, is_training=False):
    # file_path = f"{os.getcwd()}/BSDS300/images/{sub_dir}/{image_num}.jpg"
    # file_path = f"{os.getcwd()}/BSDS300/images/human/{image_num}.jpg"
    print(file_path)
    return cv2.imread(file_path, cv2.IMREAD_COLOR)


def determineFile(image_num, is_training=False, is_human=False):
    if is_training and is_human:
        return ValueError("No human segmented image is a training image.")
    elif is_human:
        file_path = f"{os.getcwd()}/BSDS300/images/human/color/1102/{image_num}.seg"
        return file_path
    else:
        sub_dir = "train" if is_training else "test"
        return f"{os.getcwd()}/BSDS300/images/{sub_dir}/{image_num}.jpg"       


def testImage(image_num, k=10):
    test_file = determineFile(image_num, False, False)
    human_file = determineFile(image_num, False, True)
    cv_image = openImage(test_file)
    plt.imshow(cv_image)
    plt.pause(100)
    segementImage(cv_image, k)


def segementImage(cv_file, k):
    clusters = [[[0 for _ in range(2)] for _ in range(1)] for _ in range(k)]
    for i in range(k):
        clusters[i].pop()
    print(cv_file.shape)
    vectorized = cv_file.reshape((-1,3))
    kmeans = KMeans(n_clusters=k, random_state = 0, n_init=5).fit(vectorized)
    centers = np.uint8(kmeans.cluster_centers_)
    print(centers)
    segmented_data = centers[kmeans.labels_.flatten()]
    print(kmeans.labels_.flatten())
    print(segmented_data)
    segmented_image = segmented_data.reshape((cv_file.shape))
    print(segmented_image)
    plt.imshow(segmented_image)
    plt.pause(100)

# segementImage(cv2.imread("corgi-white.jpeg"), 3)
# elbowMethod(cv2.imread("corgi-white.jpeg"))

# testImage(14037, 11)

# digits = load_iris()
# data = digits.data 
# data = 255-data
# print(len(data))
# print(data.shape)
# print(data)

# n=20
# kmeans = KMeans(n_clusters=n, init="random")
# kmeans.fit(data)
# Z = kmeans.predict(data)
# print(Z)


# plt.plot(range(1, 11), elbowMethod(data, 10))
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS') 
# plt.show()

# for i in range(0,n):
#     row = np.where(Z==i)[0]  # row in Z for elements of cluster i
#     num = row.shape[0]       #  number of elements for each cluster
#     r = np.floor(num/10.)    # number of rows in the figure of the cluster 

#     print("cluster "+str(i))
#     print(str(num)+" elements")

#     plt.figure(figsize=(10,10))
#     for k in range(0, num):
#         # print(int(r+1, k+1)
#         plt.subplot(int(r+1), 10, k+1)
#         image = data[row[k], ]
#         image = image.reshape(2, 2)
#         plt.imshow(image, cmap='rainbow')
#         plt.axis('off')
#     plt.show()


# n=10
# kmeans=KMeans(n_clusters=n, init="random")
# kmeans.fit(data)
# Z = kmeans.predict(data)

# for i in range(0,n):
#     row = np.where(Z==i)[0]  # row in Z for elements of cluster i
#     num = row.shape[0]       #  number of elements for each cluster
#     r = np.floor(num/10.)    # number of rows in the figure of the cluster 

#     print("cluster "+str(i))
#     print(str(num)+" elements")

#     plt.figure(figsize=(10,10))
#     for k in range(0, num):
#         # print(int(r+1, k+1)
#         plt.subplot(int(r+1), 10, k+1)
#         image = data[row[k], ]
#         image = image.reshape(8, 8)
#         plt.imshow(image, cmap='gray')
#         plt.axis('off')
#     plt.show()

