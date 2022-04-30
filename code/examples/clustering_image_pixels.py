from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np
import cv2

def clusterImagePixels():
    orig_img, reshaped_img = readImage("code/images/corgi-tree.jpeg")
    kmeans = KMeans(n_clusters=3, random_state = 0, n_init=5).fit(reshaped_img)
    centers = np.uint8(kmeans.cluster_centers_)
    segmented_data = centers[kmeans.labels_.flatten()]
    segmented_image = segmented_data.reshape((orig_img.shape))
    plt.imshow(segmented_image)
    plt.pause(100)
    return reshaped_img, kmeans.labels_


def readImage(img_file_name):
    '''Converts image into datapoints to be clustered'''
    img = cv2.imread(img_file_name, cv2.IMREAD_COLOR)
    reshaped_image = img.reshape((-1, 3))
    return img, reshaped_image

if __name__ == "__main__":
    clusterImagePixels()
