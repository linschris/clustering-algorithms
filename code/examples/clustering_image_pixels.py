from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np
import cv2

def clusterImagePixels():
    '''Reads image and cluster pixels by color!'''
    orig_img, reshaped_img = readImage("code/images/corgi-tree.jpeg")
    kmeans = KMeans(n_clusters=3, random_state = 0, n_init=5).fit(reshaped_img)
    centers = np.uint8(kmeans.cluster_centers_)
    segmented_data = centers[kmeans.labels_.flatten()]
    ''' 
        As the colors are clustered, we need to reshape the data back to
        the original image dimensions of h * w * 3 (R, G, B), so we can visualize
        the pixels being clustered together.
    '''
    segmented_image = segmented_data.reshape((orig_img.shape))
    plt.imshow(segmented_image)
    plt.pause(100)
    
    return reshaped_img, kmeans.labels_


def readImage(img_file_name):
    '''
        Converts image into datapoints to be clustered.
        As an image has a height, width, and three channels (for color),
        the number of data points generated is h * w * 3. For images,
        we cluster by only the colors of the pixels; the datapoints simply
        become the [Red, Green, Blue] Values of the pixel as a 3-D point.
    '''
    img = cv2.imread(img_file_name, cv2.IMREAD_COLOR)
    reshaped_image = img.reshape((-1, 3))
    return img, reshaped_image

if __name__ == "__main__":
    clusterImagePixels()
