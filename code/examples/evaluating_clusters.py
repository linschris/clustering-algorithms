from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, davies_bouldin_score
from clustering_images import cluster_digits, cluster_iris
from clustering_image_pixels import clusterImagePixels
import matplotlib.pyplot as plt
import numpy as np


def main():
    data, labels = cluster_digits()
    # img_data, img_labels = clusterImagePixels()
    # print(davies_bouldin_score(data, labels)) # 0 is lowest possible score
    # print(davies_bouldin_score(img_data, img_labels)) # 0 is lowest possible score
    from sklearn.datasets import load_digits
    data = load_digits()
    print(data.target)
    print(labels)
    print(list(data.target_names))
    iris_matrix = confusion_matrix(data.target, labels)
    cf_matrix = ConfusionMatrixDisplay(confusion_matrix=iris_matrix, display_labels=list(data.target_names))
    plot = cf_matrix.plot()
    plt.show()
    print(classification_report(data.target, labels))

def determineNumClusters(data, max_possible_clusters):
    wcss = elbowMethod(data, max_possible_clusters)
    plt.plot(range(1, 11), wcss)
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS') 
    plt.show()

def elbowMethod(data, max_iterations=10):
    wcss = []
    for i in range(1, max_iterations+1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss


if __name__ == '__main__':
    main()
