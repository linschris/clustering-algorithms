from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, davies_bouldin_score
from clustering_images import cluster_digits, cluster_iris
from clustering_image_pixels import clusterImagePixels
from sklearn.datasets import load_digits, load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


def main():
    ''' Cluster the data before evaluating it. '''
    digits_data, digit_labels = cluster_digits(show_clusters=False)
    iris_data, iris_labels = cluster_iris(show_clusters=False)


    ''' Evaluating based on Davies Bouldin Score which can score without the ground truth!'''
    printDBScore(digits_data, digit_labels) # Score for clustering digits without ground truth
    printDBScore(iris_data, iris_labels) # Score for clustering irises without ground truth

    ''' Get the ground truth(s) '''
    orig_digits_data = load_digits()
    orig_iris_data = load_iris()
    ground_truth = orig_iris_data.target # change to iris/digits_data
    class_names = orig_iris_data.target_names # change to iris/digits data

    # ground_truth, labels = predictions, last paramters is class names
    # Use either iris or digit_labels
    ''' May take some iterations (reruns) before class_names and iris_labels line up; this is due to clustering not identifying the classes. '''
    iris_matrix = makeConfusionMatrix(ground_truth, iris_labels, class_names)

    ''' Evaluating based on recall, precision, accuracy in a classification report '''
    print(classification_report(ground_truth, iris_labels))

    determineNumClusters(digits_data, 30)

def printDBScore(data, labels):
    '''
        DB = Davies Bouldin Score 
        0 is lowest value possible.
    '''
    print(davies_bouldin_score(data, labels))


def makeConfusionMatrix(ground_truth, predictions, class_names=None):
    ''' Create and plot a confusion matrix to visualize '''
    iris_matrix = confusion_matrix(ground_truth, predictions)
    cf_matrix = ConfusionMatrixDisplay(confusion_matrix=iris_matrix, display_labels=class_names)
    plot = cf_matrix.plot()
    plt.show()


def determineNumClusters(data, max_possible_clusters):
    ''' Computes K-Means on the data with 1 -> max_possible_clusters and shows WCSS graph (refer to README.md) '''
    ''' Look for the "elbow" or second bend to determine how many clusters to use for the data. '''
    wcss = elbowMethod(data, max_possible_clusters)
    plt.plot(range(1, max_possible_clusters+1), wcss)
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
