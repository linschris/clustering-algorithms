from sklearn.cluster import KMeans
import numpy as np

data = np.random.randint(100, size=(100,2)) # size=(m, n) => m samples of n size are pulled
k = 2
centroids, clusters = KMeans(n_clusters=k, random_state=0).fit(data)

for index,data_point in enumerate(data):
    clusters[kmeans.labels_[index]].append(list(data_point))
    print(clusters)
    
for i in range(k):
    clusters[i] = np.array(clusters[i])
    plt.scatter(clusters[i][:,0],clusters[i][:,1])
    clusters[i] = clusters[i].tolist() 
plt.show()