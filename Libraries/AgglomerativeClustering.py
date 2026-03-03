# This is unspuervised learning algorithm which means we don't have labels 
# Agglomerative clustering is bottom up view means we work from single points up to all data sets
# We don't have classifcation or regression 
import numpy as np
# We need to draw some plots
import matplotlib.pyplot as plt
# We need to draw dendrogram and we can't make it from scratch as it is tideous and have dozend of matplotlib lines
from scipy.cluster.hierarchy import dendrogram

# For warnings
import warnings
warnings.filterwarnings('ignore')

class AgglomerativeClustering():

    # Intialization
    def __init__(self, n_clusters=2, linkage='single'):
        # Here other paramerts default as API
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.X = None
        # Distance (simialarity) (proximity) matrix
        # We call it linkage because calculating distance depeneds on linkage methods
        self.linkage_matrix = None
        self._labels = None

    # Helper functions
    # Find the best merge and it is the most important function
    def _find_best_merge(self, dist_matrix, current_ids, clusters):
        # My target is to find the global min distance in all matrix
        # Intialize return value which are best distance and the best pairs
        best_dist = np.inf
        best_pair = (None, None)

        # We will loop over all current ids
        # The one current id is the iterator it self
        for i in range(len(current_ids)):
            # Each id will loop over others
            for j in range(i+1, len(current_ids)):
                # We work with two clusters and get all related distances
                c1_indices, c2_indices = clusters[current_ids[i]], clusters[current_ids[j]]
                # Collect all possible distances in subdistance matrix
                sub_dist = dist_matrix[np.ix_(c1_indices, c2_indices)]
                
                # Based on the linkage method the final distance will be one of these
                if self.linkage == 'single': 
                    d = np.min(sub_dist)
                elif self.linkage == 'complete':
                    d = np.max(sub_dist)
                elif self.linkage == 'average': 
                    # For average method
                    d = np.mean(sub_dist) 
                
                if d < best_dist:
                    # The least distance with its pairs should return
                    best_dist = d
                    # The best pair will be current ids for i and j
                    best_pair = (current_ids[i], current_ids[j])

        return best_pair[0], best_pair[1], best_dist
    
    # Silhouette function
    def _calculate_silhouette(self, X):
    # Get samples number
        n_samples = X.shape[0]

        # Get all scores cause later we get the mean
        silhouette_vals = []
        
        # loop over all points and inside each iteration we have another loop for the closest cluster
        for i in range(n_samples):
            # Get a(i) so we work in the same cluster
            # Get the cluster of point first
            inside_cluster = X[self.labels_ == self.labels_[i]]
            if len(inside_cluster) > 1:
                # Make sure it is has more than two points
                # Inside cluster is the current cluster points
                a = np.mean(np.linalg.norm(inside_cluster - X[i], axis=1))
            else:
                # No points so no averga
                a = 0

            # intialize b to get min b at the end 
            b = np.inf
            # Loop over other cluster
            for label in range(self.n_clusters):
                # Get b(i) so we work with the closets cluster ignoring our current one
                if label == self.labels_[i]:
                    continue
                
                # Each time we calcualte b till we get the min one
                # Outside cluster is the outer current cluster points
                outside_cluster = X[self.labels_ == label]
                
                if len(outside_cluster)>0:
                    # Make sure it has point
                    current_b = np.mean(np.linalg.norm(outside_cluster - X[i], axis=1))
                    b = min(b, current_b)

            # Calcualte silhouette for current point and to be added later to the silhouette_vals
            # Add stability term to avoid zero devsion
            s = (b - a)  / (max(a, b) + 1e-9)       
            silhouette_vals.append(s)

        # Get the average of all score
        return np.mean(silhouette_vals)    
    
    # Fit
    def fit_predict(self, X):
        # Convert x to array
        X = np.array(X) 
        self.X = X
        n_samples = X.shape[0]

        # i = 0
        # Step 0 --> Consider every point as sperated cluster at i = 0 
        # Here the arch is cluster : its points
        # i are the cluster and the point at the same time so cluster as key : point as value
        clusters = {i : [i] for i in range(n_samples)}

        # Step 1 --> calculate distances between all points (clusters)
        # First time it gonna be euclidean distanceso it is only distance matrix 
        # Next will be simialarity (proximity) (linkage)
        # we use np.newaxis as X is (m X n) and X (m X n) so we can't devide all i need to devide each sample point from all others at once to avoid getting zero
        # So convert from (m x n) to (m x 1 x n) 3D matrix (samples, new axis to devide and n) to get (m, m, n_differ) so it is 3D
        # The new matrix is m x m
        # Axis = 2 refers to old matrix and stored in new
        dist_matrix = np.sqrt(np.sum((self.X[:, np.newaxis] - self.X)**2, axis=2))
        # Fill diagnol with infinity as the distance between point and itself is zero but we make as inf for safety
        # Zero could be least so we will merge point with itself and that is a waste of time
        np.fill_diagonal(dist_matrix, np.inf)

        # Record all linkage distcance to view in dendrogram
        self.linkage_matrix = []

        # Detect current clusters and next cluster as we get one per iteration
        current_clusters_ids = list(clusters.keys())
        next_cluster_ids= n_samples

        while len(current_clusters_ids) > 1 :
            # I should have more than one cluster
            # The iteration is not specified with range it is specified till we got one cluster so stop case is one cluster and another rules
            # Step 2 --> find the best merge
            # I got the best merge (global min distance) through passing distance matrix, current clusters and the clusters
            # We will use linkage method
            c1, c2, min_dist = self._find_best_merge(dist_matrix, current_clusters_ids, clusters)

            # Record distnace to view it in dendrogram
            new_size = len(clusters[c1] + clusters[c2])
            # Record all things for every iteration in linkage matrix
            self.linkage_matrix.append([float(c1), float(c2), float(min_dist), new_size])

            # Merge and get one new cluster
            new_clusters = clusters[c1] + clusters[c2]
            clusters[next_cluster_ids] = new_clusters

            # Remove first the previous two clusters as they are one new cluster now
            current_clusters_ids.remove(c1)
            current_clusters_ids.remove(c2)
            current_clusters_ids.append(next_cluster_ids)

            # Check if we reach the desired n_clusters
            if len(current_clusters_ids) == self.n_clusters:
                # Intialize labels
                self.labels_ = np.zeros(n_samples)
                for cluster_label, c_id in enumerate(current_clusters_ids):
                    # Assign labels to each cluster (0, 1, 2 etc)
                    # We sarch inside the current clsuters we have
                    self.labels_[clusters[c_id]] = cluster_label
                    
            # Go to next level
            next_cluster_ids += 1

        # Convert the final linkage_matrix to numpy array
        self.linkage_matrix = np.array(self.linkage_matrix)
        
        # Silhouette case always
        self.silhouette_ = self._calculate_silhouette(X)
        return self

    # Predict
    def predict(self, X_test):
        # Here it is just geaometric prediction
        # Convert X to array
        X_test = np.array(X_test)
        # Returb predicstions
        preds = []
        for x in X_test:
            # Here we assign as out of sample assigment to the nearest cluster based on min linkage value
            dists = np.linalg.norm(self.X - x, axis=1)
            preds.append(self.labels_[np.argmin(dists)])
        return np.array(preds)
    
    # Dendrogram draw
    def plot_dendrogram(self):
        # Plotly size
        plt.figure(figsize=(10, 5))
        # Pass linkage matrix to dendrogram like sch.linkage
        dendrogram(self.linkage_matrix)
        plt.title(f"Agglomerative Dendrogram ({self.linkage})")
        plt.show()