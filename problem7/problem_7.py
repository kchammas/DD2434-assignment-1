import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import manifold

import argparse

# Function that loads the dataset and returns dataset, annotations for the points as well as color indexes for the points
#
# Input:    
#           path-string to the dataset file
#
# Output:
#           dataset - nd.array of shape (d,n) where each column is a datapoint
#           annotations - nd.array of shape (1,n) - each entry is the name of the animal of the corresponding datapoint in dataset
#           colors - nd.array of shape (1,n) - each entry is the color index of the animal of the corresponding datapoint in dataset
def load_dataset(path):
    dataset = []
    annotations = []
    colors = []
    with open(path, "r") as file:
        for line in file:
            attributes = line.rstrip("\n").split(",")
            annotations.append(attributes[0])
            colors.append(float(attributes[-1]))
            dataset.append([float(attributes[i]) for i in range(1, len(attributes)-1)])
    return np.array(dataset).T, np.array(annotations).reshape(1, -1), np.array(colors).reshape(1, -1)

# Function that normalizes the leg attribute in dataset. This is done by dividing the attribute by the maximum number 8, 
# so that we get values in the range 0 to 1 (actually normalizes all attributes, but since the other ones are binary nothing will happen)
#
# Input: 
#           dataset - nd.array of shape (d,n) where each column is a datapoint
#
# Output: 
#           dataset - nd.aray of shape (d,n) where each column is a datapoint with normalized "leg" attribute
def normalize_leg_attribute(dataset):
    return dataset / np.max(dataset, axis=1).reshape(-1, 1)

# Function that computes the SVD of a matrix with dimension m x n
#
# The dimensions of the left singular matrix is m x m
# The dimensions of the right singular matrix is n x n
# The dimensions of the matrix with singular vector is m x n, where the main diagonal has the singular values
#
# Input:   
#            A - matrix with arbitrary dimension m x n
# Output:
#           U - matrix containing the left singular vectors - dimension m x m
#           sigma - matrix with singular values - dimension m x n
#           V - matrix containing the right singular vectors - dimension n x n
def svd(A):

    A = np.copy(A)

    d, n = A.shape
    U, sigma, VT = np.linalg.svd(A)

    if d == n:
        sigma = np.diag(sigma)
    elif d > n:
        sigma = np.pad(np.diag(sigma), ((0, d-n), (0, 0)))
    else:
        sigma = np.pad(np.diag(sigma), ((0, 0), (0, n-d)))

    return U, sigma, VT.T

# Function that performs PCA on a matrix, with a given target dimension
# First, the data is centered
# Second, PCA is performed with the sklearn library (note that the library wants datapoints as rows)
#
# Input: 
#           A - matrix to perform pca on - dimension d x n (datapoints are columns)
#           n_components - number of dimension to reduce to
# Output:
#           W.T - linear mapping that maps points from the latent space back to original space - dimension ()
#           pca_obj.explained_variance_ratio_ - dimension d x k
#           A - the centered dataset - dimension d x n
def pca(A, n_components=2):

    A = np.copy(A)
    d, n = A.shape

    # data centering
    A = A - (1/n) * A @ np.ones((n, 1)) @ np.ones((n, 1)).T

    # PCA
    pca_obj = decomposition.PCA(n_components=n_components)
    pca_obj.fit(A.T)
    W = pca_obj.components_

    return W.T, pca_obj.explained_variance_ratio_, A

# Function that performs classic MDS on data matrix
#
# Input:
#           A - data matrix - dimension d x n (or a similarity matrix with dimension n x n)
#           n_components - dimension of resulting manifold
#           is_gram_matrix - True if A is already a similarity matrix, otherwise it is a data matrix
#           attribute_importance - Whether to use importance variance factors for each attribute when computing similarity matrix from data matrix
def mds(A, n_components=2, is_gram_matrix=False, attribute_importance=False):
    
    A = np.copy(A)
    d, n = A.shape

    if is_gram_matrix == False:
        
        distances = np.zeros((n, n))

        if attribute_importance == True:
            variances = np.std(A, axis=1).reshape(-1,1)**2
            variances = variances / np.max(variances)
            for col1 in range(n):
                for col2 in range(n):
                    dist = np.linalg.norm((A[:, col1] - A[:, col2])*variances.reshape(-1))
                    distances[col1, col2] = dist
        else:
            for col1 in range(n):
                for col2 in range(n):
                    dist = np.linalg.norm(A[:, col1] - A[:, col2])
                    distances[col1, col2] = dist


        S = double_centering_trick(distances)
        
    else:
        S = A

    # eigen decomposition
    eig_val, eig_vec = np.linalg.eig(S)
    eig_val = np.real(eig_val)
    eig_vec = np.real(eig_vec)

    # non-decreasing eigen value order
    idx = np.argsort(eig_val)[::-1]
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]

    # remove small number that are essentially 0 before sqrt, otherwise they cause nan
    eig_val[eig_val < 1e-13] = 0

    # calculate the final embedding of the datapoints
    X = np.eye(n_components, n) @ np.diag(np.sqrt(eig_val)) @ eig_vec.T

    return X

# Function that computed the geodesic distances, to later be used in MDS
#
# This is done by first computing all the euclidian distances
# Then a graph is created based on the shortest euclidian distances and how many neighbors each node should get
# Then we check if the graph is connected, and then we compute the geodesic distances
#
# Input:
#           A - data matrix - dimension d x n
#           n_neighbors - how many nearest neighbor each node should get
#
# Output:
#           geodesic_distances - the geodesic distances for each pair of datapoint indexes - dimension n x n
def compute_isomap_distances(A, n_neighbors=2):

    A = np.copy(A)
    d, n = A.shape

    # initialize nodes, neighbors and euclidian_distances nd.arrays
    nodes = np.array([i for i in range(0, n)])
    neighbor_matrix = np.zeros((n, n))
    euclidian_distances = np.zeros((n, n))
    geodesic_distances = np.zeros((n, n))

    # calculate eucidian distances in original space
    for col1 in range(n):
        for col2 in range(n):
            dist = np.linalg.norm(A[:, col1] - A[:, col2])
            euclidian_distances[col1, col2] = dist

    # loop over all nodes and add k neighbors
    for node in range(n):

        # sort distances
        distances = euclidian_distances[node,:]
        idx = np.argsort(distances)

        # for all the non zero k first smallest distances, add edge between corresponding nodes (undirected graph)
        neighbor_count = 0
        for i in range(len(distances)):
            if distances[idx[i]] != 0:
                neighbor_matrix[node,idx[i]] = 1
                neighbor_matrix[idx[i],node] = 1
                neighbor_count += 1
                if neighbor_count == n_neighbors:
                    break

    # check if graph is connected
    if check_graph_connectivity(neighbor_matrix) == False:
        raise ValueError

    # calculate geodesic distances
    geodesic_distances = shortest_path(neighbor_matrix, euclidian_distances)

    return geodesic_distances

# Function that checks if a graph is connected
#
# Input:
#           neighborhood - the graph - dimension n x n
# Output:
#           True - if connected
#           False - if disconnected
def check_graph_connectivity(neighborhood):

    neighborhood = np.copy(neighborhood)
    n, m = neighborhood.shape
    visited_nodes = np.zeros((n))

    assert(n == m)
    assert(neighborhood.tolist() == neighborhood.T.tolist())

    nodes_to_visit = [0]
    while nodes_to_visit != []:
        node = nodes_to_visit.pop()
        visited_nodes[node] = 1
        neighbors = neighborhood[node, :]
        for i in range(len(neighbors)):
            if neighbors[i] == 1 and visited_nodes[i] == 0:
                nodes_to_visit.append(i)

    for visit in visited_nodes:
        if visit == 0:
            return False
    return True

# Function that calculated the shortest path distances for each pair of nodes
# It is basically an implementation of the FW algorithm.
#
# Input:
#           neighbor_matrix - the input graph - dimension n x x
#           the distances for the graph - dimension n x n
#
# Output:
#           dist - the shortest path distances - dimension n x n
def shortest_path(neighbor_matrix, distances):

    assert(neighbor_matrix.shape[0] == neighbor_matrix.shape[1])
    assert(distances.shape[0] == distances.shape[1])
    assert(neighbor_matrix.shape == distances.shape)
    assert(neighbor_matrix.tolist() == neighbor_matrix.T.tolist())
    assert(distances.tolist() == distances.T.tolist())

    v = neighbor_matrix.shape[0]
    dist = np.ones((v,v))*np.inf

    for i in range(v):
        dist[i, i] = 0

    for i in range(v):
        for j in range(v):
            if neighbor_matrix[i, j] == 1:
                dist[i, j] = distances[i, j]

    for k in range(v):
        for i in range(v):
            for j in range(v):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    return dist

# Function that performs the double cenering trick
# A trick to convert distance matrix to similarity/gram matrix
#
# Input:
#           distances - matrix of distances between all pair of nodes - dimension n x n
# Output:
#           distances - matrix of similarities between all pair of nodes - dimension n x n
def double_centering_trick(distances):

    assert(distances.shape[0] == distances.shape[1])
    assert(distances.tolist() == distances.T.tolist())

    n, m = distances.shape

    distances = np.copy(distances)
    distances = -0.5*(distances - (1/n)*distances@np.ones((n,1))@np.ones((n,1)).T - (1/n)*np.ones((n,1))@np.ones((n,1)).T@distances + (1/(n**2))*np.ones((n,1))@np.ones((n,1)).T@distances@np.ones((n,1))@np.ones((n,1)).T)

    return distances

# Function that plots a set of points, with the given colors and annotations
# It has several modes
#
# Input:
#           points - points to be plotted - dimension 2 x n
#           colors - corresponding color to the points - dimension n (array)
#           annotations - if provided, also annotates all the points (might cause name overlaps because of identical points) - dimension n (array)
#           mode - whether to plot PCA, MDS or isomap information
#           pca_explained_variance - if mode is PCA, then the explained variance array should be stated - dimension 2 (array of 2 numbers)
#           attribute_importance - if mode is MDS and attribute_importance is true, the title in the plot should reflect that
#           n_neigbors - if mode is isomap, then the number of neighbors should be stated
# Output:
#           none, just plots a graph
def plot_points(points, colors, annotations=None, mode=None, pca_explained_variance=None, mds_attribute_importance=False, n_neighbors=None):
    plt.clf()
    color_mapping = [None, "blue", "red", "yellow", "green", "black", "purple", "orange"]

    for col in range(points.shape[1]):
        plt.plot(points[0, col], points[1, col], linestyle="None", marker="o", color=color_mapping[int(colors[0, col])])
        if annotations is not None:
            plt.annotate(annotations[0, col], (points[0, col], points[1, col]))

    if mode == "PCA":
        plt.title("Dimensionality Reduction with PCA")
        plt.xlabel("Principal Component 1 - " + str(np.round(pca_explained_variance[0], 3)) + "% explained variance")
        plt.ylabel("Principal Component 2 - " + str(np.round(pca_explained_variance[1], 3)) + "% explained variance")
    elif mode == "MDS":
        if mds_attribute_importance == True:
            plt.title("Dimensionalty Reduction with MDS - with attribute importance")
        else:
            plt.title("Dimensionalty Reduction with MDS - without attribute importance")
        plt.xlabel("x1")
        plt.ylabel("x2")
    elif mode == "isomap":
        plt.title("Dimensionality Reduction with isomap - Number of Neighbors: " + str(n_neighbors))
        plt.xlabel("x1")
        plt.ylabel("x2")

    plt.show()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--annotations", help="Specify this flag if annotations should be included in plots", action="store_true")
    parser.add_argument("-p", "--PCA", help="Specify this flag if dimensionality reduction method is PCA", action="store_true")
    parser.add_argument("-m", "--MDS", help="Specify this flag if dimensionality reduction method is MDS", action="store_true")
    parser.add_argument("-i", "--importance", help="Specify this flag, along with MDS flag, if feature importance should be taken into account for", action="store_true")
    parser.add_argument("-iso", "--isomap", help="Specify this flag if dimensionality reduction method is isomap", action="store_true")

    args = parser.parse_args()

    # load the dataset
    dataset, annotations, colors = load_dataset("dataset/zoo.data")

    if args.annotations == False:
        annotations = None

    # normalize the leg attribute so that the distribution is in the range [0,1]
    dataset = normalize_leg_attribute(dataset)

    # Plot for PCA
    if args.PCA == True:
        W, W_explained_variance_ratio, centered_dataset = pca(dataset, n_components=2)
        projected_dataset = W.T @ centered_dataset
        plot_points(projected_dataset, colors, annotations=annotations, mode="PCA", pca_explained_variance=W_explained_variance_ratio)

    # Plot for MDS with attribute importance
    elif args.MDS == True and args.importance == True:
        projected_dataset = mds(dataset, n_components=2, is_gram_matrix=False, attribute_importance=True)
        plot_points(projected_dataset, colors, annotations=annotations, mode="MDS", mds_attribute_importance=True)

    # plot for MDS without attribute importance
    elif args.MDS == True:
        projected_dataset = mds(dataset, n_components=2, is_gram_matrix=False, attribute_importance=False)
        plot_points(projected_dataset, colors, annotations=annotations, mode="MDS", mds_attribute_importance=False)

    # plots for isomap for different numbers of neighborhoods
    elif args.isomap == True:
        tot_neighbors = dataset.shape[1]
        for i in range(1, tot_neighbors):
            try:
                distances = compute_isomap_distances(dataset, n_neighbors=i)
            except ValueError:
                print("graph with " + str(i) + " neighbors is disconneected")
                continue
            S = double_centering_trick(distances)
            projected_dataset = mds(S, n_components=2, is_gram_matrix=True)
            plot_points(projected_dataset, colors, annotations=annotations, mode="isomap", n_neighbors=i)

if __name__ == "__main__":
    main()