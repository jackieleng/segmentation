import numpy as np


def calc_sum_squares(means, centroid):
    diff = means - centroid
    squares = diff**2
    return squares.sum(axis=1)


def calculate_centroid(means):
    num_means = means.shape[0]
    return np.true_divide(means.sum(axis=0), float(num_means))


def assign_to_clusters(X, centroids):
    """Create a new 1D array with length equal to X.shape[0] that classifies
    each sample of X to a centroid cluster.

    The assigned value is the index of the centroid in ``centroids``.
    """
    num_samples = X.shape[0]
    num_centroids = centroids.shape[0]
    sum_squares = np.zeros((num_samples, k))
    for k in range(num_centroids):
        centroid = centroids[k, :]
        sum_squares[:, k] = calc_sum_squares(X, centroid)
    return sum_squares.argmin(axis=1)


class KMeans(object):
    def __init__(self, k=3):
        """
        Args:
            k: number of clusters to classify into
        """
        self.k = k

    def fit(self, X, iterations=100):
        """
        Fit and train the model.

        Args:
            X: ndarray of shape (num_samples, num_features)
        """
        self.X = X.astype(float)
        self.num_samples, self.num_features = X.shape

        # generate initial centroids
        centroids_indices = np.random.choice(
            self.num_samples, self.k, replace=False)
        centroids = self.X[centroids_indices, :]

        clusters = None
        converged = False

        for i in range(iterations):
            print "iteration %s" % i

            # step 1: assign samples to centroids (clusters)
            new_clusters = assign_to_clusters(self.X, centroids)
            if clusters is not None and np.all(clusters == new_clusters):
                converged = True
                break
            clusters = new_clusters

            # step 2: calculate new centroids
            for k in range(self.k):
                cluster_means = self.X[clusters == k]
                centroids[k, :] = calculate_centroid(cluster_means)
        self.centroids = centroids

        if converged:
            print "Converged"
        else:
            print "Not converged"

        print centroids
        print clusters

    def predict(self, X):
        """
        Predict labels for samples.

        Args:
            X: ndarray of shape (num_samples, num_features)

        Returns:
            array of labels for each sample (size of num_samples)
        """
        return assign_to_clusters(X.astype(float), self.centroids)


class Segmentation(object):
    pass
