import numpy as np
from collections import Counter

class KNearestNeighbors(object):
    def __init__(self):
        pass
    def train(self, X, y):
        # X is N x D where each row is an example. Y is 1-dimension of size N
        # The Nearest Neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y
    def getManhattanDistance(self, point, k):
        manhattanDistance = np.sum(np.abs(self.Xtr - point), axis=1)
        return np.argsort(manhattanDistance)[0:k]
    def getEuclideanDistance(self, point, k):
        euclideanDistance = np.sqrt(np.sum(np.square(self.Xtr - point), axis=1))
        return np.argsort(euclideanDistance)[0:k]
    def predict(self, X, k, distance = 1):
        global distances
        points_labels = []
        # X is N x D where each row is an example we wish to predict label for
        num_test = X.shape[0]
        # Loop over all test rows
        for i in range(num_test):
            if(distance == 1):
                distances = self.getManhattanDistance(X[i,:], k)
            if (distance == 2):
                distances = self.getEuclideanDistance(X[i,:], k)
            results = []
            for index in distances:
                results.append(self.ytr[index])
            label = Counter(results).most_common(1)
            points_labels.append(label[0][0])
        return points_labels