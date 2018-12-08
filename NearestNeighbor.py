import numpy as np

class NearestNeighbor(object):
    def __init__(self):
        pass
    def train(self, X, y):
        # X is N x D where each row is an example. Y is 1-dimension of size N
        # The Nearest Neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y
    def getManhattanDistance(self, point):
        distances = np.sum(np.abs(self.Xtr - point), axis=1)
        return distances
    def getEuclideanDistance(self, point):
        distances = np.sqrt(np.sum(np.square(self.Xtr - point), axis=1))
        return distances
    def predict(self, X, distance = 1):
        global distances
        # X is N x D where each row is an example we wish to predict label for
        num_test = X.shape[0]
        # Lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        # Loop over all test rows
        for i in range(num_test):
            if(distance == 1):
                # Find the nearest training image to the i'th test image
                # using the L1 distance (sum of absolute value differences)
                distances = self.getManhattanDistance(X[i,:])
            if(distance == 2):
                # Find the nearest training image to the i'th test image
                # using the L2 distance (square root of sum of square differences)
                distances = self.getEuclideanDistance(X[i,:])
            min_index = np.argmin(distances) # get the index with smallest distance
            Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
        return Ypred