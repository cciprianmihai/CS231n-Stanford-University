from cs231n.data_utils import load_CIFAR10
from NearestNeighbor import *

# Load CIFAR10 dataset
Xtr, Ytr, Xte, Yte = load_CIFAR10('cs231n/data/cifar10')
Xtr = Xtr[0:100]
Ytr = Ytr[0:100]
# Flatten out all images to be one-dimensional array of 32 * 32 * 3 = 3072
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072

nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print ('accuracy: %f' % ( np.mean(Yte_predict == Yte) ))
