from cs231n.data_utils import load_CIFAR10
from KNearestNeighbors import *
# Load CIFAR10 dataset
Xtr, Ytr, Xte, Yte = load_CIFAR10('cs231n/data/cifar10')
# Take only first 10 images to train the model
Xtr = Xtr[0:10]
Ytr = Ytr[0:10]
# Flatten out all images to be one-dimensional array of 32 * 32 * 3 = 3072
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 10 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072
# Create a Nearest Neighbor classifier class
nn = KNearestNeighbors()
# Train the classifier on the training images and labels
nn.train(Xtr_rows, Ytr)
# Predict labels on the test images
Yte_predict = nn.predict(Xte_rows, 3, 2)
# Print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print ('accuracy: %f' % ( np.mean(Yte_predict == Yte) ))