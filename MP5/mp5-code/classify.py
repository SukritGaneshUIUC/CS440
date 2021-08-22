# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np
import numpy.linalg as la
import queue

def euclideanDistance(arr1, arr2):
    s = 0
    for i in range(len(arr1)):
        s += (arr2[i] - arr1[i]) ** 2
    return s

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))

def maxIndex(arr):
    maxVal = max(arr)
    for i in range(len(arr)):
        if (arr[i] == maxVal):
            return i

def mode(arr):
    trueCount = 0
    falseCount = 0
    for elem in arr:
        if (elem == True):
            trueCount += 1
        else:
            falseCount += 1

    if (falseCount >= trueCount):
        return False
    return True

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters

    W = np.zeros(train_set.shape[1] + 1)      # includes the weight for the bias

    for i in range(max_iter):
        for j in range(len(train_set)):
            currImg = np.insert(train_set[j], 0, 1)         # must insert 1 at beginning for bias
            currLabel = train_labels[j]
            otp = np.dot(currImg, W)

            # Update Case 1: currLabel = FALSE and otp > 0 (must subtract from weights)
            if (currLabel == False and otp > 0):
                W -= currImg * learning_rate
            # Update Case 0: currLabel = TRUE and otp <= 0 (must add to weights)
            elif (currLabel == True and otp <= 0):
                W += currImg * learning_rate

    return W[1:], W[0]      # first term is bias (weight0), remaining terms are weights

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set

    W, b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    W = np.insert(W, 0, b)  # add bias to beginning of weights

    labels = []

    for i in range(len(dev_set)):
        currImg = np.insert(dev_set[i], 0, 1)       # insert 1 at beginning for bias
        currOutput = np.dot(W, currImg)
        if (currOutput > 0):
            labels.append(1)
        else:
            labels.append(0)

    return labels

def classifyKNN(train_set, train_labels, dev_set, k):
    # TODO: Write your code here

    print('Beginning KNN classification')

    predicted = []
    print('Dev images:', len(dev_set))
    print('Train images:', len(train_set))

    for i in range(dev_set.shape[0]):
        print('Image:', i)
        currImg = dev_set[i]

        nearestImages = []
        nearestImageDistances = []
        nearestImageLabels = []

        for j in range(train_set.shape[0]):
            currTrainImg = train_set[j]
            currDist = la.norm(currTrainImg - currImg)
            currLabel = train_labels[i]

            if (len(nearestImages) < k):
                nearestImages.append(currTrainImg)
                nearestImageDistances.append(currDist)
                nearestImageLabels.append(train_labels[j])
            elif (currDist < max(nearestImageDistances)):
                # find the index of the image with the max distance, and replace it
                maxIdx = maxIndex(nearestImageDistances)
                nearestImages[maxIdx] = currTrainImg
                nearestImageDistances[maxIdx] = currDist
                nearestImageLabels[maxIdx] = train_labels[j]

        # find the most common label and append it to predicted

        currLabel = mode(nearestImageLabels)
        print('Prediction:', currLabel)
        print()
        predicted.append(currLabel)

    return predicted
