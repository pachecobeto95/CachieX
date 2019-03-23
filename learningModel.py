import numpy as np
from matplotlib import pyplot as plt
import cv2


class SupervisedLearning(object):

	def __init__(self, trainData, labels, testData):
		self.trainData = trainData
		self.labels = labels

	def KNN(self, nrK):
		knn = cv2.KNearest()
		knn.train(self.trainData, self.labels)
		ret, result, neighbours, dist = knn.find_nearest(self.testData, nrK)
		return (ret, result, neighbours, dist)


