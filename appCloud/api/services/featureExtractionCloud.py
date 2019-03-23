
import numpy as np
import cv2
from matplotlib import pyplot as plt

class FeatureExtractor(object):

	def __init__(self, img):
		self.img = img

	def ORB (self):
		# Initiate STAR detector
		orb = cv2.ORB()

		# Find the keypoints with ORB
		kp = orb.detect(self.img, None)

		#compute the descriptors with ORB
		kp, des = orb.compute(self.img, kp)

		return kp, des
	
