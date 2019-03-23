from flask import jsonify, current_app as app
from mongoDBManager import MongoDBManager
from pymongo import GEOSPHERE, ASCENDING
from pymongo.errors import (PyMongoError, BulkWriteError, InvalidOperation, OperationFailure)
import json, logging, sys
from featureExtractionCloud import FeatureExtractor
import cv2, logging, os
import numpy as np


try:
	manager = MongoDBManager(app)
	mongo = manager.getConnection()
	featurecol = manager.getCollection(mongo, 'dataFeature')
	modelCol = manager.getCollection(mongo, 'model')
	dirname = os.path.dirname(__file__)
	referenceDataSet = os.path.join(dirname, "ReferenceDataset")
	#kpList = np.
	desList = np.empty((0,32), int)
	imgNameList  = np.empty((0,1), int)
	i = 0 

	for img in os.listdir(referenceDataSet):
		imgNameList = np.append(imgNameList, np.array([[int(img.split(".")[0])]]), axis=0)
		picture = cv2.imread(os.path.join(referenceDataSet, str(img)),0)

		orb = cv2.ORB_create()

		kp, des = orb.detectAndCompute(picture, None)

		desList = np.append(desList, des, axis=0)


	knn = cv2.ml.KNearest_create()
	knn.train(desList, cv2.ml.ROW_SAMPLE, imgNameList)
	#knn.train(desList, imgNameList)
	#ret, result, neighbours, dist = knn.find_nearest(self.testData, nrK)



except Exception as e:
	logging.error(e, exc_info=True)
	raise e

finally:
	manager.closeConnection(mongo)
