from flask import jsonify, session, current_app as app
import cv2, logging, os, pickle, h5py,requests, sys, config, time
import numpy as np, json
#import lfucache.lfu_cache as lfu_cache
from sklearn import linear_model
from scipy.misc import derivative
#from expLatencySize import cache

def setCache(filePath, file):
	imgFile = cv2.imread(filePath, 0)
	orb = cv2.ORB_create()
	kpData, desData = orb.detectAndCompute(imgFile, None)
	with open(config.CACHE_FILE_CACHIER, 'rb') as f:
		cache = pickle.load(f)

	setTest = cache.set(file.filename, desData)

def saveImage(file):
	try:
		repoPath = os.path.join(config.DIR_NAME, "appEdge", "api", "services", file.filename)

		if (file):
			file.save(repoPath)
			setCache(repoPath, file)
	except Exception as e:
		return {'status':'error','msg': e.args}

	else:
		return {'status':'ok','msg':'Dados cadastrados com sucesso.'}

