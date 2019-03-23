from flask import jsonify, session, current_app as app
import cv2, logging, os, pickle, h5py,requests, sys, config, time
import numpy as np, json
#import lfucache.lfu_cache as lfu_cache
#from sklearn import linear_model
#from scipy.misc import derivative
#from expLatencySize import cache

def extractionFeature(img):
	sift = cv2.xfeatures2d.SIFT_create()
	kp, descriptors = sift.detectAndCompute(img, None)

	return kp, descriptors

def setCacheLFU(file, fileJson):
	try:
		capacity = 400
		searchTime = fileJson["time"]
		dirname = os.path.dirname(__file__)
		filePath = os.path.join(config.DIR_NAME, "appEdge", "api", file.filename)
	
		file.save(filePath)
		imgFile = cv2.imread(filePath, 1)

		kpData, desData = extractionFeature(imgFile)
		with open(config.CACHE_FILE_LFU, 'rb') as f:
			cache = pickle.load(f)

		cache.set(file.filename, desData)
		pickle.dump(cache, open(config.CACHE_FILE_LFU,'wb'))
		return {'status':'ok','msg':'Dados cadastrados com sucesso.'}
	except Exception as e:
		print(e)
		return {'status':'error','msg':'Não foi possível cadastrar os dados.'}


def sendRequestToCloud(url, imgName, timeMeasured):
	headers = {'Content-Type' : 'application/json'}
	dictImg = {'imgName': imgName, "timeMeasured": timeMeasured, "time": time.time()}
	r = requests.post(url, data=json.dumps(dictImg), headers=headers)
	if (r.status_code != 201 and r.status_code != 200):
		raise Exception('Received an unsuccessful status code of %s' % r.status_code)


def checkCache(imgName, uploadTime, timeMeasured):
	try:
		url = 'http://192.168.0.8:5002/api/requestimage'

		start = time.time()
		with open(config.CACHE_FILE_CACHIER, 'rb') as file:
			cache = pickle.load(file)
		end = time.time()
		searchTime = 1000*(end - start)

		if(cache.getId(imgName) == False):
			sendRequestToCloud(url, imgName, searchTime + timeMeasured)

	except Exception as err:
		print("error")
		print(err.args)
		return {'status':'error','msg': err.args}

	else:
		return {'status':'ok','msg':'Dados cadastrados com sucesso.'}

def saveImage(file, imgName):
	try:
		if (file):
			file.save(os.path.join(config.DIR_NAME, file.filename))

	except Exception as e:
		return {'status':'error','msg': err.args}

	else:
		return {'status':'ok','msg':'Dados cadastrados com sucesso.'}

