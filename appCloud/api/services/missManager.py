'''
Project Cloud Cachier 

Author: Roberto Pacheco
Date: February, 05, 2019
Modified: Roberto Pacheco
Date: February, 05, 2019

Description:
These functions are executed by the Cloud on Cachier's project
'''





from flask import jsonify, current_app as app
from .featureExtractionCloud import FeatureExtractor
import cv2, logging, os, pickle, h5py, time, json, requests, config
import numpy as np

def writeResultCloud(searchTime, fileJson, uploadTime):

	resultsPath = "./results/LFU/resultCloud_%s_%s.txt"%(fileJson["bw"], fileJson["latency"])
	result = searchTime + fileJson["searchEdgeTime"] + 2*uploadTime
	with open(resultsPath, "a") as f:
		f.write(str(result) + "\n")


def writeResult(result, latencyNet, uploadTime, bw):
	#Write some experiment's results in a file
	nameLatency = 1000*latencyNet
	dirname = os.path.dirname(__file__)
	latency = result + latencyNet + uploadTime
	print(result)
	print(latencyNet)
	print(uploadTime)
	resultsPath = "./results/cloud/result4_%s_%s"%(bw, 1000*latencyNet)
	with open(resultsPath, "a") as f:
		f.write(str(latency) + "\n")

def cloudModelCloud(file, uploadTime, fileJson):
	#receive a image from edge device
	try:

		imgName = file.filename
		dirname = os.path.dirname(__file__)
		latencyNet = fileJson["latency"]
		print("LATENCIA: %s"%(latencyNet))
		bw = fileJson["bw"]
		start = time.time() #start measure of latency
		
		goodImgList = []
		lenGoodList = []

		hf = h5py.File('data.h5py', 'r')
		imgFile = cv2.imread(os.path.join(dirname, "ReferenceDataset", file.filename), 0)
		
		orb = cv2.ORB_create()
		kpData, desData = orb.detectAndCompute(imgFile, None) #computes the descriptor of image

		# look up the received image in the reference dataset
		for line in list(hf.keys()): 
			indexDes = hf[line][:]
			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
			matches = bf.knnMatch(indexDes, desData, 2)
			good = [] 
			for m,n in matches:
				if (m.distance < 0.5*n.distance):
					good.append(m.distance)

			lenGoodList.append(len(good))
			goodImgList.append(line)
		maxLenGood = max(lenGoodList)
		end = time.time() # end measure of latency
		totaltime = end - start
		print("tempo de encontrar a imagem: %s"%totaltime)
		writeResult(totaltime, latencyNet, uploadTime, bw) #write the latency results
		print("o tamanho maximo e: %s"%maxLenGood)
		print("o index do valor maximo é :%s"%lenGoodList.index(max(lenGoodList)))
		print("a figura: %s"%(goodImgList[lenGoodList.index(max(lenGoodList))]))	
		#print("Cache: %s, latencia: %s "%(capacidade, totaltime))
		return {'status':'ok','msg':'Dados cadastrados com sucesso.'}
	except Exception as e:
		print(e)
		return {'status':'error','msg':'Não foi possível cadastrar os dados.'}



def uploadImg(url, img, imgName):
	#Send a image to a edge device
	try:
		files = {'media': open(img, 'rb'), "name": imgName}
		r = requests.post(url, files=files)
		if (r.status_code != 201 and r.status_code != 200):
			raise Exception('Received an unsuccessful status code of %s' % r.status_code)

	except Exception as err:
		print("error.")
		print(err.args)
		sys.exit()

	else:
		print("upload com sucesso")


def sendToEdge(url, imgName, fileJson, searchTime, uploadTime):
	headers = {'Content-Type' : 'application/json'}
	data = {"imgName": imgName, "latency":fileJson["latency"], "bw": fileJson["latency"], "timeMeasured": searchTime + uploadTime, "time": time.time()}
	r = requests.post(url, data=json.dumps(data), headers=headers)


def sendImgToEdge(url, imgPath, imgName, searchTime):
	dataDict = {"time": searchTime}
	files = {'file': (imgName, open(imgPath, 'rb'), 'image/x-png'), 'info': (json.dumps(dataDict), '1727968', 'text/plain;charset=ISO-8859-1')}
	r = requests.post(url, files=files)

def extractionFeature(img):
	sift = cv2.xfeatures2d.SIFT_create()
	kp, descriptors = sift.detectAndCompute(img, None)

	return kp, descriptors

def featureMatching(feature1, feature2):
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(feature1,feature2, k=2)
	return matches


def detectGoodMatch(matches):
	LOWE_RATIO = 0.75
	good = []
	for m,n in matches:
	    if m.distance < LOWE_RATIO*n.distance:
	        good.append(m)
	return good

def detectInliers(good, kp, kpRef):
	#fileList = []
	if len(good)> 0:
		#fileList.append(file)
		src_pts = np.float32([ kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kpRef[m.trainIdx] for m in good ]).reshape(-1,1,2)
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		matchesMask = mask.ravel().tolist()
		return sum(matchesMask)
	else:
		return None

def cloudModelLFU(file, fileJson, uploadTime):
	#receive a image from edge device
	try:
		urlToEdge = "http://0.0.0.0:5003/api/setcache"
		LOWE_RATIO = 0.75
		imgName = file.filename
		dirname = os.path.dirname(__file__)
		inliersList = []
		fileList = []
		print("fileeeeeeee:%s"%file)
		imgFile = cv2.imread(os.path.join(dirname, "ReferenceDataset", file.filename), 1)

		with open('cloudData.txt', 'rb') as f:
			dataSetFile = pickle.load(f, encoding='latin1')

		start = time.time() #start measure of latency
		kpQuery, feature = extractionFeature(imgFile)
		# look up the received image in the reference dataset
		for line in dataSetFile:
			try:
				kp = line["kp"]
				matches = featureMatching(feature, line["descriptors"])
				good = detectGoodMatch(matches)
				inliers = detectInliers(good, kpQuery, kp)

				if(inliers is not None):
					inliersList.append(inliers)
					fileList.append(file)
			except Exception as e:
				pass	
		
		if(len(inliersList) > 0):
			inliersListSorted = sorted(inliersList, reverse=True)
			imgRec = fileList[inliersList.index(inliersListSorted[0])]
			print(imgRec.filename)
		else:
			imgRec=None

		end = time.time() # end measure of latency
		searchTime = 1000*(end - start)
		if(imgRec is not None):
			imgRecPath = os.path.join(dirname, "ReferenceDataset", imgRec.filename)
			sendImgToEdge(urlToEdge, imgRecPath, imgRec.filename, searchTime)
		
		return {'status':'ok','msg':'Dados cadastrados com sucesso.'}
		#writeResultCloud(searchTime, fileJson, uploadTime)

		print("tempo de encontrar a imagem: %s"%searchTime)
		print("o tamanho maximo e: %s"%maxLenGood)
		print("o index do valor maximo é :%s"%lenGoodList.index(max(lenGoodList)))
		print("a figura: %s"%(imgFound))	
		return {'status':'ok','msg':'Dados cadastrados com sucesso.'}
	except Exception as e:
		print(e)
		return {'status':'error','msg':'Não foi possível cadastrar os dados.'}



def cloudModelCachier(file, fileJson, uploadTime):
	#receive a image from edge device
	try:

		dirname = os.path.dirname(__file__)
		urlEdge = "http://localhost:5020/api/checkcache"

		start = time.time() #start measure of latency
		goodImgList = []
		lenGoodList = []

		hf = h5py.File('data.h5py', 'r')

		imgFile = cv2.imread(os.path.join(dirname, "ReferenceDataset", file.filename), 0)
		
		orb = cv2.ORB_create()
		kpData, desData = orb.detectAndCompute(imgFile, None) #computes the descriptor of image

		# look up the received image in the reference dataset
		for line in list(hf.keys()): 
			indexDes = hf[line][:]
			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
			matches = bf.knnMatch(indexDes, desData, 2)
			good = [] 
			for m,n in matches:
				if (m.distance < 0.5*n.distance):
					good.append(m.distance)

			lenGoodList.append(len(good))
			goodImgList.append(line)
		maxLenGood = max(lenGoodList)
		imgFound = goodImgList[lenGoodList.index(max(lenGoodList))]

		end = time.time() # end measure of latency
		searchTime = 1000*(end - start)
		writeResultCloud(searchTime, fileJson, uploadTime) #write the latency results
		sendToEdge(urlEdge, imgFound, fileJson, searchTime, uploadTime)
		
		print("o tamanho maximo e: %s"%maxLenGood)
		print("o index do valor maximo é :%s"%lenGoodList.index(max(lenGoodList)))
		print("a figura: %s"%(goodImgList[lenGoodList.index(max(lenGoodList))]))	
		return {'status':'ok','msg':'Dados cadastrados com sucesso.'}
	except Exception as e:
		print(e)
		return {'status':'error','msg':'Não foi possível cadastrar os dados.'}





