from flask import jsonify, session, current_app as app
from .featureExtractionCloud import FeatureExtractor
from cache import LFUCache
import cv2, logging, os, pickle, h5py,requests, sys, config, time
import numpy as np, json
#import lfucache.lfu_cache as lfu_cache
#from sklearn import linear_model
from scipy.misc import derivative
#from expLatencySize import cache




def setCache(file, fileJson):
	try:
		capacity = 400
		searchTime = fileJson["time"]
		dirname = os.path.dirname(__file__)
		filePath = os.path.join(dirname, file.filename)
		if (file):
			file.save(filePath)
			imgFile = cv2.imread(filePath, 0)

		orb = cv2.ORB_create()
		kpData, desData = orb.detectAndCompute(imgFile, None)
		with open(config.CACHE_FILE_LFU, 'rb') as f:
			cache = pickle.load(f)

		setTest = cache.set(file.filename, desData)
		return {'status':'ok','msg':'Dados cadastrados com sucesso.'}
	except Exception as e:
		print(e)
		return {'status':'error','msg':'Não foi possível cadastrar os dados.'}


def derivative(f, a, h=0.01):
	return (f(a+h) - f(a-h))/2*h


def gradientDescent(k0, function, recall, pCached, rttMean):
	#E[L] = f(k)+ (1−recall(k)∗P(cached))∗(LNet +LCloud)
	iters = 0
	k = np.arange(0, 400)
	ki = k0
	previous_step_size = config.PREVIOUS_STEP_SIZE

	estimatedLatency = lambda k: function(k) + (1 - recall(k)*pCached)*(rttMean)
	y = estimatedLatency(k)

	while(previous_step_size > config.PRECISION):
		prevK = ki
		ki = ki - (config.RATE_LEARNING*derivative(estimatedLatency, prevK))
		print(ki)
		previous_step_size = abs(ki - prevK)
		iters+=1

	print("O minimo local ocorre em: %s"%(ki))

	return ki

def probCache(freqImcRec, sumFreq, lenFreq):
	#pi = Mi+alphai/sum(Mi)+sum(alphai)
	alphai = 1.0
	probCache = (freqImcRec+alphai)/(sumFreq+(lenFreq* alphai))
	return probCache


def cloudNetProfilerWrite(filename, rtt):
	
	with open(filename, 'r+b') as file:
		data = pickle.load(file)
		data["rtt"].append(rtt)
		pickle.dump(data, file)

def cloudNetProfilerMean(filename):

	headers = {'Content-Type' : 'application/json'}
	dictTest = {'name': 'test'}
	url = '%s/api/test'%(config.URL_CLOUD)
	r = requests.post(url, data=json.dumps(dictTest), headers=headers)
	rtt = r.elapsed.total_seconds()
	print(rtt)
	cloudNetProfilerWrite(filename, rtt)
	with open(filename, 'rb') as file:
		data = pickle.load(file)
		print(data)
		mediaRtt = sum(data["rtt"])/len(data["rtt"])

	print(mediaRtt)

	return mediaRtt


def estimateLatency(x0, freqImcRec, sumFreq, lenFreq):
	#E[L] = f(k)+ (1−recall(k)∗P(cached))∗(LNet +LCloud)
	
	if(config.ALGMODE == 'bf'):
		recall = lambda k: 0.66 - 0.00013*k
		function = lambda k: 60.52 + 10.42*k
	
	elif(config.ALGMODE == 'lsh'):
		recall = lambda k: 0.90 - 0.00014*k
		function = lambda k: 51.14 + 1.87*k
	else:
		return None
	
	#k = np.arange(intInf, intSup + 1)
	k = np.arange(config.INT_INF, config.INT_SUP + 1)

	rttMean = cloudNetProfilerMean(config.RTT_DATASET)
	pCached = probCache(freqImcRec, sumFreq, lenFreq)

	print("A probbilidade da imagem estar no cache: p[cached] = %s"%pCached)

	#expectedLatency = function(k) + (1 - recall(k)*pCached)*(rttMean)
	#print(expectedLatency)
	k_min,  = gradientDescent(x0, function, recall, pCached, rttMean)
	
	return k_min


def uploadImg(url, img, imgName, fileJson, searchEdgeTime):
	try:
		dataDict = {"time": time.time(), "latency":fileJson["latency"], "bw": fileJson["bw"], "searchEdgeTime": searchEdgeTime}
		files = {'file': (imgName, open(img, 'rb'), 'image/x-png'), 'info': (json.dumps(dataDict), '1727968', 'text/plain;charset=ISO-8859-1')}
		r = requests.post(url, files=files)
		rtt = r.elapsed.total_seconds()
		
		if (r.status_code != 201 and r.status_code != 200):
			raise Exception('Received an unsuccessful status code of %s' % r.status_code)

	except Exception as err:
		print("error.")
		print(err.args)
		sys.exit()

	else:
		#print("upload com sucesso")
		return rtt

def writeResults(searchEdgeTime, fileJson):
	
	fileResults = os.path.join(config.DIR_NAME, "results", "LFU")
	resultsPath = os.path.join(fileResults, "resultEdge_%s_%s.txt"%(fileJson["bw"], fileJson["latency"]))

	with open(resultsPath, "a") as f:
		f.write("%s\n"%(searchEdgeTime))

def setupNetwork(latency, bw):
	os.popen("sudo tc qdisc del dev wlp2s0 root")
	os.popen("sudo tc qdisc add dev wlp2s0 root handle 1: tbf rate %smbit burst 100mb latency 1ms"%bw)
	os.popen("sudo tc qdisc add dev wlp2s0 parent 1:1 handle 10: netem delay %sms"%(latency))

def extractionFeature(img):
	sift = cv2.xfeatures2d.SIFT_create()
	kp, descriptors = sift.detectAndCompute(img, None)

	return kp, descriptors

def edgeLFU(file, fileJson, referenceDatasetPath, startTime):
	try:
		print("query:%s"%file)
		url = '%s/api/cachemissLFU'%(config.URL_CLOUD)
		capacity = 400
		
		with open(config.CACHE_FILE_LFU, 'rb') as f:
			cache = pickle.load(f)

		imgFile = cv2.imread(referenceDatasetPath, 0)
		kpData, desData = extractionFeature(imgFile)
		print(desData)
		value = {"kp":kpData, "descriptors":desData}
		imgRec = cache.get(file.filename, value)
		end = time.time()
		searchEdgeTime = 1000*(end - startTime)
		#print("O tempo de busca: %s"%(searchEdgeTime))
		
		if (imgRec is None):
			#setupNetwork(fileJson["latency"], fileJson["bw"])
			rtt = uploadImg(url, referenceDatasetPath, file.filename, fileJson, searchEdgeTime)
			cache.set(file.filename, value)
			pickle.dump( cache, open(config.CACHE_FILE_LFU, "wb" ) )
		rtt = uploadImg(url, referenceDatasetPath, file.filename, fileJson, searchEdgeTime)
		#newCache = cache.updateCapacity(capacity, config.CACHE_FILE_LFU)
		
		#writeResults(fileResults, searchEdgeTime, fileJson)

		return {'status':'ok','msg':'Dados encontrados com sucesso.'}
	except Exception as e:
		print(e)
		return {'status':'error','msg':'Não foi possível encontrar os dados.'}


def edgeCachier(file, fileJson, referenceDatasetPath, startTime):
	try:

		url = '%s/api/cachemisscachier'%(config.URL_CLOUD)
		url_miss = "%s/api/requestimage"%(config.URL_CLOUD)
		x0 = 1.0
		
		fileResults = os.path.join(config.DIR_NAME, "results", "cachier")

		with open(config.CACHE_FILE_CACHIER, 'rb') as f:
			cache = pickle.load(f)

		imgFile = cv2.imread(referenceDatasetPath, 0)

		orb = cv2.ORB_create()
		kpData, desData = orb.detectAndCompute(imgFile, None)
		imgRec = cache.get(file.filename, desData)

		end = time.time()
		searchEdgeTime = 1000*(end - startTime)

		if (imgRec is None):
			setupNetwork(fileJson["latency"], fileJson["bw"])
			rtt = uploadImg(url, referenceDatasetPath, file.filename, fileJson, searchEdgeTime)
			cloudNetProfilerWrite(config.RTT_DATASET, rtt)
			setTest = cache.set(file.filename, desData)

		else:
			freqImgRec = imgRec["freqImg"]
			sumFreq = imgRec["sumFreqList"]
			lenFreq = imgRec["lenFreqList"]
			k_min = estimateLatency(x0, freqImgRec, sumFreq, lenFreq)
			setTest = cache.set(file.filename, desData)
			newCache = cache.updateCapacity(k_min, config.CACHE_FILE_CACHIER)

		writeResults(fileResults, searchEdgeTime, fileJson)

		return {'status':'ok','msg':'Dados encontrados com sucesso.'}

	except Exception as e:
		print(e)
		return {'status':'error','msg':'Não foi possível encontrar os dados.'}


def checkCache(nameImg):
	try:
		with open(config.CACHE_FILE, 'rb') as file:
			cache = pickle.load(file)

		if(cache.getId(nameImg) == False):
			headers = {'Content-Type' : 'application/json'}
			dictImg = {'name': nameImg}
			url = '%s/requestImage'%(config.URL_CLOUD)
			r = requests.post(url, data=json.dumps(dictImg), headers=headers)

		if (r.status_code != 201 and r.status_code != 200):
			raise Exception('Received an unsuccessful status code of %s' % r.status_code)

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

