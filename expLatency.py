import os
#from testEdge import uploadImg
#from cache import LFUCache
import cv2, logging, os, pickle, requests, sys, time
import numpy as np
import subprocess, json, math, bisect, random
from functools import reduce
from scipy.stats import zipf
import matplotlib.pyplot as plt

def uploadImg(url, img, imgName, latency, bw):
	try:
		header = {"Content-Type":"application/json"}
		dataDict = {"time": time.time(), "latency": latency, "bw": bw}
		print("AQUI")
		files = {'file': (imgName, open(img, 'rb'), 'image/x-png'), 'info': (json.dumps(dataDict), '1727968', 'text/plain;charset=ISO-8859-1')}
		r = requests.post(url, files=files)

		if (r.status_code != 201 and r.status_code != 200):
			raise Exception('Received an unsuccessful status code of %s' % r.status_code)

	except Exception as err:
		print("error.")
		print(err.args)
		sys.exit()
	else:
		print("upload com sucesso")

def getImg(path):
	imgList = os.listdir(path)
	nrRandom = np.random.randint(0, len(imgList))
	img = imgList[nrRandom]
	return os.path.join(path, img), img
"""
class ZipfGenerator(object):
	def __init__(self, n, alpha):
		tmp = [1.0/(math.pow(float(i), alpha)) for i in range(1, n+1)]
		print([tmp[-1]])
		zeta = reduce(lambda sums, x: sums + [sum[-1] + x], tmp, 0)
		#zeta  =lambda sums, x: sums + [sum[-1] + x]
		self.distMap = [x/zeta[-1] for x in zeta]

	def next(self):
		u = random.random()

		return bisect.bisect(self.distMap, u) - 1
"""



class ZipfGenerator(object):
	def __init__(self, n, alpha):
		self.n = n
		self.alpha = alpha
		harmonic = [math.pow(float(i), -alpha) for i in range(1, n + 1)]
		self.tmp = [math.pow(float(i), -alpha)/sum(harmonic) for i in range(1, n + 1)]

	def getProb(self):
		return self.tmp

	def generateZipf(self):
		r = np.random.choice(np.arange(1, self.n + 1), p=self.tmp)
		return r

alpha = 1
MAX_QNTD_REQUEST = 5000
qntdRequest = 0
nrExp = 4
dirname = os.path.dirname(__file__)
imgPath = os.path.join(dirname, "trainDataset")
sizeTrainDataset = len(os.listdir(imgPath))
sizeTrainDataset = 10
latency = 10
bw = 8


if (sys.argv[1] == "cloud"):
	url = "http://localhost:5000/api/cachemissLFU"

elif (sys.argv[1] == "lfu"):
	url = "http://localhost:6025/api/lfumodel"

elif (sys.argv[1] == "cachier"):
	url = "http://localhost:6025/api/cachiermodel"

else:
	print("ERROR ")
	sys.exit()	


z = ZipfGenerator(sizeTrainDataset, alpha)

#os.popen("sudo tc qdisc del dev lo root")
#os.popen("sudo tc qdisc add dev lo root handle 1: tbf rate %smbit burst 100mb latency 1ms"%bw)
#os.popen("sudo tc qdisc add dev lo parent 1:1 handle 10: netem delay %sms"%(latency))
#os.popen("sudo tc qdisc add dev wlp2s0 root handle 1: tbf rate %smbit")%(bw)
#os.popen("sudo tc qdisc add dev wlp2s0 parent 1:1 handle 10: netem delay %sms")%(latency)

#subprocess.Popen("sudo tc qdisc add dev eth2 root netem delay %sms")%(latency)
#subprocess.call("sudo tc qdisc add dev eth2 root netem delay %sms")%(latency)

for inter in range(0, nrExp):
	while (qntdRequest < MAX_QNTD_REQUEST):
		nrRequestZipf = z.generateZipf()
		img, imgName = getImg(imgPath)
		for nrRequest in range(0, nrRequestZipf):
			print("A imagem enviada: %s"%imgName)
			img, imgName = getImg(imgPath)
			uploadImg(url, img, imgName, latency, bw)	
			qntdRequest += nrRequest
