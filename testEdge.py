import numpy as np
import cv2, os, requests, sys, logging, time, pickle, random, json
from matplotlib import pyplot as plt
from cache import LFUCache
#from sklearn import linear_model

"""
refImg = cv2.imread('./004.jpg',0)          # queryImage
trainImg = cv2.imread('4.jpg',0) # trainImage
orb = cv2.ORB_create() 

kp1, des1 = orb.detectAndCompute(refImg,None)
kp2, des2 = orb.detectAndCompute(trainImg,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x:x.distance)
img3 = np.array([])
img3 = cv2.drawMatches(refImg,kp1,trainImg,kp2,matches[:10], img3)

plt.imshow(img3)
plt.show()
"""

def uploadImg(url, img, imgName):
	try:
		start = time.time()
		dataDict = {'latency':100, 'bw': 1}
		files = {'file': (imgName, open(img, 'rb'), 'image/x-png'), 'info': (json.dumps(dataDict), '1727968', 'text/plain;charset=ISO-8859-1')}
		r = requests.post(url, files=files)

		if (r.status_code != 201 and r.status_code != 200):
			raise Exception('Received an unsuccessful status code of %s' % r.status_code)
		else:
			end = time.time()
			return end - start

	except Exception as err:
		print("error.")
		print(err.args)
		sys.exit()

dirname = os.path.dirname(__file__)
url = "http://0.0.0.0:5000/api/lfumodel"

imgDir = os.path.join(dirname, "appCloud", "api", "services", "ReferenceDataset")
trainDataSetPath = os.path.join(dirname, "trainDataset")
trainDataSetPathList = os.listdir(trainDataSetPath)
random.shuffle(trainDataSetPathList)
zipfIndex = np.random.zipf(2)

for i in range(5):
	if(zipfIndex < len(trainDataSetPathList)):
		imgName = trainDataSetPathList[zipfIndex]
		imgPath = os.path.join(trainDataSetPath, imgName)
		uploadImg(url, imgPath, imgName)