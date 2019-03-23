import numpy as np
import cv2, os, requests, sys, logging
from matplotlib import pyplot as plt
from cache import LFUCache

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


alpha = 1.5
capacity = 400
url = 'http://localhost:5000/api/cachemiss'
nrRequestZipf = np.random.zipf(alpha, size=None)
dirname = os.path.dirname(__file__)
imageDir = os.path.join(dirname, "trainDataset")
imgList = os.listdir(imageDir)
cacheTest = LFUCache(capacity)


for nrRequest in range(0, nrRequestZipf):
	imgName = imgList[np.random.randint(0, len(imgList))]
	print(imgName)
	img = os.path.join(imageDir, imgName)
	uploadImg(url, img, imgName)
	break




	



