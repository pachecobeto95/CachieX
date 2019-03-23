import cv2, os, sys

import os, cv2, imutils, math, sys, random
import numpy as np
import matplotlib.pyplot as plt
from imutils import perspective
from imutils import contours
from imutils.object_detection import non_max_suppression

def alignImage(img, imgRef):
	orb = cv2.ORB_create()
	keyPoint, des = orb.detectAndCompute(img, None)
	keyPointRef, desRef = orb.detectAndCompute(imgRef, None)
	matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	matches = matcher.match(des, desRef, None)
	matches.sort(key=lambda x: x.distance, reverse=False)
	#numGoodMatches = int(len(matches) * 0.15)
	matches = matches[:4]
	imgMatches = cv2.drawMatches(img, keyPoint, imgRef, keyPointRef, matches, None)
	cv2.imwrite("output.jpg", imgMatches)

	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)
	for i, match in enumerate(matches):
		points1[i, :] = keyPoint[match.queryIdx].pt
		points2[i, :] = keyPointRef[match.trainIdx].pt

	h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
	print(h)
	height, width = img.shape

	M = np.eye(3)

	imgReg = cv2.warpPerspective(img, M, (width, height))
	cv2.imwrite("saida2.jpg", imgReg)
	
	return imgReg, h

def detectAndDescribe(image):
	orb = cv2.ORB_create()
	kps, des = orb.detectAndCompute(image, None)
	kps = np.float32([kp.pt for kp in kps])
	return kps, des

def matchFeatures(kps1, kps2, features1, features2, ratio=0.7, reproj=4.0):
	matcher = cv2.DescriptorMatcher_create("BruteForce")
	rawMatches = matcher.knnMatch(features1, features2, 2)
	matches = []

	for m in rawMatches:
		if(len(m) == 2 and m[0].distance < m[1].distance*ratio):
			matches.append((m[0].trainIdx, m[0].queryIdx))

	pts1 = np.float32([kps1[i] for (_, i) in matches])
	pts2 = np.float32([kps2[i] for (i, _) in matches])

	h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, reproj)
	return matches, h, mask

def warpImages(img1, img2, H):
	rows1, cols1 = img1.shape[:2]
	rows2, cols2 = img2.shape[:2]

	listPoints1 = np.float32([[0,0], [0,rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
	temp_points = np.float32([[0,0], [0,rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

	list_of_points_2 = cv2.perspectiveTransform(temp_points, H)


def preProcessing(img):
	boxList = []
	area = []
	img = cv2.GaussianBlur(img, (5,5), 0)
	edged = cv2.Canny(img, 50, 100)
	thresh = cv2.threshold(edged, 60, 255, cv2.THRESH_BINARY)[1]
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.04*peri, True)
		if(len(approx) == 4):
			shape = "rectangle"
			x,y,w,h = cv2.boundingRect(approx)
			box = cv2.minAreaRect(c)
			box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
			box = np.array(box, dtype="int")
			box = perspective.order_points(box)
			area.append(cv2.contourArea(box))
			boxList.append(box)
	areaMax = area.index(max(area))
	box = boxList[area.index(max(area))]
	mask = np.zeros(img.shape, dtype = "uint8")
	cv2.drawContours(mask, [box.astype("int")], -1, (255,255,255), -1)
	dst = cv2.bitwise_and(img, mask)
	#cv2.imwrite("k.png", dst)
	return dst

def calcHist(img):
	histList = []
	hist = cv2.calcHist([img],[0, 1, 2], None,[16, 16, 16],[0,256, 0, 256, 0, 256])
	hist = cv2.normalize(hist, hist).flatten()
	for h in hist:
		if(h>0):
			histList.append(h)
	histList = np.asarray(histList)
	
	#plt.hist(img.ravel(),256,[0,256]); plt.show()
	return histList

def shapeDetector(c):
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.04*peri, True)
	if(len(approx) == 4):
		shape = "rectangle"
		rect = cv2.boundingRect(approx)
	return shape, rect

def extractionFeature(img):
	sift = cv2.xfeatures2d.SIFT_create()
	kp, descriptors = sift.detectAndCompute(img, None)

	return kp, descriptors

def featureMatching(feature1, feature2):
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(feature1,feature2, k=2)
	return matches

def compareHist(hist, histRef):
	if(len(histRef) >= len(hist)):
		histRef = histRef[:len(hist)]
	else:
		hist = hist[:len(histRef)]
	correlation = cv2.compareHist(hist, histRef, cv2.HISTCMP_CHISQR)
	return correlation

def detectGoodMatch(matches):
	for m,n in matches:
	    if m.distance < LOWE_RATIO*n.distance:
	        good.append(m)
	return good

def detectInliers(good, kp, kpRef):
	#fileList = []
	if len(good)> 0:
		#fileList.append(file)
		src_pts = np.float32([ kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kpRef[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		matchesMask = mask.ravel().tolist()
		print(sum(matchesMask))
		return sum(matchesMask)
	else:
		return None

LOWE_RATIO = 0.75
#draw_params = dict(matchColor = (0,255,0), singlePointColor = None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)



dirname = os.path.dirname(__file__)

imgDir = os.path.join(dirname, "appCloud", "api", "services", "ReferenceDataset")
trainDataSetPath = os.path.join(dirname, "trainDataset")

inliersList = []
fileList = []
imgPath = os.path.join(trainDataSetPath, '4.jpg')
imgQuery = cv2.imread(imgPath, 1)
kp, feature = extractionFeature(imgQuery)
for file in sorted(os.listdir(imgDir), reverse=False):
	try:
		print("Arquivo: %s"%file)
		imgReferencePath = os.path.join(imgDir, file)
		imgReference = cv2.imread(imgReferencePath, 1)
		kpRef, featureRef = extractionFeature(imgReference)	
		for i in kpRef:
			print(i.pt)
		sys.exit()
		matches = featureMatching(feature, featureRef)
		good = []
		good = detectGoodMatch(matches)
		inliers = detectInliers(good, kp, kpRef)
		if(inliers is not None):
			#print("inliers: %s"%inliers)
			inliersList.append(inliers)
			fileList.append(file)
	except Exception as e:
		pass					
	
inliersListSorted = sorted(inliersList, reverse=True)
fileFound = fileList[inliersList.index(inliersListSorted[0])]
print(fileFound)


"""
inliersListSorted = sorted(inliersList, reverse=True)
print(fileList[inliersList.index(inliersListSorted[0])])
if(inliersList[0] - inliersList[1] > 100):
	filePath0 = fileList[inliersList.index(inliersListSorted[0])]
	img0 = cv2.imread(os.path.join(imgDir, filePath0), 1)
	hist0 = calcHist(img0)
	filePath1 = fileList[inliersList.index(inliersListSorted[1])]
	img1 = cv2.imread(os.path.join(imgDir, filePath0), 1)
	hist1 = calcHist(img1)

	correlation0 = compareHist(hist, hist0)
	correlation1 = compareHist(hist, hist1)

	if(correlation0 > correlation1):
		print(filePath0)
	else:
		print(filePath1)
"""



"""
imgDirSorted = sorted(os.listdir(imgDir), reverse=False)

#for cont in range(0, 10):
#	fileQuery = os.listdir(trainDir)[random.randint(1, sizeTrainDir)]
#	print("Arquivo query: %s"%fileQuery)
#	fileQueryPath = os.path.join(trainDir, fileQuery)
fileQueryPath = os.path.join(trainDir, "1.jpg")
imgFile = cv2.imread(fileQueryPath, 1)
hist = calcHist(imgFile)
kp, feature = extractionFeature(imgFile)
fileList = []
inliersList = []
"""
"""	
for file in sorted(os.listdir(imgDir), reverse=False):
	#print("Arquivo: %s"%imgDirSorted[file])
	try:
		print("Arquivo: %s"%file)
		imgReferencePath = os.path.join(imgDir, file)
		#imgReferencePath = os.path.join(imgDir, imgDirSorted[file])
		imgReference = cv2.imread(imgReferencePath, 1)
		histRef = calcHist(imgReference)
		cmpHist = compareHist(hist, histRef)
		print("distancia: %s"%cmpHist)	
		cmpHistList.append(cmpHist)
		mediaCmpHist = sum(cmpHistList)/len(cmpHistList)
			
		if(cmpHist <= mediaCmpHist):
			kpRef, featureRef = extractionFeature(imgReference)		
			matches = featureMatching(feature, featureRef)
			good = []
			good = detectGoodMatch(matches)
			print("Good: %s"%len(good))
			inliers = detectInliers(good)
			print("inliers: %s"%inliers)
			inliersList.append(inliers)

	except Exception as e:
		pass					
	
inlierSorted = sorted(inliersList, reverse=True)[:5]
print(inlierSorted)
topFile = [fileList[inliersList.index(inlier)] for inlier in inlierSorted]
print(topFile)
"""
