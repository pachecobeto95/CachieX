import cv2, pickle, os, h5py
import numpy as np





dirname = os.path.dirname(__file__)
referenceDataSet = os.path.join(dirname, "ReferenceDataset")
orb = cv2.ORB_create()
indexList = []
f = h5py.File("data.h5py", 'w')

for imgFileName in os.listdir(referenceDataSet):

	imgPath = os.path.join(referenceDataSet, imgFileName)

	trainImg = cv2.imread(imgPath, 0)

	kp, des = orb.detectAndCompute(trainImg, None)
	if (des is not None):
		f.create_dataset(imgFileName, data=des, shape=(500, 32))
f.close()



"""
with open(os.path.join(dirname, "featuresDES.txt"), "ab") as f:
	#pickle.dump(len(des), f)
	pickle.dump(des, f)
	pickle.dump('\n', f)
	pickle.dump(des2, f)

"""

"""
for img in os.listdir(referenceDataSet):
	index = []
	imgPath = os.path.join(referenceDataSet, img)
	trainImg = cv2.imread(imgPath, 0)
	kp, des = orb.detectAndCompute(trainImg, None)

	for point in kp:
		temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
		index.append(temp)

	f = open(os.path.join(dirname, "featuresKP.txt"), "a")
	fDes = open(os.path.join(dirname, "featuresDes.txt"), "a")
	f.write(pickle.dumps(index))
	fDes.write(pickle.dumps(index))
	f.close()
	fDes.close()

"""


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