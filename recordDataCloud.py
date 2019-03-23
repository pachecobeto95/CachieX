import os, cv2, pickle, sys
dirname = os.path.dirname(__file__)
referenceDatasetPath = os.path.join(dirname, "appCloud", "api", "services", "ReferenceDataset")

def extractionFeature(img):
	sift = cv2.xfeatures2d.SIFT_create()
	kp, descriptors = sift.detectAndCompute(img, None)

	return kp, descriptors

dataList = []
with open("cloudData.txt", "wb") as f:
	for file in sorted(os.listdir(referenceDatasetPath)):
		print(file)
		imgPath = os.path.join(referenceDatasetPath, file)
		img = cv2.imread(imgPath, 1)
		kp, des = extractionFeature(img)
		kpList = [i.pt for i in kp]
		value = {"file":file, "kp":kpList, "descriptors": des}
		dataList.append(value)
	pickle.dump(dataList, f, protocol=pickle.HIGHEST_PROTOCOL)

"""
with open('cloudData.txt', 'rb') as f:
    b = pickle.load(f)

for i in list(b):
	kp = i["kp"]
	for j in kp:
		print(j.pt)
	sys.exit()
"""