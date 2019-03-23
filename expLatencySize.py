import os
from testEdge import uploadImg
#from cache import LFUCache
import cv2, logging, os, pickle, h5py,requests, sys, time
import numpy as np

class CacheNode(object):

	def __init__(self, key, value, freq_node, pre, nxt):
		self.key = key
		self.value = value
		self.freq_node = freq_node
		self.pre = pre
		self.nxt =nxt

	def free_myself(self):
		if (self.freq_node.cache_head == self.freq_node.cache_tail):
			self.freq_node.cache_head = None
			self.freq_node.cache_tail = None
		elif (self.freq_node.cache_head == self):
			self.nxt.pre = None
			self.freq_node.cache_head = self.nxt
		elif (self.freq_node.cache_tail == self):
			self.pre.nxt = None
			self.freq_node.cache_tail = self.pre
		else:
			self.pre.nxt = self.nxt
			self.nxt.pre = self.pre


		self.pre = None
		self.nxt = None
		self.freq_node = None


class FreqNode(object):

	def __init__(self, freq, pre, nxt):
		self.freq = freq
		self.pre = pre
		self.nxt = nxt
		self.cache_head = None
		self.cache_tail = None

	def count_caches(self):
		if (self.cache_head is None and self.cache_tail is None):
			return 0
		elif (self.cache_head == self.cache_tail):
			return 1
		else:
			return '2+'

	def remove(self):
		if(self.pre is not None):
			self.pre.nxt = self.nxt
		if(self.nxt is not None):
			self.nxt.pre = self.pre

		pre = self.pre
		nxt = self.nxt
		self.pre = self.nxt = self.cache_head = self.cache_tail = None

		return (pre, nxt)

	def pop_head_cache(self):
		if self.cache_head is None and self.cache_tail is None:
			return None
		elif self.cache_head == self.cache_tail:
			cache_head = self.cache_head
			self.cache_head = self.cache_tail = None
			return cache_head
		else:
			cache_head = self.cache_head
			self.cache_head.nxt.pre = None
			self.cache_head = self.cache_head.nxt
			return cache_head

	def append_cache_to_tail(self, cache_node):
		cache_node.freq_node = self

		if self.cache_head is None and self.cache_tail is None:
			self.cache_head = self.cache_tail = cache_node
		else:
			cache_node.pre = self.cache_tail
			cache_node.nxt = None
			self.cache_tail.nxt = cache_node
			self.cache_tail = cache_node

	def insert_after_me(self, freq_node):
		freq_node.pre = self
		freq_node.nxt = self.nxt

		if self.nxt is not None:
			self.nxt.pre = freq_node

		self.nxt = freq_node

	def insert_before_me(self, freq_node):
		if self.pre is not None:
			self.pre.nxt = freq_node
        
		freq_node.pre = self.pre
		freq_node.nxt = self
		self.pre = freq_node


class LFUCache(object):
	def __init__(self, capacity):
		self.cache = {}
		self.capacity = capacity
		self.freq_link_head = None

	def get(self, key, value, algMode):
		lenGoodList = []
		goodImgList = []

		for keyLine in self.cache.keys():
			good = []
			bad = []			
			descriptor = self.cache[keyLine].value

			if(algMode == 'bf'):
				bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
				matches = bf.knnMatch(value, descriptor, 2)

			elif(algMode == 'lsh'):
				FLANN_INDEX_LSH = 6
				index_params= dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
				search_params = dict(checks=50)
				flann = cv2.FlannBasedMatcher(index_params,search_params)
				matches = flann.knnMatch(value, descriptor, 2)

			else:
				return None


			for m, n in matches:
				if(m.distance < 0.7*n.distance):
					good.append(m.distance)

				else:
					bad.append(m.distance)

			if(len(good) > len(bad)):
				lenGoodList.append(len(good))
				goodImgList.append(keyLine)

		if(len(lenGoodList) > 0):
			maxLenGood = max(lenGoodList)
			imgRec = goodImgList[lenGoodList.index(maxLenGood)]
			return imgRec, self.cache[imgRec].freq_node.freq
		else:
			return None


	def set2(self, key, value):

		if (self.capacity <= 0):
			return -1

		else:
			if(len(self.cache) >= self.capacity):
				self.dump_cache()
			else:
				if(len(self.cache.keys()) == 0):
					self.createCache(key, value)
				else:
					cache_node = self.cache[key]
					freq_node = cache_node.freq_node
					value = cache_node.value
					self.moveForward(cache_node, freq_node)
					print("OI")


	def set(self, key, value):

		if (self.capacity <= 0):
			return -1

		else:
			#print("1")
			if (key not in self.cache):

				if(len(self.cache) >= self.capacity):

					self.dump_cache()

				else:
					#print("aqui")
					self.createCache(key, value)
			else:
				#print("2")
				cache_node = self.cache[key]
				freq_node = cache_node.freq_node
				value = cache_node.value

				self.moveForward(cache_node, freq_node)

	def moveForward(self, cache_node, freq_node):
		if freq_node.nxt is None or freq_node.nxt.freq != freq_node.freq + 1:
			target_freq_node = FreqNode(freq_node.freq + 1, None, None)
			target_empty = True
		else:
			target_freq_node = freq_node.nxt
			target_empty = False

		cache_node.free_myself()
		target_freq_node.append_cache_to_tail(cache_node)

		if (target_empty):
			freq_node.insert_after_me(target_freq_node)

		if(freq_node.count_caches() == 0):
			if(self.freq_link_head == freq_node):
				self.freq_link_head = target_freq_node

			freq_node.remove()

	def dump_cache(self):
		head_freq_node = self.freq_link_head
		self.cache.pop(head_freq_node.cache_head.key)
		head_freq_node.pop_head_cache()

		if (head_freq_node.count_caches() == 0):
			self.freq_link_head = head_freq_node.nxt
			head_freq_node.remove()

	def createCache(self, key, value):
		cache_node = CacheNode(key, value, None, None, None)
		self.cache[key] = cache_node

		if (self.freq_link_head is None or self. freq_link_head.freq != 0):
			new_freq_node = FreqNode(0, None, None)
			new_freq_node.append_cache_to_tail(cache_node)

			if(self.freq_link_head is not None):
				self.freq_link_head.insert_before_me(new_freq_node)

			self.freq_link_head = new_freq_node

		else:
			self.freq_link_head.append_cache_to_tail(cache_node)


class FeatureExtractor(object):

	def __init__(self, img):
		self.img = img

	def ORB (self):
		# Initiate STAR detector
		orb = cv2.ORB()

		# Find the keypoints with ORB
		kp = orb.detect(self.img, None)

		#compute the descriptors with ORB
		kp, des = orb.compute(self.img, kp)

		return kp, des

def setCache(file, imgName):
	try:
		capacity = 400
		dirname = os.path.dirname(__file__)
		if (file):
			file.save(os.path.join(dirname, file.filename))
			imgFile = cv2.imread(os.path.join(dirname, file.filename), 0)

		orb = cv2.ORB_create()
		kpData, desData = orb.detectAndCompute(imgFile, None)
		#cache = LFUCache(capacity)
		setTest = cache.set(file.filename, desData)
		return {'status':'ok','msg':'Dados cadastrados com sucesso.'}
	except Exception as e:
		print(e)
		return {'status':'error','msg':'Nao foi possivel cadastrar os dados.'}

def cloudModel(descriptor, imgName):
	try:
		goodImgList = []
		lenGoodList = []

		dirname = os.path.dirname(__file__)
		startCloud = time.time()
		hf = h5py.File('data.h5py', 'r')

		for line in list(hf.keys()):
			indexDes = hf[line][:]
			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
			matches = bf.knnMatch(indexDes, descriptor, 2)
			good = [] 
			for m,n in matches:
				if (m.distance < 0.5*n.distance):
					good.append(m.distance)

			lenGoodList.append(len(good))
			goodImgList.append(line)
		maxLenGood = max(lenGoodList)
		endCloud = time.time()
		cloudTime = endCloud - startCloud

		#print("o tamanho maximo e: %s"%maxLenGood)
		#print("o index do valor maximo e :%s"%lenGoodList.index(max(lenGoodList)))
		#print("a figura: %s"%(goodImgList[lenGoodList.index(max(lenGoodList))]))
		
		return {'status':'ok','time':cloudTime}

	except Exception as e:
		print(e)
		return {'status':'error','msg':'Nao foi possivel cadastrar os dados.'}



algMode = sys.argv[1]

dirname = os.path.dirname(__file__)
expImgDir = os.path.join(dirname, "exp")
getImgDir = os.path.join(dirname, "getExp")
url = 'http://localhost:5000/api/cachemiss'
capacityList = [50]


for capacity in capacityList:
	lookupTime0 = []
	lookupTime = []
	lookupTime2 = []
	cache = LFUCache(capacity)
	contacts = 0

	if(capacity > len(os.listdir(expImgDir))):
		for cont in range(0, capacity - len(os.listdir(expImgDir))):
			imgPathList = os.listdir(os.path.join(dirname, "trainDataset"))
			imgFile = imgPathList[np.random.randint(101, len(imgPathList))]

			picture = cv2.imread(os.path.join(dirname, "trainDataset", imgFile), 0)
			orb = cv2.ORB_create()
			kpData, desData = orb.detectAndCompute(picture, None)
			setCache = cache.set(imgFile, desData)


	for img in os.listdir(expImgDir):
		#cloudTime = 0
		#timeUpload = 0
		imgFile = os.path.join(expImgDir, img)
		picture = cv2.imread(imgFile, 0)
		orb = cv2.ORB_create()
		kpData, desData = orb.detectAndCompute(picture, None)
		setCache = cache.set(img, desData)


	for imgGet in os.listdir(getImgDir):
		cloudTime = 0
		timeUpload = 0

		startEdge = time.time()
		imgFile = os.path.join(getImgDir, imgGet)
		picture = cv2.imread(imgFile, 0)
		orb = cv2.ORB_create()
		kpData2, desData2 = orb.detectAndCompute(picture, None)
		imgRec, freqImcRec = cache.get(imgGet, desData2, algMode)
		endEdge = time.time()
		edgeTime = endEdge - startEdge

		if(value is None):
			timeUpload = uploadImg(url, imgFile, imgGet, capacity)
			contacts+=1
		endEdge2 = time.time()

		totalTime = edgeTime + timeUpload

		lookupTime0.append(edgeTime)
		lookupTime.append(totalTime)
		lookupTime2.append(endEdge2 - startEdge)
	print("numero de contato: %s"%contacts)
	print("0 - A media do tempo de busca na edge: %s"%(sum(lookupTime0)/len(lookupTime0)))	
	print("A media do tempo de busca na edge: %s"%(sum(lookupTime)/len(lookupTime)))
	print("2 - A media do tempo de busca na edge: %s"%(sum(lookupTime2)/len(lookupTime2)))
	break



