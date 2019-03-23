import cv2, operator, pickle, os, config, json
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

	def updateCapacity(self, capacity, filename):
		self.capacity = capacity
		newCacheDict = {}
		newCache = LFUCache(capacity)

		for key in self.cache.keys():
			newCacheDict.update({key: self.cache[key].freq_node.freq})

		freq_sorted = sorted(newCacheDict.items(), key=operator.itemgetter(1), reverse=True)
		newCacheDict = dict(freq_sorted[0:self.capacity])

		for key in newCacheDict.keys():
			newCache.set(key, self.cache[key].value)

		with open(filename, 'wb') as f:
			pickle.dump(newCache, f)
		return newCache

	def getId(self, nameImg):

		for keyLine in self.cache.keys():
			print(keyLine)
			if(nameImg == keyLine):
				return True

		return False

				
	def get(self, key, value, algMode='bf'):
		fileList = []
		inliersList = []
		LOWE_RATIO = 0.75
		self.freqImg = []
		for keyLine in self.cache.keys():
			good = []			
			descriptor = self.cache[keyLine].value["descriptors"]
			kpRef = self.cache[keyLine].value["kp"]
			self.freqImg.append(self.cache[keyLine].freq_node.freq)

			if(algMode == 'bf'):
				bf = cv2.BFMatcher()
				matches = bf.knnMatch(value["descriptors"], descriptor, k=2)

			elif(algMode == 'lsh'):
				FLANN_INDEX_LSH = 6
				index_params= dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
				search_params = dict(checks=50)
				flann = cv2.FlannBasedMatcher(index_params,search_params)
				matches = flann.knnMatch(value, descriptor, 2)

			else:
				return None

			for m,n in matches:
				if m.distance < LOWE_RATIO*n.distance:
					good.append(m)

			if len(good)> 0:
				src_pts = np.float32([ value["kp"][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
				dst_pts = np.float32([ kpRef[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
				M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
				matchesMask = mask.ravel().tolist()
				inliersList.append(sum(matchesMask))
				fileList.append(keyLine)
		if(len(inliersList) > 0):
			inliersListSorted = sorted(inliersList, reverse=True)
			imgRec = fileList[inliersList.index(inliersListSorted[0])]
			return {"imgRec":imgRec, "freqImg": self.cache[imgRec].freq_node.freq, "sumFreqList":sum(self.freqImg),"lenFreqList":len(self.freqImg)}
		else:
			return None


	def set(self, key, value):

		if (self.capacity <= 0):
			return -1

		else:
			if (key not in self.cache):

				if(len(self.cache) >= self.capacity):

					self.dump_cache()

				else:
					#print(value)
					self.createCache(key, value)
			else:
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
		#print(cache_node)
		self.cache[key] = cache_node

		if (self.freq_link_head is None or self. freq_link_head.freq != 0):
			new_freq_node = FreqNode(0, None, None)
			new_freq_node.append_cache_to_tail(cache_node)

			if(self.freq_link_head is not None):
				self.freq_link_head.insert_before_me(new_freq_node)

			self.freq_link_head = new_freq_node

		else:
			self.freq_link_head.append_cache_to_tail(cache_node)

def extractionFeature(img):
	sift = cv2.xfeatures2d.SIFT_create()
	kp, descriptors = sift.detectAndCompute(img, None)

	return kp, descriptors

cache = LFUCache(400)

dirname = os.path.dirname(__file__)
trainDataSetPath = os.path.join(dirname, "trainDataset", '1.jpg')
img = cv2.imread(trainDataSetPath, 1)
kp, descriptors = extractionFeature(img)
value = {"kp": kp, "descriptors": descriptors}
cache.set('1.jpg', value)
pickle.dump( cache, open( "lfu.txt", "wb" ) )
#cache = pickle.load( open( "save.p", "rb" ) )
#print(cache)



