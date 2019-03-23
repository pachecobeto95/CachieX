import os


ALGMODE = 'bf'

INT_INF = 0
INT_SUP = 400

PICKLE_FILE = "rttDataSet.txt"
CACHE_FILE = "cache.txt"
CACHE_FILE_LFU = "lfu.txt"


MAX_ITER = 1000
PRECISION = 1e-6
PREVIOUS_STEP_SIZE = 1e-6
RATE_LEARNING = 0.01

HOST_CLOUD = "localhost"
PORT_CLOUD = 5000
URL_CLOUD = "http://%s:%s"%(HOST_CLOUD, PORT_CLOUD)

DIR_NAME = os.path.dirname(__file__)

if(algMode == 'bf'):
	recall = lambda k: 0.66 - 0.00013*k
	function = lambda k: 60.52 + 10.42*k

elif(algMode == 'lsh'):
	recall = lambda k: 0.90 - 0.00014*k
	function = lambda k: 51.14 + 1.87*k

else:
	function = None

