import os

DATASETPATH = './dataset'

DEBUG = True


ALGMODE = 'bf'

INT_INF = 0
INT_SUP = 400
DIR_NAME = os.path.dirname(__file__)

RTT_DATASET = "rttDataSet.txt"
CACHE_FILE_CACHIER = "cachier.txt"


CACHE_FILE_LFU = os.path.join(DIR_NAME, "lfu.txt")

MAX_ITER = 10000
PRECISION = 1e-6
PREVIOUS_STEP_SIZE = 1.0
RATE_LEARNING = 0.1

HOST_CLOUD = "0.0.0.0"
PORT_CLOUD = 5002
URL_CLOUD = "http://%s:%s"%(HOST_CLOUD, PORT_CLOUD)



LOWE_RATIO = 0.75


if(ALGMODE == 'bf'):
	recall = lambda k: 0.66 - 0.00013*k
	function = lambda k: 60.52 + 10.42*k

elif(ALGMODE == 'lsh'):
	recall = lambda k: 0.90 - 0.00014*k
	function = lambda k: 51.14 + 1.87*k

else:
	function = None

