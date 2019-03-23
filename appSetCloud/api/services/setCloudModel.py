from flask import jsonify, session, current_app as app
import cv2, logging, os, pickle, h5py,requests, sys, config, time
import numpy as np, json
#import lfucache.lfu_cache as lfu_cache
from sklearn import linear_model
from scipy.misc import derivative
#from expLatencySize import cache


def uploadImg(url, img, imgName):
	#Send a image to a edge device
	try:
		print("4.1")
		files = {'file': (imgName, open(img, 'rb'), 'image/x-png')}
		r = requests.post(url, files=files)
		print("4.2")		
		if (r.status_code != 201 and r.status_code != 200):
			raise Exception('Received an unsuccessful status code of %s' % r.status_code)

	except Exception as err:
		print("error.")
		print(err.args)
		sys.exit()

	else:
		print("upload com sucesso")


def writeResultCloud(fileJson, uploadTime):
	
	resultsPath = "./results/cachier/resultCloud_%s_%s.txt"%(fileJson["bw"], fileJson["latency"])
	result = fileJson["timeMeasured"] + uploadTime

	with open(resultsPath, "a") as f:
		f.write(str(result) + "\n")



def imageRequest(uploadTime, fileJson):
	# look up if a requested image is in reference dataset, the send it to edge device.
	try:
		referenceDataset  = os.path.join(config.DIR_NAME,"appCloud","api", "services", "ReferenceDataset")
		url = 'http://localhost:5030/api/setimagecache'

		if(fileJson["imgName"] in os.listdir(referenceDataset)):
			img =  os.path.join(referenceDataset, fileJson["imgName"])
			uploadImg(url, img, fileJson["imgName"])
			writeResultCloud(fileJson, uploadTime)

		else:
			raise Exception('Image not found')

		return {'status':'ok','msg':'Dados cadastrados com sucesso.'}
		
	except Exception as e:
		return {'status':'error','msg':'Não foi possível cadastrar os dados.'}