'''
Project: Cachier (Cloud) 

Author: Roberto Pacheco
Date: February, 05, 2019
Modified: Roberto Pacheco
Date: February, 05, 2019

Description:
These functions are executed by the Cloud on Cachier's project
'''


from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
from .services import missManager
import logging, os, time, json
import numpy as np

api = Blueprint("api", __name__, url_prefix="/api")



#look up in Reference dataset a similar image
@api.route('/cachemissCloud', methods = ['POST'])
def cachemissCloud():

	fileImg = request.files["file"]
	info = request.files["info"]
	fileJson = json.loads(info.filename)
	end = time.time()

	dirname = os.path.dirname(__file__)
	referenceDatasetPath = os.path.join(dirname, "services", "ReferenceDataset", fileImg.filename)
	if (fileImg):
		fileImg.save(referenceDatasetPath) #save the received image

	uploadTime = end - fileJson["time"]

	result = missManager.cloudModelCloud(fileImg, uploadTime, fileJson)

	if result['status'] == 'ok':
		print(result['msg'])
		return jsonify(result), 200
	else:
		print(result['msg'])
		return jsonify(result), 200


#look up in Reference dataset a similar image
@api.route('/cachemissLFU', methods = ['POST'])
def cacheMissLFU():
	
	info = request.files["info"]
	fileJson = json.loads(info.filename)
	uploadTime = time.time() - fileJson["time"]
	fileImg = request.files["file"]
	dirname = os.path.dirname(__file__)
	referenceDatasetPath = os.path.join(dirname, "services", "ReferenceDataset", fileImg.filename)
	if (fileImg):
		fileImg.save(referenceDatasetPath) #save the received image

	result = missManager.cloudModelLFU(fileImg, fileJson, uploadTime)

	if result['status'] == 'ok':
		print(result['msg'])
		return jsonify(result), 200
	else:
		print(result['msg'])
		return jsonify(result), 200

@api.route('/cachemisscachier', methods = ['POST'])
def cachemissCachier():
	fileImg = request.files["file"]
	info = request.files["info"]
	fileJson = json.loads(info.filename)
	uploadTime = time.time() - fileJson["time"]

	dirname = os.path.dirname(__file__)
	referenceDatasetPath = os.path.join(dirname, "services", "ReferenceDataset", fileImg.filename)
	if (fileImg):
		fileImg.save(referenceDatasetPath) #save the received image

	result = missManager.cloudModelCachier(fileImg, fileJson, uploadTime)

	if result['status'] == 'ok':
		print(result['msg'])
		return jsonify(result), 200
	else:
		print(result['msg'])
		return jsonify(result), 200


@api.route('/test', methods = ['POST'])
def test():

	testJson = request.json
	return jsonify({"status":"ok"})