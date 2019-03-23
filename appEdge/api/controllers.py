from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
from .services import edgeModel
import logging, json, os, config, time

api = Blueprint("api", __name__, url_prefix="/api")


@api.route('/lfumodel', methods = ['POST'])
def lfuModel():
	dirname = os.path.dirname(__file__)
	file = request.files.to_dict(flat=False)
	fileImg = file['file'][0]
	fileJson = json.loads(file['info'][0].filename)
	referenceDatasetPath = os.path.join(dirname, fileImg.filename)
	start = time.time()

	if (fileImg):
		fileImg.save(referenceDatasetPath) #save the received image

	result = edgeModel.edgeLFU(fileImg, fileJson, referenceDatasetPath, start)
	if result['status'] == 'ok':
		print(result['msg'])
		return jsonify(result), 200
	else:
		print(result['msg'])
		return jsonify(result), 200


@api.route('/cachiermodel', methods = ['POST'])
def cachierModel():

	fileImg = request.files['file']
	info = request.files['info']
	fileJson = json.loads(info.filename)
	print("chegou alguma image do cliente")
	dirname = os.path.dirname(__file__)
	referenceDatasetPath = os.path.join(dirname, "services", fileImg.filename)
	start = time.time()

	if (fileImg):
		fileImg.save(referenceDatasetPath) #save the received image

	result = edgeModel.edgeCachier(fileImg, fileJson, referenceDatasetPath, start)
	
	if result['status'] == 'ok':
		print(result['msg'])
		return jsonify(result), 200
	else:
		print(result['msg'])
		return jsonify(result), 200

@api.route('/imgfound', methods = ['POST'])
def txtFromCloud():
	nameImg = request.json['img']
	result = edgeModel.checkCache(nameImg)

	if result['status'] == 'ok':
		print(result['msg'])
		return jsonify(result), 200
	else:
		print(result['msg'])
		return jsonify(result), 200