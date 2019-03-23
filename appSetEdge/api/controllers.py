from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
from .services import setEdgeModel
import logging, json, os, config, time

api = Blueprint("api", __name__, url_prefix="/api")


@api.route('/setcache', methods = ['POST'])
def setcache():
	print("chegou no sett")
	file = request.files['file']
	info = request.files["info"]
	fileJson = json.loads(info.filename)
	
	result = setEdgeModel.setCacheLFU(file, fileJson)

	if result['status'] == 'ok':
		print(result['msg'])
		return jsonify(result), 200
	else:
		print(result['msg'])
		return jsonify(result), 200


@api.route('/checkcache', methods = ['POST'])
def txtFromCloud():
	fileJson = request.json
	print(fileJson) 
	nameImg = fileJson['imgName']
	uploadTime = time.time() - fileJson['time']
	timeMeasured =  fileJson["timeMeasured"]
	result = setEdgeModel.checkCache(nameImg, uploadTime, timeMeasured)

	if result['status'] == 'ok':
		print(result['msg'])
		return jsonify(result), 200
	else:
		print(result['msg'])
		return jsonify(result), 200

