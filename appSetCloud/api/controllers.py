from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
from .services import setCloudModel
import logging, json, os, config, time

api = Blueprint("api", __name__, url_prefix="/api")


#Cachier cache when requesta image
@api.route('/requestimage', methods = ['POST'])
def requestImage():

	fileJson = request.json
	uploadTime = time.time() - request.json["time"]
	print("A imagem solicitada: %s"%fileJson["imgName"])
	result = setCloudModel.imageRequest(uploadTime, fileJson)

	if result['status'] == 'ok':
		print(result['msg'])
		return jsonify(result), 200
	else:
		print(result['msg'])
		return jsonify(result), 200