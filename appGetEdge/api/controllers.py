from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
from .services import getEdgeModel
import logging, json, os, config

api = Blueprint("api", __name__, url_prefix="/api")


@api.route('/setimagecache', methods = ['POST'])
def recimage():

	file = request.files['file']
	result = getEdgeModel.saveImage(file)
	if result['status'] == 'ok':
		print(result['msg'])
		return jsonify(result), 200
	else:
		print(result['msg'])
		return jsonify(result), 200