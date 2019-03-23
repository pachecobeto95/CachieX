from flask import Flask, render_template, session
from appCloud.api.controllers import api


app = Flask(__name__, static_folder="static")
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config.from_object("config")
app.config['JSON_AS_ASCII'] = False


app.register_blueprint(api)