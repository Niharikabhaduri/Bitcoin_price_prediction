from flask import *
from flask_cors import CORS

def create_app():
    app = Flask(__name__,template_folder='template')
    CORS(app,supports_credentials=True)
    return app