from flask import Flask, abort, request
import pandas as pd
import os
from werkzeug.utils import secure_filename
from util_functions import *
import subprocess
import yaml
import json

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

model_folder = params['workspace']['folders']['model']

app = Flask('Myapp')
app.secret_key = os.urandom(24)

uploads_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

@app.route('/forward', methods=['POST'])
def forward():
    return predict(request.data, model_folder)

@app.route('/metadata')
def metadata():
    sess = return_model(model_folder)
    return sess._model_meta.custom_metadata_map

@app.route('/forward_batch', methods=['POST'])
def forward_batch():

    flask_file = request.files['file']
    if not flask_file:
        return 'Upload a CSV file'

    flask_file.save(os.path.join(uploads_dir, secure_filename(flask_file.filename)))

    return 'Data uploaded'
    
@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate():

    flask_file = request.files['file']
    if not flask_file:
        return 'Upload a CSV file'

    text = pd.read_csv(flask_file).to_json()

    return predict(text, model_folder)

@app.route('/retrain')
def retrain():
    #TODO: закончить с моделью
    # folder_name = 'instance/uploads'
    # _ = subprocess.run(['dvc', 'exp', 'run', '-C', 'workspace.folders.input=', f'{folder_name}'], stdout=subprocess.PIPE)''
    exp_json = subprocess.run(['dvc', 'exp', 'show', '--json'], stdout=subprocess.PIPE)
    json_dic = json.loads(exp_json)
    return json_dic.keys()

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000)

