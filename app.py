from crypt import methods
from sys import stdout
from flask import Flask, abort, request
import pandas as pd
import os
from werkzeug.utils import secure_filename
from util_functions import *
import subprocess
import json

params = open_yaml('params.yaml')
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
    metadata_dic = sess._model_meta.custom_metadata_map
    metadata_dic['Hash'] = get_hash()
    return metadata_dic

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
    
    EXP_ID = get_id()
    _ = subprocess.run(['dvc', 'exp', 'run', '-n', f'{EXP_ID}'], check=True)

    return f'New model trained, experiment id {EXP_ID}\n'


@app.route('/metrics/<int:exp_id>', methods=['GET'])
def metrics(exp_id):
    exp_json = subprocess.run(['dvc', 'exp', 'show', '--json'], stdout=subprocess.PIPE).stdout
    last_commit = subprocess.run(['git', 'log', '-n', '1', '--format="%H"'], stdout=subprocess.PIPE).stdout

    json_dic = json.loads(exp_json)
    exp_dic = json_dic[str(last_commit)[3:-4]]

    return get_metrics(exp_dic, exp_id)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000)

