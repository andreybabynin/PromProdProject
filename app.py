from flask import Flask, abort, request
import onnxruntime as rt
import pickle
from fit_model import create_corpus
import pandas as pd
import os
from werkzeug.utils import secure_filename
from util_functions import extract_data, output
import subprocess


with open('models/cv.pickle', 'rb') as f:
    cv = pickle.load(f)

#TODO: параметризовать выбор модели
sess = rt.InferenceSession("models/rfc.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

def get_prediction(text_list):
    try:
        vector = cv.transform(text_list)
        return sess.run([label_name], {input_name: vector.toarray()})[0]
    except: abort(403, 'model failed on data')

def predict(text_json):

    text = extract_data(text_json)
    text_list = []
    if type(text) == str:
        text_list.append(create_corpus(create_corpus(text)))

    else:
        for _, v in text.items():
            text_list.append(create_corpus(v))

    result = get_prediction(text_list)

    return output(result)

app = Flask('Myapp')
app.secret_key = os.urandom(24)

uploads_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)


@app.route('/forward', methods=['POST'])
def forward():
    return predict(request.data)

@app.route('/metadata')
def metadata():
    sess = rt.InferenceSession("models/rfc.onnx")
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

    return predict(text)

@app.route('/retrain')
def retrain():

    out = subprocess.run(['dvc', 'status'], stdout=subprocess.PIPE)
    return out.stdout

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000)

