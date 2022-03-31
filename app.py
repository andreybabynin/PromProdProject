from crypt import methods
from flask import Flask, abort, session, request, jsonify
import onnxruntime as rt
import pickle
from fit_model import create_corpus
import json
import csv
import codecs
import pandas as pd
import os
from werkzeug.utils import secure_filename

CLASSIFIER = {0: 'NOT SPAM', 
              1: 'SPAM'}

with open('models/cv.pickle', 'rb') as f:
    cv = pickle.load(f)

#TODO: параметризовать выбор модели
sess = rt.InferenceSession("models/rfc.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

def extract_data(text_json):
    #decode json
    try:
        text = json.loads(text_json)['text']
    except:
        abort(400, 'bad request')
    return text

def get_prediction(text):
    vector = cv.transform([text])
    return sess.run([label_name], {input_name: vector.toarray()})[0][0]


def predict(text_json):

    text = extract_data(text_json)
    text = create_corpus(text)
    try:
        result = get_prediction(text)
    except:
        abort(403, 'model failed on data')
    return {'Type': CLASSIFIER[int(result)], 'Result': int(result)}

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
    
@app.route('/evaluate', methods=['GET'])
def evaluate():
    # TODO: реализовать функцию загрузки множества объектов и их оценки



if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000)

