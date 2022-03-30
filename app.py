from flask import Flask, abort
from flask import request
import onnxruntime as rt
import pickle
from fit_model import create_corpus
import json

CLASSIFIER = {0: 'NOT SPAM', 
              1: 'SPAM'}

with open('models/cv.pickle', 'rb') as f:
    cv = pickle.load(f)

def extract_data(text_json):
    #decode json
    try:
        text = json.loads(text_json)['text']
    except:
        abort(400, 'bad request')
    return text

def predict(text_json):

    text = extract_data(text_json)
    text = create_corpus(text)
    try:
        vector = cv.transform([text])
        sess = rt.InferenceSession("models/rfc.onnx")
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        result = sess.run([label_name], {input_name: vector.toarray()})[0][0]
    except:
        abort(403, 'model failed on data')
    return {'Type': CLASSIFIER[int(result)], 'Result': int(result)}

app = Flask('Myapp')

# @app.route('/page')
# def get_page():
#     return 'ok'

@app.route('/forward', methods=['POST'])
def get_not_page():
    return predict(request.data)

@app.route('/metadata')
def metadata():
    sess = rt.InferenceSession("models/rfc.onnx")
    return sess._model_meta.custom_metadata_map

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000)

