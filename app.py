from flask import Flask
from flask import request
import onnxruntime as rt
import pickle
from fit_model import create_corpus

with open('models/cv.pickle', 'rb') as f:
    cv = pickle.load(f)

def convert_to_vector(text):
    text = text.decode('utf-8')
    text = create_corpus(text)
    vector = cv.transform([text])
    return vector


def predict(text):

    vector = convert_to_vector(text)
    sess = rt.InferenceSession("models/rfc.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    result = sess.run([label_name], {input_name: vector.toarray()})[0][0]

    return {'Type': int(result)}

app = Flask('Myapp')

@app.route('/page')
def get_page():
    return 'ok'

@app.route('/forward', methods=['POST'])
def get_not_page():
    return predict(request.data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

