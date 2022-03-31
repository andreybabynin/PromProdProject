import json
from flask import abort
import argparse
import onnxruntime as rt
import pickle
import re
# import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
# nltk.download('stopwords')
import yaml

CLASSIFIER = {0: 'NOT SPAM', 
              1: 'SPAM'}

parser = argparse.ArgumentParser()
#specify parameters
parser.add_argument('-o', '--output',
                    type=str,
                    default='data',
                    help='output folder for artifcats')

parser.add_argument('-m', '--model',
                    type=str,
                    default='models',
                    help='models folder for learned models')

parser.add_argument('-i', '--input',
                    type=str,
                    default='data',
                    help='input folder for data')

parser.add_argument('--metrics',
                    type=str,
                    default='metrics',
                    help='metrics folder')

parser.add_argument('-n', '--name',
                    type=str,
                    default='default',
                    help='name of a trained model')

args = parser.parse_args()

def create_corpus(row):
    review = re.sub('[^a-zA-Z]', ' ', row)
    review = review.lower()
    review = review.split()
    review = [PorterStemmer().stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

def open_yaml(file_path):
    with open(file_path, 'rb') as f:
        return yaml.safe_load(f)


def return_model(model_folder):

    return rt.InferenceSession(f"{model_folder}/rfc.onnx")

def run_model(vector, model_folder):

    sess = return_model(model_folder)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    return sess.run([label_name], {input_name: vector.toarray()})[0]

def return_cv(model_folder):
    with open(f'{model_folder}/cv.pickle', 'rb') as f:
        cv = pickle.load(f)
    return cv

def get_prediction(text_list, model_folder):
    try:
        cv = return_cv(model_folder)
        vector = cv.transform(text_list)

        return run_model(vector, model_folder)
    except: abort(403, 'model failed on data')

def predict(text_json, model_folder):

    text = extract_data(text_json)
    text_list = []
    if type(text) == str:
        text_list.append(create_corpus(text))

    else:
        for _, v in text.items():
            text_list.append(create_corpus(v))

    result = get_prediction(text_list, model_folder)

    return output(result)


def extract_data(text_json):
    try:
        text = json.loads(text_json)['text']
    except:
        abort(400, 'bad request')
    return text


def output(result):
    dic = {}
    for i in range(len(result)):
        dic[i] = {'Type': CLASSIFIER[int(result[i])], 'Result': int(result[i])}
    return dic

def get_hash():
    loc = open_yaml('dvc.lock')
    return loc['stages']['fit_model']['outs'][5]['md5']


