import json
from flask import abort
import argparse

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
                    help='input folder for learned data')

parser.add_argument('--metrics',
                    type=str,
                    default='metrics',
                    help='metrics folder')


args = parser.parse_args()





def extract_data(text_json):
    #decode json
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