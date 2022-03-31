import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import onnxruntime as rt
import os
import pickle
import json
from util_functions import args

sess = rt.InferenceSession(f"{args.model}/rfc.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

with open(f'{args.input}/X_train.pickle', "rb") as f:
    X_train = pickle.load(f)

with open(f'{args.input}/X_test.pickle', "rb") as f:
    X_test = pickle.load(f)

with open(f'{args.input}/y_train.pickle', "rb") as f:
    y_train = pickle.load(f)

with open(f'{args.input}/y_test.pickle', "rb") as f:
    y_test = pickle.load(f)


roc_train = roc_auc_score(y_train, sess.run([label_name], {input_name: X_train})[0])
roc_test  = roc_auc_score(y_test, sess.run([label_name], {input_name: X_test})[0])

os.makedirs(f'{args.metrics}', exist_ok = True)

with open(f'{args.metrics}/metrics.json', 'w') as f:
    json.dump({
        'train': {
            'roc_auc': roc_train},
        'test': {
            'roc_auc':roc_test}
        }, f)

os.makedirs(f'{args.metrics}/plots', exist_ok = True)

fpr, tpr, _ = roc_curve(y_test, sess.run([label_name], {input_name: X_test})[0])
#create plots
pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(f'{args.metrics}/plots/roc_auc.csv', index = False)