
# import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import Int64TensorType

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

import pickle
import yaml
import json
import os

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)


df = pd.read_csv("data/spam.csv", encoding = 'latin-1')
df.drop(df.columns[[2,3,4]], axis = 1, inplace = True)
df.columns = ['target','message']

df.target = df.target.map({'ham':0, 'spam':1})

def create_corpus(row):
    review = re.sub('[^a-zA-Z]', ' ', row)
    review = review.lower()
    review = review.split()
    review = [PorterStemmer().stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

df['preprocess'] = df['message'].apply(create_corpus)


cv = CountVectorizer(max_features=5000, min_df=3)

x = cv.fit_transform(df['preprocess'].values).toarray()
with open('models/cv.pickle', 'wb') as f:
    pickle.dump(cv, f)

X_train, X_test, y_train, y_test = train_test_split(x, df.target, train_size=0.7, random_state=103, stratify=df.target)

rfc = RandomForestClassifier(n_estimators=params['feature_storage']['n_estimators'], random_state=103, n_jobs=-1, max_features='log2')

rfc.fit(X_train, y_train)

roc_train = roc_auc_score(y_train, rfc.predict(X_train))
roc_test  = roc_auc_score(y_test, rfc.predict(X_test))

onx_rfc = convert_sklearn(rfc, initial_types=[('int_input', Int64TensorType([None, 2237]))])

with open('models/rfc.onnx', "wb") as f:
    f.write(onx_rfc.SerializeToString())

os.makedirs('metrics', exist_ok = True)

with open('metrics/metrics.json', 'w') as f:
    json.dump({
        'train': {
            'roc_auc': roc_train},
        'test': {
            'roc_auc':roc_test}
        }, f)
