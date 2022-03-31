from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import Int64TensorType

from util_functions import *

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import pickle
import yaml
import hashlib

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

try:
    df = pd.read_csv(f'{args.input}/spam.csv', encoding = 'utf-8')
except:
    df = pd.read_csv(f'{args.input}/spam.csv', encoding = 'latin-1')
# TODO: придумать как обрабатывать колонки
df.drop(df.columns[[2,3,4]], axis = 1, inplace = True)
df.columns = ['target','message']
df.target = df.target.map({'ham':0, 'spam':1})

df['preprocess'] = df['message'].apply(create_corpus)

cv = CountVectorizer(max_features=5000, min_df=3)
x = cv.fit_transform(df['preprocess'].values).toarray()
X_train, X_test, y_train, y_test = train_test_split(x, df.target, train_size=0.7, random_state=103, stratify=df.target)

rfc = RandomForestClassifier(n_estimators=params['feature_storage']['n_estimators'], random_state=103, n_jobs=-1, max_features='log2')
rfc.fit(X_train, y_train)
onx_rfc = convert_sklearn(rfc, initial_types=[('int_input', Int64TensorType([None, 2237]))])

#add metadata
meta = onx_rfc.metadata_props.add()
meta.key = "Date"
meta.value = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
meta = onx_rfc.metadata_props.add()
meta.key = "Name"
meta.value = params['feature_storage']['experiment_name']
meta = onx_rfc.metadata_props.add()
meta.key = 'Hash'
meta.value = hashlib.md5(datetime.now().strftime("%Y-%m-%d %H-%M-%S").encode('utf-8')).hexdigest()

with open(f'{args.model}/rfc.onnx', "wb") as f:
    f.write(onx_rfc.SerializeToString())

with open(f'{args.model}/cv.pickle', 'wb') as f:
    pickle.dump(cv, f)

with open(f'{args.output}/X_train.pickle', "wb") as f:
    pickle.dump(X_train, f)

with open(f'{args.output}/X_test.pickle', "wb") as f:
    pickle.dump(X_test, f)

with open(f'{args.output}/y_train.pickle', "wb") as f:
    pickle.dump(y_train, f)

with open(f'{args.output}/y_test.pickle', "wb") as f:
    pickle.dump(y_test, f)



