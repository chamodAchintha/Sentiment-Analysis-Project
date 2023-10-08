import numpy as np
import pandas as pd
import re
import string
import pickle

from nltk.stem import PorterStemmer
ps = PorterStemmer()

with open('./static/model/supportVector.pickle', 'rb') as f:
    model = pickle.load(f)

with open('./static/model/corpora/stopwords/english', 'r') as file:
    stopwords = file.read().splitlines()

vocab = pd.read_csv('./static/model/vocabulary.txt', header=None)
tokens = vocab[0].tolist()

def remove_punctuations(text):
    for punct in string.punctuation:
        text = text.replace(punct,'')
    return text

def preprocess(tweet):
    data = pd.DataFrame([tweet],columns=["tweet"])
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*','',x, flags=re.MULTILINE) for x in x.split()))
    data['tweet'] = data['tweet'].apply(remove_punctuations)
    data['tweet'] = data['tweet'].str.replace('\d+', '', regex=True)
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
    return data["tweet"]


def vectorization(data):
    vectorized_list=[]
    for sentance in data:
        sentence_list = np.zeros(len(tokens))
        for i in range(len(tokens)):
            if tokens[i] in sentance.split():
                sentence_list[i] = 1
        vectorized_list.append(sentence_list)
    vectorized_list = np.asarray(vectorized_list, dtype=np.float32)
    return vectorized_list

def prediction(vectorized_txt):
    pred = model.predict(vectorized_txt)
    if pred == 0:
        return "Positive"
    return "Negative"


