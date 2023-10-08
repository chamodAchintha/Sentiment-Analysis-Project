from flask import Flask, render_template,request, redirect
from helper import preprocess, vectorization, prediction
from logger import logging

app = Flask(__name__)

logging.info('Flask server started')

data = dict()
reviews = []
positive = 0
negative = 0

@app.route("/")
def index():
    data['reviews'] = reviews
    data['positive'] = positive
    data['negative'] = negative

    logging.info('========== Open home page ============')

    return render_template('index.html', data=data)

@app.route("/", methods = ['post'])
def my_post():
    text = request.form['text']
    logging.info(f'Text : {text}')

    preprocessed_txt = preprocess(text)
    logging.info(f'Preprocessed Text : {preprocessed_txt}')

    vectorized_txt = vectorization(preprocessed_txt)
    logging.info(f'Vectorized Text : {vectorized_txt}')

    pred = prediction(vectorized_txt)
    logging.info(f'Prediction : {pred}')

    if pred == 'Negative':
        global negative
        negative += 1
    else:
        global positive
        positive += 1
    
    reviews.insert(0, text)
    return redirect(request.url)

if __name__ == "__main__":
    app.run()