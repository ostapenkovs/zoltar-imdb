from flask import Flask, request, render_template
import numpy as np
from tensorflow import keras
import re

app = Flask(__name__)

app.config['DEBUG'] = False

word_index = keras.datasets.imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNKN>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return " ".join( [reverse_word_index.get(i,"?") for i in text] )

def encode_review(text):
    cleaned = [re.sub("[^a-zA-Z]","",token) for token in text.split() if (re.sub("[^a-zA-Z]","",token) != "")]
    encoded = [1]

    for word in cleaned:
        if((word.lower() in word_index) and (word_index[word.lower()]<50000)):
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['review']
        if not text:
            return render_template('index.html', message='Please enter required fields')

        model = keras.models.load_model("model.h5")
    
        encoded = encode_review(text)
        encoded = keras.preprocessing.sequence.pad_sequences([encoded],value=word_index["<PAD>"],padding="post",maxlen=500)
        test_pred = model.predict(np.array( encoded, ))

        if(test_pred[0]>.5):
            percent = int(test_pred[0]*100)
            message = "Zoltar is {}% confident that you liked the movie.".format(percent)
        else:
            percent = int((1-test_pred[0])*100)
            message = "Zoltar is {}% confident that you did not like the movie.".format(percent)

        return render_template('predict.html', message=message)

if __name__ == '__main__':
    app.run()