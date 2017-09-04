from flask import Flask,render_template,request
from scipy.misc import imread,imresize,imsave
import numpy as np
from keras import models
import re
import os
import sys
import base64
from model.load import *
app = Flask(__name__)
global model, graph
model, graph = init()

dir = os.path.dirname(__file__)
png = os.path.join(dir,'output.png')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
    convertoImage(request.get_data())
    x = imread(png,mode='L')
    x = np.invert(x)
    x = imresize(x,(28,28))

    x = x.reshape(1,28,28,1)
    with graph.as_default():
        out = model.predict(x)
        response = np.array_str(np.argmax(out,axis=1))
        return response



def convertoImage(imgData):
    imgStr = re.search(b'base64,(.*)',imgData).group(1)
    with open(png,'wb') as output:
        output.write(base64.decodebytes(imgStr))


if __name__ == '__main__':
    app.run()
