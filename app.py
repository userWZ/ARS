import os
from os.path import join, dirname, realpath
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename  # web底层框架
from restore import *
import time
import json

# 初始化flask

app = Flask(__name__)
data_table = []
Bootstrap(app)
app.config['UPLOAD_FOLDER'] = os.path.dirname(__file__)

@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        file = request.files['upload-file']
        data,path = restore(file)
        # image = request.files['upload-image']
        # print("image", image.filename)
        # image.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image.filename)))
        # filename = os.path.join(app.config['UPLOAD_FOLDER']) + image.filename
        # print("file", filename)
        return render_template ('data.html', data=data, data1=path)


if __name__ == '__main__':
    app.run(debug=True)