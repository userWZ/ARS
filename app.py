import os
from os.path import join, dirname, realpath
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename  # web底层框架
from restore_v2 import *
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
        file = 'IEEE30_2.xlsx'
        Black_Start = BlackStartGrid(file)

        data, path = Black_Start.restore()

        # image = request.files['upload-image']
        # print("image", image.filename)
        # image.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image.filename)))
        # filename = os.path.join(app.config['UPLOAD_FOLDER']) + image.filename
        # print("file", filename)
        return render_template('data.html', data=data, data1=path)
    else:
        file = 'IEEE30.xlsx'
        Black_Start = BlackStartGrid(file)
        data, path = Black_Start.restore()
        return render_template('data.html', data=data, data1=path)

@app.route('/view', methods=['Get', 'Post'])
def grid_view():
    return render_template('grid_view.html')

@app.route('/view2')
def grid_view_baidu():
    return render_template('grid_view_baidu.html')

@app.route('/view3')
def grid_view_line():
    return render_template('grid_view_line.html')

if __name__ == '__main__':
    app.run(debug=True)