
UPLOAD_FOLDER = 'uploads/'
from tensorflow.keras.models import load_model
import matplotlib.pylab as plt
import numpy as np
# import cv2
from skimage.transform import resize
from keras.backend import clear_session

categories=['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___healthy','Peach___Bacterial_spot','Peach___healthy']
 



def getPrediction():
    img = plt.imread('uploads/test.jpg')
    resImage = resize(img,(28,28))
    model = load_model('model.h5')
    model._make_predict_function()
    prob = model.predict(np.array([resImage],))
    sortProb = np.argsort(prob[0,:])
    return categories[sortProb[-1]]
    

# print(getPrediction())


# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

import os
import urllib.request
from flask import Flask, request, redirect, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__,static_url_path="/static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     print(request.files)
#     # checking if the file is present or not.
#     if 'file' not in request.files:
#         return "No file found"

#     file = request.files['file']
#     file.save("static/test.jpg")
#     return "file successfully saved"
@app.route('/home')
def upload_form():
	return render_template('upload.html')


@app.route('/', methods=['GET'])
def home():
    return app.send_static_file('index.html')




@app.route('/api/file-upload', methods=['GET', 'POST'])
def upload_file():
	# check if the post request has the file part
    if 'file' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message' : 'No file selected for uploading'})
        resp.status_code = 400
        return resp
        
    if file and allowed_file(file.filename):
        filename = "test.jpg"
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        return render_template('answer.html', answer=getPrediction())
        
        # resp = jsonify({'message' :getPrediction()})
        # resp = jsonify({'message' : getPrediction()})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify({'message' : 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
        resp.status_code = 400
        return resp

if __name__ == "__main__":
    app.run()