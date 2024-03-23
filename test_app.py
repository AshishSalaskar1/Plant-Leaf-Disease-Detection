
UPLOAD_FOLDER = 'uploads/'
from tensorflow.keras.models import load_model
import matplotlib.pylab as plt
import numpy as np
# import cv2
from skimage.transform import resize
from flask_cors import CORS, cross_origin
# import model_train
# from keras.backend import clear_session

categories=['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___healthy','Peach___Bacterial_spot','Peach___healthy']
 

# def train_model():
#     return_metrics = model_train.train_model()
#     return return_metrics

def getPrediction():
    img = plt.imread('uploads/test.jpg')
    resImage = resize(img,(28,28,3))
    model = load_model('model.h5')
    model.make_predict_function()
    prob = model.predict(np.array([resImage],))
    sortProb = np.argsort(prob[0,:])
    return categories[sortProb[-1]]
    
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
import os
import urllib.request
from flask import Flask, request, redirect, jsonify, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__,static_url_path="/static")

# cors = CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
cors = CORS(app, resources={r"/train": {"origins": "https://cnn-flask.herokuapp.com/home"}})

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/home')
@cross_origin()
def upload_form():
	return render_template('upload.html')

# @app.route('/train')
# @cross_origin()
# def train_form():
#     return_metrics = train_model()
#     return jsonify(return_metrics)
    # response = jsonify(return_metrics)
    # response.headers.add('Access-Control-Allow-Origin', '*')
    # return response



@app.route('/test')
@cross_origin()
def test_fn():
    data = {"test_accuracy":0.9090909361839294,"test_loss":0.2362778918309645,"train_acc":[0.3465346395969391,0.6707921028137207,0.7301980257034302,0.8589109182357788,0.8861386179924011,0.9084158539772034,0.9504950642585754,0.9678217768669128,0.9603960514068604,0.9678217768669128],"train_loss":[1.5189724712088557,0.9187239137026343,0.582697061618956,0.37744680401122216,0.3049584026678954,0.22594869977766924,0.14196028095660823,0.10174716522197912,0.12360440991302528,0.09216664952806908],"val_acc":[0.5643564462661743,0.5841584205627441,0.7326732873916626,0.8910890817642212,0.8415841460227966,0.8811880946159363,0.9504950642585754,0.9306930899620056,0.9207921028137207,0.9405940771102905],"val_loss":[1.2257271745417377,0.8644408707571501,0.6019150957022563,0.32721499582328417,0.43055959739307365,0.2777669866486351,0.14636361097345257,0.16552021361813687,0.19681064600106513,0.17120785996465399]}
    return jsonify(data)


@app.route('/getAllDiseases')
@cross_origin()
def getALlDiseases():
    return jsonify({"No_of_Disease":len(categories),"Diseases":categories})

@app.route('/', methods=['GET'])
@cross_origin()
def home():
    return app.send_static_file('index.html')

#Dual get post
@app.route('/api/file-upload', methods=['GET', 'POST'])
@cross_origin()
def upload_file():
	# check if the post request has the file part
    if 'file' not in request.files:
        #return render_template('./CCR/index.html')
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

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     print(request.files)
#     # checking if the file is present or not.
#     if 'file' not in request.files:
#         return "No file found"

#     file = request.files['file']
#     file.save("static/test.jpg")
#     return "file successfully saved"
