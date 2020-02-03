from tensorflow.keras.models import load_model
import numpy as np
import cv2
from skimage.transform import resize

categories=['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___healthy','Peach___Bacterial_spot','Peach___healthy']
 
model = load_model('model.h5')

def getPrediction():
    img = cv2.imread('test.jpg')
    resImage = resize(img,(28,28))
    prob = model.predict(np.array([resImage],))
    sortProb = np.argsort(prob[0,:])
    return categories[sortProb[-1]]

print(getPrediction())
