from skimage.transform import resize
from skimage import io
from skimage import img_as_ubyte


import numpy as np
import matplotlib.pyplot as plt
import os


# save numpy array as csv file
from numpy import asarray
from numpy import savetxt
from numpy import save
from numpy import load

datadir1="Datasets/Testing_Dataset"
datadir="Datasets/Training_Dataset"
training_data=[]
testing_data=[]

categories=['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___healthy','Peach___Bacterial_spot','Peach___healthy']

def create_training_data():
        for category in categories:
            path=os.path.join(datadir,category)
            class_num=categories.index(category)
            for img in os.listdir(path):
                # img_array=cv2.imread(os.path.join(path,img))
                # img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                # new_array=cv2.resize(img_array,(28,28))

                img_array = io.imread(os.path.join(path,img))
                img_array = img_array[:, :, ::-1]
                new_array = resize(img_array,(28,28))
                new_array = img_as_ubyte(new_array)
              
                training_data.append([new_array,class_num])

def create_testing_data():
    for category in categories:
            path=os.path.join(datadir1,category)
            class_num=categories.index(category)
            for img in os.listdir(path):
                #opencv code
                # img_array=cv2.imread(os.path.join(path,img))
                # img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                # new_array=cv2.resize(img_array,(28,28))

                img_array = io.imread(os.path.join(path,img))
                img_array = img_array[:, :, ::-1]
                new_array = resize(img_array,(28,28))
                new_array = img_as_ubyte(new_array)
                    
                testing_data.append([new_array,class_num])


# print(training_data[0][0][0])

#save data to npz

# create_training_data()
# create_testing_data()

# from numpy import savez_compressed
# train_data = asarray([training_data])
# savez_compressed('train_data.npz', train_data)

# test_data = asarray([testing_data])
# savez_compressed('test_data.npz', test_data)



#load data from npz
# dict_data = load('data.npz',allow_pickle=True)
# data = dict_data['arr_0']
# print(data[0])
#s