import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.enable_eager_execution()

import tensorflow.keras
from tensorflow.python.keras.models import Sequential,Input,Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU

# from skimage import io

from sklearn.model_selection import train_test_split

from skimage.transform import resize
from skimage import io
from skimage import img_as_ubyte

from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import load
# import cv2

def train_model():

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



    # create_training_data()
    # create_testing_data()

    #load already present values
    test_data_dict = load('test_data.npz',allow_pickle=True)
    test_data = test_data_dict['arr_0']
    testing_data = test_data[0]

    train_data_dict = load('train_data.npz',allow_pickle=True)
    train_data = train_data_dict['arr_0']
    training_data = train_data[0]

    ##################Loading done

    train_X=[]
    train_Y=[]

    for feature,label in training_data:
        train_X.append(feature)
        train_Y.append(label)
    type(train_X)

    test_X=[]
    test_Y=[]
    for feature,label in testing_data:
        test_X.append(feature)
        test_Y.append(label)



    train_X=np.array(train_X)#.reshape(-1,28,28,1)
    train_Y=np.array(train_Y)
    test_X=np.array(test_X)#.reshape(-1,28,28,1)
    test_Y=np.array(test_Y)
    #type(train_X)

    # get_ipython().run_line_magic('matplotlib', 'inline')

    # print('Training data shape : ', train_X.shape, train_Y.shape)

    # print('Testing data shape : ', test_X.shape, test_Y.shape)




    classes = np.unique(train_Y)
    nClasses = len(classes)
    # print('Total number of outputs : ', nClasses)
    # print('Output classes : ', classes)



    train_X = train_X.reshape(-1, 28,28, 3)
    test_X = test_X.reshape(-1, 28,28, 3)
    train_X.shape, test_X.shape

    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    train_X = train_X / 255.
    test_X = test_X / 255.




    # Change the labels from categorical to one-hot encoding
    train_Y_one_hot = to_categorical(train_Y)
    test_Y_one_hot = to_categorical(test_Y)





    train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
    train_X.shape,valid_X.shape,train_label.shape,valid_label.shape


    batch_size = 32
    epochs = 10
    num_classes = 5



    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(Dense(num_classes, activation='softmax'))



    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy, optimizer=tensorflow.keras.optimizers.Adam(),metrics=['accuracy'])
    # model.summary()

    model_train = model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))


    test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=0)
    # print('Test loss:', test_eval[0])
    # print('Test accuracy:', test_eval[1])
    test_loss = float(test_eval[0])
    test_accuracy = float(test_eval[1])



    train_loss = model_train.history['loss']
    val_loss = model_train.history['val_loss']

    train_acc = model_train.history['acc']
    val_acc = model_train.history['val_acc']




    return_value = dict()
    return_value['train_loss'] = [float(i) for i in train_loss] 
    return_value['val_loss'] = [float(i) for i in val_loss]
    return_value['train_acc'] = [float(i) for i in train_acc]
    return_value['val_acc'] = [float(i) for i in val_acc]
    return_value['test_loss'] = test_loss
    return_value['test_accuracy'] = test_accuracy

    print(return_value)


    return return_value


# train_model()