from matplotlib import test
from re import L
import numpy as np
import matplotlib.pyplot as plt
import os
from cv2 import *
from PIL import Image
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten , Conv2D , MaxPooling2D
import pickle
Directory = "PATH"

Categories_Directory = ['DOG','CAT',]

Testing_Categories = ['Dog','Cat']


IMG_SIZE = 70

training_data = []
testing_data =[]
prediction_data =[]
new_test_data = []
def create_training_data():
    for category in Categories_Directory:
        path = os.path.join(Directory,category)
        class_num = Categories_Directory.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except:
                pass

create_training_data()

def create_testing_data():
    for test_category in Testing_Categories:
        path = os.path.join(Directory,test_category)
        class_num = Testing_Categories.index(test_category)
        for img in os.listdir(path):
                test_img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                test_new_array = cv2.resize(test_img_array,(IMG_SIZE,IMG_SIZE)) 
                testing_data.append([test_new_array,class_num])

create_testing_data()

def create_testing_data():
    for test_category in New_Test:
        path = os.path.join(Directory,test_category)
        class_num = New_Test.index(test_category)
        for img in os.listdir(path):
                test_img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                test_new_array = cv2.resize(test_img_array,(IMG_SIZE,IMG_SIZE)) 
                new_test_data.append([test_new_array,class_num])

create_testing_data()


def prediction():
    for predict in Prediction:
      path = os.path.join(Directory,predict)
      class_num2 = Prediction.index(predict)
      for img in os.listdir(path):
        prediction_img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        prediction_new_array = cv2.resize(prediction_img_array,(IMG_SIZE,IMG_SIZE))
        prediction_data.append([prediction_new_array,class_num2])

prediction()



random.shuffle(training_data)

random.shuffle(prediction_data)

x = []
y = []

test_x = []
test_y = []


new_x =[]
new_y = []


for features,label in training_data:
    x.append(features)
    y.append(label)


for features_test , label_test in testing_data:
    test_x.append(features_test)
    test_y.append(label_test)

for feature , brandname in testing_data:
  new_x.append(feature)
  new_y.append(brandname)

X = np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,1)

pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")

X = pickle.load(pickle_in)

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("Y.pickle","rb"))

X = X/255.0

model = Sequential()
model.add(Conv2D(64,(3),input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =( 2)))

model.add(Conv2D(64,(3),input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size =( 2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = "binary_crossentropy",optimizer='adam',metrics=["accuracy"])

X = np.array(X)
y = np.array(y)

model.fit(X,y,batch_size = 36,epochs = 20) 


test_x = np.array(test_x)
test_y = np.array(test_y)


new_x = np.array(new_x)
new_y = np.array(new_y)

model.evaluate(test_x,test_y)

