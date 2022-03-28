import numpy as np
import seaborn as sns
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pickle

data = pd.read_csv(r'C:\Users\ekrem\OneDrive\Masaüstü\Machine learning DATAS\Facial Expression Recognition\icml_face_data.csv')
data.head()

#df = pd.DataFrame(columns=['pixel'+str(i) for i in range(len(data[' pixels'][0].split(' ')))],index=[a for a in range(len(data[' pixels']))])

#b = []
#for a in range(len(data[' pixels'])):
#    for i  in data[' pixels'][a].split(' '):
#        b.append(int(i))
#    df.loc[a] =list(b) 
#    b=[]
#df.to_scv(r'C:\Users\ekrem\OneDrive\Masaüstü\Machine learning DATAS\Facial Expression Recognition\facial_detection.csv')
df = pd.read_csv(r'C:\Users\ekrem\OneDrive\Masaüstü\Machine learning DATAS\Facial Expression Recognition\facial_detection.csv')
df.head()

df.drop('Unnamed: 0',axis=1,inplace=True)

data_final = pd.concat([data,df],axis=1)
data_final.head() 

# Splitting dataset into Train, test and evaluation subsets

train_data = data_final[data_final[' Usage']== 'Training']
test_data = data_final[(data_final[' Usage']=='PrivateTest')]
eval_data = data_final[data_final[' Usage']=='PublicTest']

test_data = test_data.drop(columns=' Usage')
train_data = train_data.drop(' Usage',axis=1)
eval_data = eval_data.drop(' Usage',axis=1)

test_data = test_data.drop(columns=' pixels')
train_data = train_data.drop(' pixels',axis=1)
eval_data = eval_data.drop(' pixels',axis=1)

# Creating X and Y sets 

#Test Data
X_test = test_data.drop('emotion',axis=1)
y_test = test_data['emotion']

#Train Data
X_train = train_data.drop('emotion',axis=1)
y_train = train_data['emotion']

#Evaluation Data
X_eval = eval_data.drop('emotion',axis=1)
y_eval = eval_data['emotion']

#Reshaping the pixels (X_train and X_test)
X_train = np.array(X_train).reshape((28709,48,48,1))
X_test  = np.array(X_test).reshape((3589,48,48,1))
X_eval = np.array(X_eval).reshape((3589,48,48,1))
plt.imshow(X_train[0].squeeze(axis=2))

#(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
numofClass=len(y_train.unique())
numofClass

# CNN Model
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Dense,Flatten
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()

# 1 - Convolution
model.add(Conv2D(32,(3,3), padding='same', input_shape=(48, 48,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd Convolution layer
model.add(Conv2D(64,(5,5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(numofClass))
model.add(Activation('softmax'))


model.compile(loss='sparse_categorical_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy']) #Targets are not One-hot encoded therefore we use sparse_categorical_crossentropy


batch_size =32

train_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range = 0.1,
                                  zoom_range=0.3)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow(X_train,y_train,
                                     batch_size=batch_size)
test_generator = test_datagen.flow(X_test,y_test)


hist=model.fit_generator(train_generator,
                   steps_per_epoch=28709//batch_size,
                   epochs=50,
                   validation_data=test_generator,
                   validation_steps=3589//batch_size)



print(hist.history.keys())
plt.plot(hist.history['loss'],label='Train Loss')
plt.plot(hist.history['val_loss'],label='Validation Loss')
plt.legend()
plt.figure()
plt.plot(hist.history['acc'],label='Train Accuracy')
plt.plot(hist.history['val_acc'],label='Validation Accuracy')
plt.legend()
plt.show()


pickle_out = open("emotion_detection_model.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()



# show the confusion matrix of our predictions
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow(X_eval,y_eval)













