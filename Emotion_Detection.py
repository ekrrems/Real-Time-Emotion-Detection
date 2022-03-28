import pandas as pd
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt

pickle_in = open('emotion_detection_model.p','rb')
model = pickle.load(pickle_in)

def pred(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(48,48))
    pred = model.predict(img[np.newaxis,:,:,np.newaxis])
    return str(np.argmax(pred.astype('int')))

# Making Predictions on Real-Time Video
    
cap = cv2.VideoCapture(0)

while 1:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    print(pred(frame))
    cv2.putText(frame,pred(frame),(50,50),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0))
    cv2.imshow('frame',frame)

    if cv2.waitKey(1)& 0xFF==ord('q'):
        break