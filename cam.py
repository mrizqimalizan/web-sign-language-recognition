import numpy as np
import cv2
import math
import matplotlib.pylab as plt
import pandas as pd
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import keras.backend as K


new_model = load_model('newmodel77.h5')


class vid():
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        #self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        #self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    def __del__(self):
        self.cam.release()

    def getFrame(self):
    	work, img = self.cam.read()
    	img = cv2.flip(img,+1)
    	cv2.rectangle(img,(100,100),(300,300),(0,255,0),0)
    	self.crop_image = img[100:300, 100:300]
    	self.gray = cv2.cvtColor(self.crop_image,cv2.COLOR_BGR2GRAY)
    	self.blur = cv2.GaussianBlur(self.gray, (5, 5), 0)
    	self.test = self.blur
    	self.eval = self.evaluate(self.test)
    	self.evalPredict = self.eval
    	print(self.evalPredict)
    	ret,jpeg = cv2.imencode('.jpeg',self.test)
    	return jpeg.tobytes()

    def evaluate(self,img):
        img = cv2.resize(img,(28,28))
        img2 = np.reshape(img,[1,28,28,1])
        self.predict = self.prediction(img2)
        self.predicted_class = np.argmax(predictions[0])
        self.predict = self.predictions[0][self.predicted_class]
        return self.predict

    def prediction(self,test):
    	self.testo = test
    	self.predict = new_model.predict(self.testo)
    	K.clear_session()
    	return self.predict

'''
while(True):
	check, frame = video.read()
	print(check)
	print(frame)
	num=num + 1
	#make mirror image
	mframe = cv2.flip(frame,+1)
	# Get hand data from the rectangle sub window
	cv2.rectangle(mframe,(100,100),(300,300),(0,255,0),0)
	crop_image = mframe[100:300, 100:300]

	# Create a blank 300x300 black image
	blk = np.zeros((200, 200, 3), np.uint8)
	blk2 = np.zeros((200, 500, 3), np.uint8)

	gray = cv2.cvtColor(crop_image,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	#_,tresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
	#g=cv2.cvtColor(tresh,cv2.COLOR_BGR2GRAY)

	#create the test data
	test = blur
	test = cv2.resize(test,(28,28))
	test = np.reshape(test,[1,28,28,1])
	test = test / 255
	#print(train.shape)
	

	# doing predictions
	predictions = new_model.predict(test)
	#test_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))

	predicted_class = np.argmax(predictions[0])
	prob = predictions[0][predicted_class]
	prob = prob*100
	prob = round(prob,2)
	out=("recognized: {} with the probability of: {} %".format(chr(predicted_class+65), prob))
	cv2.putText(blk2, out, (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)


	cv2.imshow('mframe',mframe)
	cv2.imshow('cropped image',crop_image)
	cv2.imshow('gray',gray)
	cv2.imshow('blur',blur)
	#cv2.imshow('threshold image',tresh)
	cv2.imshow('blck2',blk2)


	#for playing the vid
	k=cv2.waitKey(30) & 0xff
	if k == 27:
		break

#shutdown the camera
cap.release()

cv2.destroyAllWindows()
'''