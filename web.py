from flask import Flask,render_template,url_for,request,Response,session,jsonify
from flask_bootstrap import Bootstrap
from cam import vid
from flask_socketio import SocketIO, send,emit
import cv2

from keras.models import load_model
import numpy as np
import keras.backend as K
import tensorflow as tf

import json

new_model = load_model('newmodel9191.h5')
graph=tf.get_default_graph()

app = Flask(__name__)
Bootstrap(app)

#app.config['SECRET_KEY'] ='secret'
socketio = SocketIO(app,message = 'http://127.0.0.1:500',async_mode='threading')

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/hrc')
def hrc():
	return render_template('HRC.html')

def gen():
    i=1
    while i<10:
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+str(i)+b'\r\n')
        i+=1

def get_frame():

    camera_port=0

    ramp_frames=100

    camera=cv2.VideoCapture(camera_port) #this makes a web cam object

    
    i=1
    while True:
        retval, im = camera.read()
        #make mirror image
        mframe = cv2.flip(im,+1)
        # Get hand data from the rectangle sub window
        cv2.rectangle(mframe,(100,100),(300,300),(0,255,0),0)
        crop_image = mframe[100:300, 100:300]
        gray = cv2.cvtColor(crop_image,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        #create the real time data 
        test = blur
        test = cv2.resize(test,(28,28))
        test = np.reshape(test,[1,28,28,1])
        test = test / 255

        global graph
        with graph.as_default():
            # doing predictions
            predictions = new_model.predict(test)
            #test_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(predictions, axis=1))
            predicted_class = np.argmax(predictions[0])
            prob = predictions[0][predicted_class]
            prob = prob*100
            prob = round(prob,2)
            out=("recognized: {} with the probability of: {} %".format(chr(predicted_class+65), prob))
            predict = json.dumps(out)
            print(predict)
            handleMessage(predict)
        imgencode=cv2.imencode('.jpg',mframe)[1]
        stringData=imgencode.tostring()
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
        i+=1

    del(camera)

@app.route('/calc')
def calc():
     return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('message')
def handleMessage(predict):
	with app.app_context():
		socketio.emit('response message',predict,namespace='/predict',broadcast=True)
    
 	

if __name__ == '__main__':
	socketio.run(app, debug=True)

'''
		gray = cv2.cvtColor(crop_image,cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray, (5, 5), 0)
		#_,tresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
		#g=cv2.cvtColor(tresh,cv2.COLOR_BGR2GRAY)

		#create the real time data 
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
		'''
