import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model
import numpy as np


facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)
font=cv2.FONT_HERSHEY_COMPLEX


model = load_model('my_model_og4.h5')


def get_className(classNo):
	return str(classNo)

while True:
	sucess, imgOrignal=cap.read()
	faces = facedetect.detectMultiScale(imgOrignal,1.3,5)
	for x,y,w,h in faces:
		crop_img=imgOrignal[y:y+h,x:x+h]
		img=cv2.resize(crop_img, (200,200))
		img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img=img.reshape(1, 200, 200, 1)
		prediction=model.predict(img)
		probabilityValue=np.amax(prediction)
		print("Prediction:", prediction)
		prediction = prediction[0].argmax()
		print("Prediction2:", prediction)
		if (prediction == 0):
			prediction = "1-2 yrs"
		elif (prediction == 1):
			prediction = "3-9 yrs"
		elif (prediction == 2):
			prediction = "10-20 yrs"
		elif (prediction == 3):
			prediction = "21-27 yrs"
		elif (prediction == 4):
			prediction = "28-45 yrs"
		elif (prediction == 5):
			prediction = "46-65 yrs"
		elif (prediction == 6):
			prediction = "65+ yrs"


		if 0==0:
			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
			cv2.putText(imgOrignal, str(get_className(prediction)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)

		cv2.putText(imgOrignal,str(round(probabilityValue*100, 2))+"%" ,(180, 75), font, 0.75, (255,0,0),2, cv2.LINE_AA)
	cv2.imshow("Result",imgOrignal)
	k=cv2.waitKey(1)
	if k==ord('q'):
		break


cap.release()
cv2.destroyAllWindows()








