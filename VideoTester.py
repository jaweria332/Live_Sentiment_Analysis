#Importing necessary libraries
import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

#Loading the model
model=model_from_json(open("fer.json").read())
#Loading the weight
model.load_weights('fer.h5')

#Loading HaarCascade Classifier to detect face
face_haar_cascade=cv2.CascadeClassifier('E:\\TRY_ON_VIRTUAL\\Step_1_face_recognize\\haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)
while True:
    #Capturing the frames
    ret,test_img=cap.read()
    #If no face found
    if not ret:
        continue
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    #Getting detected face (if any)
    faces_detected=face_haar_cascade.detectMultiScale(gray_img,1.32,2)

    #Getting dimensions of detected face
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),2)
        #Croping our region of interest ie, face
        roi_gray=gray_img[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48))
        #Converting image to array
        img_pixels=image.img_to_array(roi_gray)
        img_pixels=np.expand_dims(img_pixels,axis=0)
        img_pixels /=255

        #Drawing prediction
        predictions =model.predict(img_pixels)
        max_index=np.argmax(predictions[0])

        #Defining emotions on the basis of our dataset
        emotions=('angry','disgust','fear','happy','sad','surprise','neutral')
        predicted_emotions=emotions[max_index]

        cv2.putText(test_img,predicted_emotions,(int(x),int(y)),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

    resize_img=cv2.resize(test_img,(400,200))
    cv2.imshow('Facial emotion analysis',resize_img)

    #Wait until 'q' is pressed
    if cv2.waitKey(10)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()