import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from keras.preprocessing import image
os.chdir(os.path.dirname(__file__))
model=keras.models.load_model("model_cnn_pawan.h5")
list_res=["Mask","Without Mask"]
vid = cv2.VideoCapture(0)
while(True):
    ret, frame = vid.read()
    sec=frame
    g=frame
    g=cv2.cvtColor(g,cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(g, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(g, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faces = frame[y:y + h, x:x + w]
#         cv2.imshow("face",faces)
        test_image=cv2.cvtColor(faces,cv2.COLOR_BGR2RGB)
        test_image=cv2.resize(test_image, (128,128))
        test_image=image.img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)
        result=model.predict(test_image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        val=list_res[int(result[0][0])]
        te= cv2.putText(sec,val, org, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('frame', te)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
