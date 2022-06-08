import cv2
import numpy as np
import face_recognition
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import os
from tensorflow.keras.models import model_from_json
# Load Anti-Spoofing Model graph
json_file = open('Anti_spoofing_model/Face_Antispoofing_System/antispoofing_models/antispoofing_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load antispoofing model weights
model.load_weights('Anti_spoofing_model/Face_Antispoofing_System/antispoofing_models/antispoofing_model.h5')
print("Model loaded from disk")
cap = cv2.VideoCapture(0)
cap.set(10,50)
while True:
    success,img = cap.read()
    # we need to reduce the real time image for to increase the speed
    imgs  = cv2.resize(img,(0,0),None,0.25,0.25) # it is resizing the 1/4th of orginal image
    # converting current image bgr to rgb
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    # finding the face locations in current image
    facescurrframe = face_recognition.face_locations(imgs)
    # encoding the faces in current frame
    #encodinscurrframe = face_recognition.face_encodings(imgs,facescurrframe)
    for faceloc in facescurrframe:
        y1, x2, y2, x1 = faceloc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # this is because we reduce the size by 1/4th so for to create a bounding box aroound the face we need to multiply with 4
        resized_face = cv2.resize(img, (160, 160))
        resized_face = resized_face.astype("float") / 255.0
        resized_face = img_to_array(resized_face)
        resized_face = np.expand_dims(resized_face, axis=0)
        preds = model.predict(resized_face)[0]
        if preds > 0.5:
            label = 'spoof'
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, label, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            #cv2.rectangle(img, (x, y), (x+w,y+h), (0, 0, 255), cv2.FILLED)
            #cv2.putText(img, label, (y1 + 6, x2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        else:
            label = 'real'
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, label, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            # cv2.rectangle(img, (x, y), (x+w,y+h), (0, 0, 255), cv2.FILLED)
            #cv2.putText(img, label, (y1 + 6, x2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('webcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break