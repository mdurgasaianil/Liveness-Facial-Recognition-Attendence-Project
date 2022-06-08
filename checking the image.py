import cv2
import numpy as np
import face_recognition

import os
path = 'images_attendence'
images = [] # for to store all imported images
classnames = []
mylist = os.listdir(path) # all the images from path stored in mylist variable
#print(mylist)
for cls in mylist:
    curimg = cv2.imread(f'{path}/{cls}')# reading the image and storing in images list
    images.append(curimg)
    classnames.append(os.path.splitext(cls)[0]) # split the name of the image.jpg to image and image name will be stored in class names list
#print(classnames)

encodelist = []
for img in images:
    # for encoding the image we need to convert bgr to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # finding encodings in the images
    encode = face_recognition.face_encodings(img)[0]
    encodelist.append(encode)
print(len(encodelist),len(images))


