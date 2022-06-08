import cv2
import numpy as np
import face_recognition

# code for automatically convert the image to bgr to rgb from particular file
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

# encoding the all images
def findencodings(images):
    encodelist = []
    for img in images:
        # for encoding the image we need to convert bgr to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # finding encodings in the images
        encode = face_recognition.face_encodings(img)[0] # pretrained model for to give 128 measurements of a image
        encodelist.append(encode)
    return encodelist
# creating a function for attendence sheet
# recording name and time
from datetime import datetime
def markattendence(name):
    with open('Attendence.csv','r+') as f:
        mydatalist = f.readlines()
        namelist = [] # for to store the list of values in the csv file. for to avoid storing the repeat value
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')
            #print(namelist,name,mydatalist)

encodelistknown_faces = findencodings(images)
print('Encoding Complete')
import pickle
pickle.dump(encodelistknown_faces,open("encodelistknown_faces","wb"))
pickle.dump(classnames,open("classnames","wb"))
