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

# now we need to match the encodings with unknown image
# the unknown image will come from webcam

cap = cv2.VideoCapture(1)
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
    encodinscurrframe = face_recognition.face_encodings(imgs,facescurrframe)

    for encodeface,faceloc in zip(encodinscurrframe,facescurrframe): # for to take both encoding and locations in same loop we are using zip
        matches = face_recognition.compare_faces(encodelistknown_faces,encodeface)
        faceDis = face_recognition.face_distance(encodelistknown_faces,encodeface)
        # taking out the lowest distance which tells us the best match for the given image
        matchindex = np.argmin(faceDis)
        # now for match index we need to draw the bounding box and the name to it
        if matches[matchindex]:
            # printing the name of the current image if the faces are matches
            name = classnames[matchindex].upper()
            #print(name)
            # drawing a bounding box to matched face
            y1,x2,y2,x1 = faceloc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4 # this is because we reduce the size by 1/4th so for to create a bounding box aroound the face we need to multiply with 4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markattendence(name)

    cv2.imshow('webcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
