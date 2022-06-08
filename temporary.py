def attendence():
    import pickle
    import cv2
    import numpy as np
    import face_recognition

# code for automatically convert the image to bgr to rgb from particular file
    import os

# creating a function for attendence sheet
# recording name and time
    from datetime import datetime
#def markattendence(name):
 #   with open('Attendence.csv','r+') as f:
  #      mydatalist = f.readlines()
   #     namelist = [] # for to store the list of values in the csv file. for to avoid storing the repeat value
    #    for line in mydatalist:
     #       entry = line.split(',')
      #      namelist.append(entry[0])
       # if name not in namelist:
        #    now = datetime.now()
         #   dtstring = now.strftime('%H:%M:%S')
          #  f.writelines(f'\n{name},{dtstring}')

# loading the encodelistknown_faces
    encodelistknown_faces = pickle.load(open("encodelistknown_faces","rb"))
    classnames = pickle.load(open("classnames","rb"))
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
                with open('Attendence.csv', 'r+') as f:
                    mydatalist = f.readlines()
                    namelist = []  # for to store the list of values in the csv file. for to avoid storing the repeat value
                    for line in mydatalist:
                        entry = line.split(',')
                        namelist.append(entry[0])
                    if name not in namelist:
                        now = datetime.now()
                        dtstring = now.strftime('%H:%M:%S')
                        f.writelines(f'\n{name},{dtstring}')
            #markattendence(name)

        cv2.imshow('webcam',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
