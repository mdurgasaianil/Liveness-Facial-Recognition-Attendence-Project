import cv2
import numpy as np
import face_recognition

# Step:1 - loading the images and convert them into rgb
img = face_recognition.load_image_file('images/Elon_Musk.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('images/billgates_test.jpg')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

# Step:2 - Finding the faces & encodings in the images
faceloc = face_recognition.face_locations(img)[0]
encodeelon = face_recognition.face_encodings(img)[0]
# detecting the faces in an image
cv2.rectangle(img,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloctest = face_recognition.face_locations(imgtest)[0]
encodeelontest = face_recognition.face_encodings(imgtest)[0]
# detecting the faces in an image
cv2.rectangle(imgtest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

# print(faceloc) printing the face locations of x and y values
# step 4: comparing both images distances
# for back end there will be linear svm for comparing the both 128 image value distances
results = face_recognition.compare_faces([encodeelon],encodeelontest)
# for to find similarity between the images
facedis = face_recognition.face_distance([encodeelon],encodeelontest)
# for lesser distance between the images it consider as same
# for both elon mask images distance is 0.35358715
# for both elon mask & bill gates images distance is 0.78692069
print(results,facedis)
cv2.putText(imgtest,f'{results} {round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.imshow('Elon_musk',img)
cv2.imshow('Elon_musk_test',imgtest)
cv2.waitKey(0)