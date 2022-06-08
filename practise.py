import cv2
import numpy as np
#######################
#######################
cap = cv2.VideoCapture(0)
cap.set(3,640) # Width
cap.set(4,480) # Height
cap.set(10,100)
while True:
    success,img = cap.read()
    cv2.imshow('webcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()