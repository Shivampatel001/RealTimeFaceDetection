import numpy as np
import cv2
    
faceCascade = cv2.CascadeClassifier("C:/Users/DEADPOOL/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("C:/Users/DEADPOOL/Desktop/dev_trial/eye.xml") #Directory pf eye.xml
mouthCascade = cv2.CascadeClassifier("C:/Users/DEADPOOL/Desktop/dev_trial/mouth.xml") #Directory of mouth.xml

#VideCapture has an attribute value 0 which means it will take input directly from the camera
cap = cv2.VideoCapture(0)
#Set Width
cap.set(3,640)
#Set Height 
cap.set(4,480) 
bw_threshold = 83
#Coordinates where messages will be displayed
org = (30, 30)
font = cv2.FONT_HERSHEY_COMPLEX
thickness = 2
font_scale = 1
WearedMaskFontColor = (255, 255, 255)
NotWearedMaskFontColor = (0 , 0, 255)
WeardMask = "Mask Detected"
NotWeardMask = "Mask Not Detected"



while True:
    ret, img = cap.read()
    #img = cv2.flip(img, -1)
    #Convert image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Convert image to Black and White
    thresh, black_and_white = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)

    #detectMultiScale will return a rectangle with coordinates(x,y,w,h) around the detected face.
    #face detection
    faces = faceCascade.detectMultiScale(            
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )

    #face detection for black and white
    faces_bw = faceCascade.detectMultiScale(
        black_and_white,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (20, 20)
    )





    #Condition if no face is found
    if(len(faces) == 0 and len(faces_bw) == 0):
        cv2.putText(img , "No face found", org, font, font_scale, WearedMaskFontColor, thickness, cv2.LINE_AA )
    
    #Condition for white mask detection
    elif(len(faces) == 0 and len(faces_bw) == 1):
        cv2.putText(img, WeardMask, org, font , font_scale, WearedMaskFontColor, thickness,cv2.LINE_AA)

    else:
        #Create rectangle around the detected face
        for (x,y,w,h) in faces:
            #Mark a rectangle around the face
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w] 

            eyes = eyeCascade.detectMultiScale(roi_gray)
            #To create a rectangle around the eyes
            #for(a,b,c,d) in eyes:
                #cv2.rectangle(roi_color,(a,b), (a+c, b+d), (0, 255, 0), 2)
    
            #Detect mouth
            mouths = mouthCascade.detectMultiScale(gray, 1.7, 11)

            #Condition if no mouth is found but face is found which means face mask detected
            if(len(mouths) == 0 and (len(eyes)) != 0):
                cv2.putText(img, WeardMask, org, font, font_scale, WearedMaskFontColor, thickness, cv2.LINE_AA)
            else:
                #Checking if lips are present or not
                for(mx, my, mw, mh) in mouths:
                    #Coordinates of lips lie under coordinates of Face 
                    if(y < my < y + h):
                        #Lips detected  which means no mask found
                        cv2.putText(img, NotWeardMask, org, font, font_scale, NotWearedMaskFontColor, thickness, cv2.LINE_AA)
                        break

    #Display the main window
    cv2.imshow('Real_Time_Face_Detection',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()