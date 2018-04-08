"""
Submitted by: Preethi Sekar
Subject: Recognize the face
Program:Recognizes the face

"""

import numpy
import cv2

dictionary={1:'Preethi'}


def converttogray(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#Face Recognition
def RecognizeFace(video1):
    facedetector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eyedetector = cv2.CascadeClassifier('haarcascade_eye.xml')
    rec=cv2.createLBPHFaceRecognizer();
    rec.load("LBPH.xml")
    while(True):
        if(cv2.waitKey(1) & 0xFF == ord('a')):
            break
        ret,image = video1.read()
        #convert it to gray scale for better identification
        grayscaleimage = converttogray(image)
        faces = facedetector.detectMultiScale(grayscaleimage,3,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            eyes = eyedetector.detectMultiScale(grayscaleimage[y:y+h,x:x+w])
            for(eyx,eyy,eyw,eyh) in eyes:
                cv2.rectangle(image[y:y+h,x:x+w],(eyx,eyy),(eyx+eyw,eyy+eyh),(0,255,0),2)
            Id, conf=rec.predict(grayscaleimage[y:y+h,x:x+w])
            for key in dictionary:
                if(Id == key):
                    Id=dictionary[key]    
            cv2.cv.PutText(cv2.cv.fromarray(image),str(Id),(x,y+h),cv2.cv.InitFont(cv2.cv.CV_FONT_ITALIC,5,1,0,4),255)
        cv2.imshow('Recognize_Face',image)
    video1.release()
    cv2.destroyAllWindows()

#program execution begins here
print('Recognizing...')
#recognize face
print('press button a to quit the program')
video = cv2.VideoCapture(0)
RecognizeFace(video)

