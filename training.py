"""
Submitted by: Preethi Sekar
Subject: Recognize the face
Program:Creates a dataset and trains it

"""

#libraries required
import numpy
import time
import cv2
import os
from PIL import Image

facedetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyedetector = cv2.CascadeClassifier('haarcascade_eye.xml')
recognizer=cv2.createLBPHFaceRecognizer();
faces=[]
Ids=[]
numberOfFrames = 35
waitTime = 100
path='dataset'


def collectVideoInputData():
    videoSrc =  cv2.VideoCapture(0) 
    TrainFace(videoSrc)

def converttogray(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

def retrieveImages(path):
    paths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in paths:
        #convert to grayscale
        faceImage=Image.open(imagePath).convert('L');
        #converting to numpy array
        numpyarray=numpy.array(faceImage,'uint8')
        #retrieve the id
        Id=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(numpyarray)
        Ids.append(Id)
    return faces,numpy.array(Ids)

def TrainFace(src):
    id=raw_input('Enter unique user identification number : ') 
    counterCheck = 0
    while(counterCheck < numberOfFrames):
        ret,image = src.read()
        #convert it to gray scale for better identification
        grayscaleimage = converttogray(image)
        faces = facedetector.detectMultiScale(grayscaleimage,3,5)
        for(x,y,w,h) in faces:
            counterCheck = counterCheck+1
            #store training data with id extension
            cv2.imwrite("dataset/User."+id+"."+str(counterCheck)+".jpg",grayscaleimage[y:y+h,x:x+w])
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            eyes = eyedetector.detectMultiScale(grayscaleimage[y:y+h,x:x+w])
            for(eyx,eyy,eyw,eyh) in eyes:
                cv2.rectangle(image[y:y+h,x:x+w],(eyx,eyy),(eyx+eyw,eyy+eyh),(0,255,0),2)
            cv2.waitKey(waitTime)
        cv2.imshow('Training_Face',image)
        cv2.waitKey(1)
    src.release()
    cv2.destroyAllWindows()

def TrainData():
    faces,Ids=retrieveImages(path)
    recognizer.train(faces,Ids)
    recognizer.save('LBPH.xml')

#program execution begins here
print('Collecting Input Data...')
#collect data
collectVideoInputData()
print('Training Input Data...')
#train data
t1=time.time()
TrainData()
t2=time.time()
acc=t2-t1
print('Accuracy is '+str(acc)+' secs..')
print('Done...')
