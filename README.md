# Face-Recognition
ReadMe:

1.training.py is used to create and train the dataset.
2.recognizer.py identifies the face in the video.
3.haarcascade_frontalface_default.xml is utilized by the training.py and detector.py to detect the face.
4.haarcascade_eye.xml is utilized by the training.py and detector.py to detect the eyes in the face.
5.LBPH.xml is the file generated once the dataset is trained to recognize the face.
6.dataset consists of the set of frames that are used for detection.It is created once the training.py is used to create the dataset.
7.User identification Number (required for training.py)corresponds to the Name which should be mapped in the dictionary(recognizer.py).

Python version 2.7.14
Libraries required:
numpy
openCV
os
time
pillow


