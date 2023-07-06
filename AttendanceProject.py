import cv2
import numpy as np
import face_recognition
import os  # for getting path
from datetime import datetime

# creating a list so that it can detect and generates encodings automatically rather than creating 1 by 1
path = 'ImagesAttendance'
# creating list of all images which we import
images = []
# wrting the names of all the images
classNames = []
# grabbing the images from the folder
myList = os.listdir(path)
print(myList)

# using the names and importing the images we can use load image and also we can use imread fun
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}') # reading the path and name of our image
    images.append(curImg) # appending images into the list
    classNames.append(os.path.splitext(cl)[0]) # appending the class name just by the name(ie not jpg) and first element
print(classNames)

# finding encodings of each image from the imported ones with the same process

def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

# creating a new function to mark attendance in excel file with name and time
def markAttendance(name):

    # opening the file and read nd write at same time
    with open('Attendance.csv','r+') as f:
        # reading the lines ofn data as we dont want the name of the same person to keep on repeating
        myDataList = f.readlines()
        #print(myDataList)
#markAttendance('a')
        # putting all the names we find into the list
        nameList = []
        for line in myDataList:
            # finding entry and spliting
            entry = line.split(',')
            nameList.append(entry[0]) #name

            #checking if current name is present in the list or not
        if name not in nameList:
            now = datetime.now() #gives the date and time
            time = now.strftime('%H:%M:%S')
            date = now.strftime('%Y-%m-%d')
            f.writelines(f'\n{name},{time},{date}')




# running thr function
encodeListKnown = findEncodings(images)
#print(len(encodeListKnown))
print('Encoding Complete')

# initializing the webcam

cap = cv2.VideoCapture(0)

# to get each frame we use while loo[
while True:
    success, img = cap.read() # gives us the image
    # as we are doing in real time we need to reduce the size for speeding the process

    # only defining the scale not the pixel, 0.25 is 1/4 of img
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    # converting to rgb
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # finding the encoding of webcam
    # as we may find multiple faces so we are going to find loc of our faces and send this to encoding func
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    # finding matches and iterating through all the faces in cur frame
    # and comparing all the faces with encodings found before
    # zip is used as we need both encodeFace,faceLoc in one loop
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        # doing the matching with thw list of known faces with the face encoded
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        # finding the distance with same parameters
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        # to get the best match ie least distance we can use this below function and it gives out an index
        matchIndex = np.argmin(faceDis)
        #print(matchIndex)

        # once the person is known from index we can  display a bounding box around them write thier name


        if faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
            markAttendance(name)
        else:
            name = 'Unknown'
        # print(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)



    cv2.imshow('webcam',img)
    cv2.waitKey(1)


