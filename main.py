import cv2
import numpy as np
import face_recognition

# importing images and loading
imgElon = face_recognition.load_image_file('ImageBasic/Elon Musk.jpg')
# converting bgr to rgb
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)


# importing test images for encoding

imgTest = face_recognition.load_image_file('ImageBasic/Elon Test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

# finding the images and encodings

# detecting faces use face loacation function
faceLoc = face_recognition.face_locations(imgElon)[0] # [0] getting first element of the single image

# encode the face which is detected

encodeElon = face_recognition.face_encodings(imgElon)[0]  # [0] getting first element of the single image

#print(faceLoc) doing this we get 4 diffrent values of top, right bottom and left

# face locations can be detected by using a rectangle function of open cv
# based on face loc we can give x1 y1 and x2 y2
# face loc are defult points of image and color is defined by 255 0 255 ie purple and thickness is 2

cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

# doing the same for test image
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

# comparing and finding distance

# comparing the encodings , in the backend it uses linear svm to find weather it is a match or not

results = face_recognition.compare_faces([encodeElon],encodeTest)
#print(results)

# to find how similar these images the best match we can use the face distance function,lesser the dis better the match

faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)

# displaying the results on the test image of first element(as it is a array and not a element) and round it upto 2
# giving the origin(98,98) and the font and the scale = 1 and any color red and thickness as 2
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(98,98),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

# displaying the images

cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Test',imgTest)
cv2.waitKey(0)



# if matches[matchIndex]:
        #     name = classNames[matchIndex].upper() # converting to upper case letters
        #     #print(name)
        #     # creating the bounding box
        #     # top bottom right left
        #     y1,x2,y2,x1 = faceLoc
        #     # resing the image to the oroginal one
        #     y1, x2, y2, x1= y1*4,x2*4,y2*4,x1*4
        #     # drawing the rectangle with color and thickness
        #     cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        #     # rectangle at bottom and shows the name
        #     cv2.rectangle(img,(x1,y2-30),(x2,y2),(0,255,0),cv2.FILLED)
        #
        #     cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        #     markAttendance(name)