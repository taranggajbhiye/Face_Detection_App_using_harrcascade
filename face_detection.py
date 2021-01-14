import cv2

#train raw data(xml format images of front face shapes) using harr-cascade-classifier 
trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#initialize webcam or video file path
webcam = cv2.VideoCapture(0)

while True:
    #read each frame of video
    _,img = webcam.read()

    #preprocessing of image(converting color image into black and white)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #get coordnates of detected faces in list(multiple faces in single frame)
    faces_coordinates = trained_data.detectMultiScale(gray_img)

    #superimpose rectangle in live video frame
    for(x,y,w,h) in faces_coordinates: 
        cv2.rectangle(img, (x,y), (x+w, x+h), (0,255,0),2)

    cv2.imshow('Face Detection Application', img)

    key = cv2.waitKey(30)
    if(key==27): #to exit from loop press escape key
        break

webcam.release()

#test for single image
# test = cv2.imread("myphoto1.jpg")
# grey_img = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
# coordinates = data.detectMultiScale(grey_img, 1.1, 4)
# print(coordinates)