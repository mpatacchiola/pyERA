#!/usr/bin/python

## Massimiliano Patacchiola, Plymouth University 2016
#
# In this demo a NAO robot is used to learn the association between
# a teacher head pose and its own head pose. The setup requires a
# table with some objects. The naomarks must be stick on the objects. 
#
#
# Requirements: you need the pynaoqi (>=2.1.4) for python 2.7 (32 or 64 bit)
# You can download it from the aldebaran website. The folder containing
# pynaoqi must be copied in the ex_nao_head_imitation folder.
#

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys  
import time

#The naoqi libraries
#Requires pynaoqi >= 2.1.4
#Change this path according to your pynaoqi folder location
sys.path.insert(1, "./pynaoqi-python2.7-2.1.4.13-linux64")
from naoqi import ALProxy

#
from haar_cascade import haarCascade

#It requires the pyERA library
from pyERA.som import Som
from pyERA.utils import ExponentialDecay




def main():

    # The state machine has the following states:
    # -1, 0, 1, 2, 3, 5
    #
    #

    NAO_IP = "192.168.0.100"
    NAO_PORT = 9559
    STATE = -1


    while(True):

        #Empty STATE, to be used for test
        if(STATE == -1):
            
            STATE = 0

        # The zero state is an init phase
        # In this state all the proxies are
        # created and they subscribe to the services
        #
        elif(STATE==0):
            #Getting the nao proxies
            print("")
            print("STATE: " + str(STATE))
            print("Operation: ALProxy init")
            print("")
            _al_motion_proxy = ALProxy("ALMotion", NAO_IP, int(NAO_PORT))
            _al_tts_proxy = ALProxy("ALTextToSpeech", NAO_IP, int(NAO_PORT))
            _al_landmark_proxy = ALProxy("ALLandMarkDetection", NAO_IP, int(NAO_PORT))
            _al_face_proxy = ALProxy("ALFaceDetection", NAO_IP, int(NAO_PORT))
            _al_memory_proxy = ALProxy("ALMemory", NAO_IP, int(NAO_PORT))
            _al_video_proxy = ALProxy("ALVideoDevice", NAO_IP, int(NAO_PORT))
            #_al_speechrecognition_proxy = ALProxy("ALSpeechRecognition", NAO_IP, int(NAO_PORT))

            # Subscribe to the proxies services
            print("")
            print("STATE: " + str(STATE))
            print("Operation: ALProxy Landmark init")
            print("")
            period = 500 #in msec
            #_al_face_proxy.subscribe("Test_Face", period, 0.0 )
            _al_face_proxy.subscribe("Test_Face")
            #_al_landmark_proxy.subscribe("Test_LandMark", period, 0.0 )
            _al_landmark_proxy.subscribe("Test_LandMark")
            try:
                #"Test_Video", CameraIndex=1, Resolution=1, ColorSpace=0, Fps=5
                #CameraIndex= 0(Top), 1(Bottom)
                #Resolution= 0(160*120), 1(320*240), VGA=2(640*480), 3(1280*960)
                #ColorSpace= AL::kYuvColorSpace (index=0, channels=1), AL::kYUV422ColorSpace (index=9,channels=3),
                #AL::kRGBColorSpace RGB (index=11, channels=3), AL::kBGRColorSpace BGR (to use in OpenCV) (index=13, channels=3)
                #Fps= OV7670 VGA camera can only run at 30, 15, 10 and 5fps. The MT9M114 HD camera run from 1 to 30fps.
                camera_name_id = _al_video_proxy.subscribeCamera("Test_Video", 0, 1, 13, 15)
            except BaseException, err:
                print("[ERROR] connectToCamera: catching error " + str(err))
                return
            #Adding to the speech recognition proxy a vocabulary
            #_al_speechrecognition_proxy.setLanguage("English")
            #vocabulary = ["good", "bad", "nao"]
            #_al_speechrecognition_proxy.setVocabulary(vocabulary, False)
            #_al_speechrecognition_proxy.setVocabulary(vocabulary, False) #If you want to enable word spotting
            #_al_speechrecognition_proxy.subscribe("Test_ASR")

            #Wake up the robot
            _al_motion_proxy.wakeUp()

            #Reset the head position
            _al_motion_proxy.setAngles("HeadPitch", 0.0, 0.3)
            _al_motion_proxy.setAngles("HeadYaw", 0.0, 0.3)

            #Set up the head pose estimator
            my_cascade = haarCascade("./haarcascade_frontalface_alt.xml", "./haarcascade_profileface.xml")

            #Swithc to STATE > 1
            print("")
            print("STATE: " + str(STATE))
            print("Operation: Changing to next state")
            print("")
            time.sleep(2)
            STATE = 1

        elif(STATE==1):
            #time.sleep(0.5)

            # Get data from landmark detection (assuming face detection has been activated).
            #[ [ TimeStampField ] [ Mark_info_0 , Mark_info_1, ... , Mark_info_N-1 ] ]
            #Mark_info = [ ShapeInfo, ExtraInfo ]
            #ExtraInfo = [ MarkID ]
            #Mark ID is the number written on the naomark and which corresponds to its pattern
            #ShapeInfo = [ 0, alpha, beta, sizeX, sizeY, heading]
            #alpha and beta represent the Naomark location in terms of camera angles
            #sizeX and sizeY are the mark size in camera angles
            #heading describes how the Naomark is oriented about the vertical axis with regards to NAO head.
            naomark_vector = _al_memory_proxy.getData("LandmarkDetected")
            if(len(naomark_vector) > 0):
                print("[NAO] Landmark detected!")
                mark_info_vector = naomark_vector[1]
                for mark_info in mark_info_vector:
                    shape_info = mark_info[0]
                    extra_info = mark_info[1]
                    print("Mark: " + str(extra_info[0]))
                    print("Alpha: " + str(shape_info[1]))
                    print("Beta: " + str(shape_info[2]))
                    print("SizeX: " + str(shape_info[3]))
                    print("SizeY: " + str(shape_info[4]))
                    print("Heading: " + str(shape_info[5]))
                    print("")
            #else:
                 #print("No naomark detected...")

            #Get Data for Face Detection
            face_data = _al_memory_proxy.getData("FaceDetected", 0)
            if(len(face_data) > 0):
                print("[NAO] Face detected!")
                face_vector = face_data[1]
                face_info = face_vector[0]

                alpha_face = face_info[0][1]
                beta_face = face_info[0][2]
                width_face = face_info[0][3]
                height_face = face_info[0][4]
                print("Alpha: " + str(alpha_face))
                print("Beta: " + str(beta_face))
                print("Width: " + str(width_face))
                print("Height: " + str(height_face))
                print("")

                #face_counter = 1
                #for face_info in face_vector:
                    #face_shape_info = face_info[0]
                    #alpha = face_shape_info[1]
                    #beta = face_shape_info[2]
                    #width = face_shape_info[3]
                    #height = face_shape_info[4]
                    #print("Face: " + str(face_counter))
                    #print("Alpha: " + str(alpha))
                    #print("Beta: " + str(beta))
                    #print("Width: " + str(width))
                    #print("Height: " + str(height))
                    #print("")
                    #face_counter += 1
            #else:
                 #print("No face detected...")

            STATE = 2


        # In this state it is captured a stream of images from
        # the NAO camera and it is convertend in a Numpy matrix
        # The Numpy matrix cam be analysed as an image from OpenCV
        # in other states.
        elif(STATE==2):

            #Get Images from camera
            naoqi_img = _al_video_proxy.getImageRemote(camera_name_id)
            if(naoqi_img != None):
                img = (
                       np.reshape(
                          np.frombuffer(naoqi_img[6], dtype='%iuint8' % naoqi_img[2]),
                          (naoqi_img[1], naoqi_img[0], naoqi_img[2])
                                  )
                       )
            else:
               img = np.zeros((240, 320))

            #Switch to state 3
            STATE = 3

        # In this state the frames from previous stage
        # are analysed
        #
        elif(STATE==3):
            #Looking for faces with cascade
            #The classifier moves over the ROI
            #starting from a minimum dimension and augmentig
            #slightly based on the scale factor parameter.
            #The scale factor for the frontal face is 1.10 (10%)
            #Scale factor: 1.15=15%,1.25=25% ...ecc
            #Higher scale factors means faster classification
            #but lower accuracy.
            #
            #Return code: 1=Frontal, 2=FrontRotLeft, 
            # 3=FrontRotRight, 4=ProfileLeft, 5=ProfileRight.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            my_cascade.findFace(gray, runFrontal=True, runFrontalRotated=True, runLeft=True, runRight=True, frontalScaleFactor=1.2, rotatedFrontalScaleFactor=1.2, leftScaleFactor=1.1, rightScaleFactor=1.1, minSizeX=40, minSizeY=40, rotationAngleCCW=30, rotationAngleCW=-30, lastFaceType=my_cascade.face_type)

            if(my_cascade.face_type > 0 and my_cascade.face_type < 4):
                face_x1 = my_cascade.face_x 
                face_y1 = my_cascade.face_y 
                face_x2 = my_cascade.face_x + my_cascade.face_w 
                face_y2 = my_cascade.face_y + my_cascade.face_h 
                face_w = my_cascade.face_w 
                face_h = my_cascade.face_h
                print("FACE: ", face_x1, face_y1, face_x2, face_y2, face_w, face_h)
                cv2.rectangle(img, 
                             (face_x1, face_y1), 
                             (face_x2, face_y2), 
                             (0, 255, 0),
                              2)

            #Go back to the GRAB state
            STATE = 4

        # Show the image on a window
        #
        elif(STATE==4):
            cv2.imshow('image',img)
            #When pressing Q on the keyboard swith to EXIT state
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                cv2.destroyAllWindows()
                STATE = 5
            else:
                #Go to the analysis state
                STATE = 1

        # Exit State, unsubscribe from all the proxies
        # and exit.
        #
        elif(STATE==5):
            print("")
            print("STATE: " + str(STATE))
            print("Operation: Unsubscribing ALProxy and Exit")
            print("")
            _al_video_proxy.unsubscribe(camera_name_id)
            _al_landmark_proxy.unsubscribe("Test_LandMark")
            _al_face_proxy.unsubscribe("Test_Face")
            #_al_speechrecognition_proxy.unsubscribe("Test_ASR")
            return #Exit



    #Hello world
    #_al_tts_proxy.say("Oh, it's time to work!")

    #HeadPitch takes: angle (radians), speed (m/sec)
    #_al_motion_proxy.setAngles("HeadPitch", 0.0, 0.3)
    #_al_motion_proxy.setAngles("HeadYaw", 0, HEAD_SPEED) #reset the yaw

    #print _al_motion_proxy.getSummary()
    #print("Hello")
    #_al_motion_proxy.wakeUp()
    #_al_motion_proxy.rest()
    #_posture_proxy.goToPosture("Stand", 1)

if __name__ == "__main__":
    main()
