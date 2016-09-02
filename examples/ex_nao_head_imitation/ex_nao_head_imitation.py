#!/usr/bin/python

## Massimiliano Patacchiola, Plymouth University 2016
#
# In this demo a NAO robot is used to learn the association between
# a teacher head pose and its own head pose. The setup requires a
# table with some objects. Stick the naomarks on the objects.
# Pressing the button on the keyboard you can teach the robot and
# control its actions. 
#
# Requirements: you need the pynaoqi (>=2.1.4) for python 2.7 (32 or 64 bit)
# You can download it from the aldebaran website. The folder containing
# pynaoqi must be copied in the ex_nao_head_imitation folder.
#
# The joints limit in degrees [radians] for the NAO's head are:
# PITCH: +29.5 [0.51] (down), -38.5 [-0.67] (up)
# YAW: -119.5 [-2.08] (right), +119.5 [2.08] (left)

import numpy as np
import cv2
import time

#The naoqi libraries
#Requires pynaoqi >= 2.1.4
#Change this path according to your pynaoqi folder location
import sys
sys.path.insert(1, "./pynaoqi-python2.7-2.1.4.13-linux64")
from naoqi import ALProxy

#from haar_cascade import haarCascade

#It requires the pyERA library
from pyERA.som import Som
from pyERA.utils import ExponentialDecay

def main():
    # The state machine has the following states:
    # [VOID, INIT, FIND, IMG, SHOW, KEY, COUNTER, NAO, WHICH, QUIT]
    #
    # VOID: Empty state, to use for test
    # INIT: It is called only once for the initialisation
    # FIND: use the hardware face and landmark detection libraries
    # IMG: Aquire an image from the camera and convert in in numpy array
    # SHOW: Print the image on screen using OpenCV
    # KEY: Check which key is pressed
    # COUNTER: Check if some counters are alive. It avoids the use of sleeps.
    # NAO: Pressing the (h) button the robot look in front of itself
    # WHICH: Pressing the (w) button is like asking to the robot to look to a object on the table
    # QUIT: Pressing (q) unsubscribe the proxy and close the script

    #Configuration Variables, adjust to taste
    NAO_IP = "192.168.0.100"
    NAO_PORT = 9559
    VOICE_ENBLED = True #If True the robot speaks
    STATE = "VOID" #The initial state
    #These two lists contains the landmarks ID
    #and the associated object name. Stick many naomarks
    #on different objects and be sure the robot can look at 
    #them when turning the head.
    landmark_id_list = [64, 68, 80, 85]
    landmark_name_list = ["sponge", "stapler", "book", "cup"]

    while(True):

        #Empty STATE, to be used for test
        if(STATE == "VOID"):            
            STATE = "INIT"

        # The zero state is an init phase
        # In this state all the proxies are
        # created and tNAO subscribe to the services
        #
        elif(STATE=="INIT"):
            #Init some generic variables
            #This counter allows continuing the program flow
            #without calling a sleep
            which_counter = 0 #Counter increased when WHICH is called
            which_counter_limit = 30 #Limit for the which counter

            #Getting the nao proxies
            print("[STATE " + str(STATE) + "] " + "ALProxy init" + "\n")
            _al_motion_proxy = ALProxy("ALMotion", NAO_IP, int(NAO_PORT))
            _al_posture_proxy = ALProxy("ALRobotPosture", NAO_IP, int(NAO_PORT))
            _al_tts_proxy = ALProxy("ALTextToSpeech", NAO_IP, int(NAO_PORT))
            _al_landmark_proxy = ALProxy("ALLandMarkDetection", NAO_IP, int(NAO_PORT))
            _al_face_proxy = ALProxy("ALFaceDetection", NAO_IP, int(NAO_PORT))
            _al_memory_proxy = ALProxy("ALMemory", NAO_IP, int(NAO_PORT))
            _al_video_proxy = ALProxy("ALVideoDevice", NAO_IP, int(NAO_PORT))
            #_al_speechrecognition_proxy = ALProxy("ALSpeechRecognition", NAO_IP, int(NAO_PORT))

            # Subscribe to the proxies services
            print("[STATE " + str(STATE) + "] " + "ALProxy Landmark init" + "\n")
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
                resolution_type = 1
                cam_w = 320
                cam_h = 240
                camera_name_id = _al_video_proxy.subscribeCamera("Test_Video", 0, resolution_type, 13, 15)
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
            print("[STATE " + str(STATE) + "] " + "Waking up the NAO..." + "\n")
            _al_motion_proxy.wakeUp()
            #_al_motion_proxy.rest()
            time.sleep(2)
            print("[STATE " + str(STATE) + "] " + "Go to Crouch Pose..." + "\n")
            _al_posture_proxy.goToPosture("Crouch", 0.5)
            #_al_posture_proxy.goToPosture("StandZero", 0.5)
            #Reset the head position
            _al_motion_proxy.setAngles("HeadPitch", 0.0, 0.3)
            _al_motion_proxy.setAngles("HeadYaw", 0.0, 0.3)

            #Swithc to STATE > 1
            print("[STATE " + str(STATE) + "] " + "Switching to next state" + "\n")
            time.sleep(2)
            STATE = "FIND"

        # Get data from landmark detection and 
        # face detection.
        #
        elif(STATE=="FIND"):
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
            naomark_list = list()
            if(naomark_vector and isinstance(naomark_vector, list) and len(naomark_vector) > 0):
                print("[NAO] Landmark detected!")
                #is_landmark_detected = True
                mark_info_vector = naomark_vector[1]
                for mark_info in mark_info_vector:
                    shape_info = mark_info[0]
                    extra_info = mark_info[1]
                    #Declaring the mark data
                    id_mark = extra_info[0]
                    alpha_mark = shape_info[1]
                    beta_mark = shape_info[2]
                    size_x_mark = shape_info[3]
                    size_y_mark = shape_info[4]
                    heading_mark = shape_info[5]
                    #Append the info of each single mark to the list
                    naomark_list.append([id_mark, alpha_mark, beta_mark, size_x_mark, size_y_mark, heading_mark])
                    print("Mark: " + str(id_mark))
                    print("Alpha: " + str(alpha_mark))
                    print("Beta: " + str(beta_mark))
                    print("SizeX: " + str(size_x_mark))
                    print("SizeY: " + str(size_y_mark))
                    print("Heading: " + str(heading_mark))
                    print("")
            #else:
                 #print("No naomark detected...")
                 #is_landmark_detected = False

            #Get Data for Face Detection
            face_data = _al_memory_proxy.getData("FaceDetected", 0)
            if(face_data and isinstance(face_data, list) and len(face_data) > 0):
            #if(len(face_data) > 0):
                print("[NAO] Face detected!")
                face_vector = face_data[1]
                face_info = face_vector[0]
                is_face_detected = True
                alpha_face = face_info[0][1]
                beta_face = face_info[0][2]
                width_face = face_info[0][3]
                height_face = face_info[0][4]
                print("Alpha: " + str(alpha_face))
                print("Beta: " + str(beta_face))
                print("Width: " + str(width_face))
                print("Height: " + str(height_face))
                print("")
                #Check for anomalies in the data
                if(alpha_face>1.0 or beta_face>1.0 or width_face>1.0  or height_face>1.0): is_face_detected = False

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
            else:
                 #print("No face detected...")
                 is_face_detected = False

            STATE = "IMG"

        # In this state it is captured a stream of images from
        # the NAO camera and it is convertend in a Numpy matrix
        # The Numpy matrix cam be analysed as an image from OpenCV
        elif(STATE=="IMG"):
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
               img = np.zeros((cam_h, cam_w))

            #Switch to next state
            STATE = "SHOW"

        # Show the image on a window and
        # draws faces and landmarks
        elif(STATE=="SHOW"):
            if(is_face_detected == True):
                #The alpha and beta give the coords
                # of x2 and y2 not of x1 and y1
                face_h = int(height_face * cam_w)
                face_w = int(width_face * cam_w)
                face_centre_x = -1 * (alpha_face - 0.5)
                face_centre_x = int(face_centre_x * cam_w)
                face_centre_y = (beta_face + 0.5)
                face_centre_y = int(face_centre_y * cam_h)
                face_x1 = int(face_centre_x - (face_h / 2))
                face_y1 = int(face_centre_y - (face_h / 2))
                face_x2 = int(face_centre_x + (face_h / 2))
                face_y2 = int(face_centre_y + (face_h / 2))
                cv2.rectangle(img, 
                             (face_x1, face_y1), 
                             (face_x2, face_y2), 
                             (0, 255, 0),
                              2)
                #print("[NAO] Face position:")
                #print("x1: " + str(face_x1) + "; y1: " + str(face_y1))

            #Landmarks detected
            if(len(naomark_list) >= 1):
                counter = 1
                for naomark in naomark_list:
                    id_mark = naomark[0]
                    alpha_mark = naomark[1]
                    beta_mark = naomark[2]
                    size_x_mark = naomark[3]
                    size_y_mark = naomark[4]
                    heading_mark = naomark[5]
                    
                    centre_x_mark = -1 * (alpha_mark - 0.5)
                    centre_x_mark = int(centre_x_mark * cam_w)
                    centre_y_mark = beta_mark + 0.5
                    centre_y_mark = int(centre_y_mark * cam_h)
                    radius = int(size_x_mark * cam_w / 2)
                    centre_y_mark += radius #evaluate if necessary
                    #Draw a blue circle on the landmark
                    cv2.circle(img, 
                             (centre_x_mark, centre_y_mark), 
                             radius, 
                             (255, 0, 0),
                              2)
                    #print("[NAO] Landmarks list has " + str(len(naomark_list)) + " elements...")
                    #print("number: " + str(counter))                   
                    #print("centre_x: " + str(centre_x_mark))
                    #print("centre_y: " + str(centre_y_mark))
                    #print("")
                    #counter += 1

            cv2.imshow('image',img)
            STATE = "KEY"


        # Check which key is pressed and
        # regulates the state macchine passages
        elif(STATE=="KEY"):
            #When pressing Q on the keyboard swith to QUIT state
            #Check which key is pressed
            key_pressed = cv2.waitKey(1)
            #print("Key pressed: " + str(key_pressed))
            if key_pressed==113:  #q=QUIT
                print("[STATE " + str(STATE) + "] " + "Button (q)uit pressed..." + "\n") 
                cv2.destroyAllWindows()
                STATE = "QUIT"
            elif key_pressed==110: #n=NAO
                print("[STATE " + str(STATE) + "] " + "Button (n)ao pressed..." + "\n")
                STATE = "NAO"
            elif key_pressed==103: #g=GOOD
                print("[STATE " + str(STATE) + "] " + "Button (g)ood pressed..." + "\n")
            elif key_pressed==98: #b=BAD
                print("[STATE " + str(STATE) + "] " + "Button (b)ad pressed..." + "\n")
            elif key_pressed==119: #w=WHICH object I'm looking
                print("[STATE " + str(STATE) + "] " + "Button (w)hich pressed..." + "\n")
                which_counter = 0                
                STATE = "WHICH"
            elif key_pressed==104: #w=WHICH object I'm looking
                print("[STATE " + str(STATE) + "] " + "Button (h)elp pressed..." + "\n")
                print("(q)uit, (n)ao, (g)ood, (b)ad, (w)hich, (h)elp" + "\n")
                STATE = "FIND"
            elif key_pressed==-1: #Nothing
                STATE = "COUNTER"
            else:
                print("[STATE " + str(STATE) + "] " + "Button pressed: " + str(key_pressed) + "h\n")

        #Check the state of the counters
        #If some counters are alive then calls the associated states
        elif(STATE=="COUNTER"):
            if(which_counter!= None and which_counter > 0): STATE = "WHICH"
            else: STATE = "FIND"

        #Move the NAO head
        elif(STATE=="NAO"):
            print("[STATE " + str(STATE) + "] " + "Hey NAO, look at me!" + "\n")
            #Reset the head position
            _al_motion_proxy.setAngles("HeadPitch", 0.0, 0.3)
            _al_motion_proxy.setAngles("HeadYaw", 0.0, 0.3)
            STATE = "FIND"

        #Asking to NAO to estimate the Teacher head pose
        #and to generate a head movement from the Hebbian net.
        #If there is a landmark close to the center of the camera
        #the NAO says the name of the associated object.
        elif(STATE=="WHICH"):
            print("[STATE " + str(STATE) + "] " + "NAO, which object I'm looking?" + "\n")
            #Reset the head position
            _al_motion_proxy.setAngles("HeadPitch", 0.5, 0.4)
            #_al_motion_proxy.setAngles("HeadYaw", 0.5, 0.3)
            #Only one landmark found
            if(len(naomark_list) == 1):
                 id_mark = naomark_list[0][0]                
                 try:
                     index_mark = landmark_id_list.index(id_mark)
                     name_mark = landmark_name_list[index_mark]
                     random_sentence = np.random.randint(3)
                     if(random_sentence == 0 and VOICE_ENBLED==True): _al_tts_proxy.say("I see only one object! It's a " + str(name_mark))
                     elif(random_sentence == 1 and VOICE_ENBLED==True): _al_tts_proxy.say("One object here! It's a " + str(name_mark))
                     elif(random_sentence == 2 and VOICE_ENBLED==True): _al_tts_proxy.say("There is one object! It's a " + str(name_mark))
                 #If the landmark is too small it is possible to have
                 #some recognition errors. Here we catch them!
                 except ValueError:
                     if(VOICE_ENBLED==True): _al_tts_proxy.say("I don't know this object.")
                 which_counter = 0
            #Multiple landmarks found
            elif(len(naomark_list) > 1):                 
                 if(VOICE_ENBLED==True): _al_tts_proxy.say("I see many objects!")
                 for naomark in naomark_list:
                     try:
                         id_mark = naomark[0]
                         index_mark = landmark_id_list.index(id_mark)
                         name_mark = landmark_name_list[index_mark]         
                         _al_tts_proxy.say(name_mark) #Says the name of each single object
                     except ValueError:
                         if(VOICE_ENBLED==True): _al_tts_proxy.say("I don't know this object.")
                 which_counter = 0
            else:
                which_counter = which_counter + 1
                if(which_counter >= which_counter_limit):                
                    random_sentence = np.random.randint(3)
                    if(random_sentence == 0 and VOICE_ENBLED==True): _al_tts_proxy.say("I am sorry, I don't see any object!")
                    elif(random_sentence == 1 and VOICE_ENBLED==True): _al_tts_proxy.say("No objects here!")
                    elif(random_sentence == 2 and VOICE_ENBLED==True): _al_tts_proxy.say("There is nothing around!")
                    which_counter = 0

            #Swith to next state
            STATE = "FIND"

        # QUIT State, unsubscribe from all the proxies
        # and QUIT.
        elif(STATE=="QUIT"):
            print("[STATE " + str(STATE) + "] " + "Unsubscribing ALProxy..." + "\n")
            _al_video_proxy.unsubscribe(camera_name_id)
            _al_landmark_proxy.unsubscribe("Test_LandMark")
            _al_face_proxy.unsubscribe("Test_Face")
            #_al_speechrecognition_proxy.unsubscribe("Test_ASR")
            print("[STATE " + str(STATE) + "] " + "Terminating the script!" + "\n")
            return #QUIT

if __name__ == "__main__":
    main()
