#!/usr/bin/python

## Massimiliano Patacchiola, Plymouth University 2016
#
# In this demo the iCub robot is used to learn the association between
# a teacher head pose and its own head pose. The setup requires a
# table with some objects.
# Pressing the button on the keyboard you can teach the robot and
# control its actions. 
#
# Requirements:
# ------------
#
# PLAY:
# Audio file player for linux. It plays the Acapela audio file.
# 
# ACAPELA:
# Online Text to Speech Syntesizer.
# If you want to use this code you need to register for free
# to the acapela website and get your password and username.
# Otherwise you can use your own speech syntesizer or turn it off.
#
# SWIG:
# sudo apt-get install swig
#
# YARP:
# sudo apt-get install git cmake cmake-curses-gui libgsl0-dev libace-dev libreadline-dev
# sudo apt-get install qtbase5-dev qtdeclarative5-dev qtmultimedia5-dev \
#  qml-module-qtquick2 qml-module-qtquick-window2 \
#  qml-module-qtmultimedia qml-module-qtquick-dialogs \
#  qml-module-qtquick-controls libqt5svg5
# git clone https://github.com/robotology/yarp.git
# cd yarp; mkdir build; cd build; ccmake ../
# CREATE_GUIS, set to ON CREATE_LIB_MATH, set to ON
# For the Python Bindings you have to turn ON the flag YARP_COMPILE_BINDINGS 
# and then CREATE_PYTHON=ON 
#
# YARP (only if you pre-installed YARP without the bindings):
# You can compile the Python binding after the YARP compilation.
# cd $YARP_ROOT/bindings; mkdir build; cd build; ccmake ..;
# Then switch: CREATE_PYTHON=ON
# make: sudo make install;
# export PYTHONPATH=$PYTHONPATH:/path/to/bindings/build
# Now to import yarp in a python project you need the two files: yarp.py and _yarp.so
# If you have problem with the export then you can copy the two files
# directly inside the project folder. I provide the two files compiled in Ubuntu 14.04 (64bit).
# 
# ICUB SImulator:
# Install ODE following: http://wiki.icub.org/wiki/Linux:_Installing_ODE
# sudo apt-get install libglut3 libglut3-dev
# sudo apt-get install libsdl1.2-dev
# sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 57A5ACB6110576A6
# sudo apt-get install icub-common
# git clone https://github.com/robotology/icub-main.git
# mkdir build; cd build; ccmake ..
# Now you have to use CMAKE and turn ON the flags:
# CMAKE_BUILD_TYPE to "Release"
# If you want to use the virtual screen: OPENCV_GRABBER=ON
# ENABLE_icubmod_cartesiancontrollerclient ON
# ENABLE_icubmod_cartesiancontrollerserver ON
# ENABLE_icubmod_gazecontrollerclient      ON
# Make sure the compiler matches the one you used to compile YARP.
# make; sudo make install
#
# YARP Topics
# -----------
# The Topics used in this examples are 
# "/icubSim/cam/left" to acquire the images from one camera.
# "/icubSim/head/rpc:i" the head control topic (6 joints)
# Tutorial: http://wiki.icub.org/brain/icub_python_simworld_control.html
# HEAD axes: 0=Pitch, 1=Roll, 2=Yaw
# To start a virtual screen with the webcam projection:
# yarpdev --device opencv_grabber
# yarp connect /grabber /icubSim/texture/screen

import numpy as np
import cv2
import time
import sys
import yarp
import acapela
import subprocess
import os
import dlib

from deepgaze.haar_cascade import haarCascade

#Importing the pyERA modules
from pyERA.som import Som
from pyERA.utils import ExponentialDecay

##
# It sets the icub head using the RPC port
# HEAD axes: 0=Pitch, 1=Roll, 2=Yaw
#
#
def set_icub_head_pose(rpc_client, roll, pitch, yaw):
    bottle = yarp.Bottle()
    result = yarp.Bottle()
    return_list = list()
    #Set ROLL
    bottle.clear()
    bottle.addString("set")
    bottle.addString("pos")
    bottle.addInt(1) #Joint
    bottle.addInt(roll) #Angle
    rpc_client.write(bottle, result) #Send
    if(result == "[OK]"): return_list.append(True)
    else: return_list.append(False)
    #Set PITCH
    bottle.clear()
    bottle.addString("set")
    bottle.addString("pos")
    bottle.addInt(0) #Joint
    bottle.addInt(pitch) #Angle
    rpc_client.write(bottle, result) #Send
    if(result == "[OK]"): return_list.append(True)
    else: return_list.append(False)
    #Set YAW
    bottle.clear()
    bottle.addString("set")
    bottle.addString("pos")
    bottle.addInt(2) #Joint
    bottle.addInt(yaw) #Angle
    rpc_client.write(bottle, result) #Send
    if(result == "[OK]"): return_list.append(True)
    else: return_list.append(False)


##
# It uses the Festival TTS to produce a sentence
# The process can run in background so it does not require
# to wait until it finishes.
#
def say_something(text, in_background=True):
    if(in_background==True): os.system("echo \"" + str(text) +  "\" | festival --tts &")
    else: os.system("echo \"" + str(text) +  "\" | festival --tts")


def main():
    # The state machine has the following states:
    # [VOID, INIT, FIND, SHOW, KEY, ICUB, WHICH, QUIT]
    #
    # VOID: Empty state, to use for test
    # INIT: It is called only once for the initialisation
    # FIND: use the hardware face and landmark detection libraries
    # SHOW: Print the image on screen using OpenCV
    # KEY: Check which key is pressed
    # ICUB: Pressing the (h) button the robot look in front of itself
    # WHICH: Pressing the (w) button is like asking to the robot to look to a object on the table
    # QUIT: Pressing (q) unsubscribe and close the script

    #Configuration Variables, adjust to taste
    #ICUB_IP = "192.168.0.100"
    #ICUB_PORT = 9559
    RECORD_VIDEO = True #If True record a video from the ICUB camera
    USE_FESTIVAL_TTS = True #To use the Festival Text To Speach
    #If you want to use Acapela TTS you have to fill the
    #following variable with the correct values
    #USE_ACAPELA_TTS = False
    #ACCOUNT_LOGIN = '---'
    #APPLICATION_LOGIN = '---'
    #APPLICATION_PASSWORD = '---'
    #SERVICE_URL = 'http://vaas.acapela-group.com/Services/Synthesizer'
    #The initial state
    STATE = "VOID" 

    while(True):

        #Empty STATE, to be used for test
        if(STATE == "VOID"):            
            STATE = "INIT"

        # The zero state is an init phase
        # In this state all the proxies are
        # created and tICUB subscribe to the services
        #
        elif(STATE=="INIT"):
            #Init some generic variables
            #This counter allows continuing the program flow
            #without calling a sleep
            which_counter = 0 #Counter increased when WHICH is called
            which_counter_limit = 30 #Limit for the which counter

            #Init YARP
            print("[STATE " + str(STATE) + "] " + "YARP init" + "\n")
            yarp.Network.init()
            
            #Camera connection
            try:
                print("[STATE " + str(STATE) + "] " + "Waiting for '/icubSim/cam/left' ..." + "\n")
                cam_w = 320 #640
                cam_h = 240 #480
                input_port = yarp.Port()
                input_port.open("/pyera-image-port")
                yarp.Network.connect("/icubSim/cam/left", "/pyera-image-port")
                img_array = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
                yarp_image = yarp.ImageRgb()
                yarp_image.resize(cam_w, cam_h)
                yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
            except BaseException, err:
                print("[ERROR] connect To Camera catching error " + str(err))
                return

            #Head connection and Reset
            print("[STATE " + str(STATE) + "] " + "Waiting for '/icubSim/head/rpc:i' ..." + "\n")
            rpc_client = yarp.RpcClient()
            rpc_client.addOutput("/icubSim/head/rpc:i")
            print("[STATE " + str(STATE) + "] " + "Reset the head position..." + "\n")
            set_icub_head_pose(rpc_client, roll=0, pitch=0, yaw=0)

            #Initialise the OpenCV video recorder
            if(RECORD_VIDEO == True):
                print("[STATE " + str(STATE) + "] " + "Starting the video recorder..." + "\n")
                fourcc = cv2.cv.CV_FOURCC(*'XVID')
                video_out = cv2.VideoWriter("./output.avi", fourcc, 20.0, (cam_w,cam_h))

            #Init dlib Face detector
            #my_face_detector = dlib.get_frontal_face_detector()

            #Init the Deepgaze face detector
            my_cascade = haarCascade("./haarcascade_frontalface_alt.xml", "./haarcascade_profileface.xml")

            #Talking            
            if(USE_FESTIVAL_TTS == True): 
                print("[STATE " + str(STATE) + "] " + "Trying the TTS Festival..." + "\n")
                say_something("Hello World, I'm ready!")

            #if(USE_ACAPELA_TTS == True):                
                #print("[ACAPELA] Downloading the mp3 file...")
                #tts_acapela = acapela.Acapela(account_login=ACCOUNT_LOGIN, application_login=APPLICATION_LOGIN, application_password=APPLICATION_PASSWORD, 
                                              #service_url=SERVICE_URL, quality='22k', directory='/tmp/')    
                #tts_acapela.prepare(text="Hello world, I'm ready!", lang='US', gender='M', intonation='NORMAL')
                #output_filename = tts_acapela.run()
                #subprocess.Popen(["play","-q","/tmp/" + str(output_filename)])
                #print "[ACAPELA] Recorded TTS to %s" % output_filename           

            #Swithc to STATE > 1
            print("[STATE " + str(STATE) + "] " + "Switching to next state" + "\n")
            time.sleep(2)
            STATE = "FIND"

        # Get data from landmark detection and 
        # face detection.
        #
        elif(STATE=="FIND"):

            #Get Data for Face Detection
            #if(face_data and isinstance(face_data, list) and len(face_data) > 0):
                #print("[ICUB] Face detected!")
            #else:
                 #print("No face detected...")
                 #is_face_detected = False
            input_port.read(yarp_image)

            '''
            faces_array = my_face_detector(img_array, 1)
            print("Total Faces: " + str(len(faces_array)))
            for i, pos in enumerate(faces_array):

                face_x1 = pos.left()
                face_y1 = pos.top()
                face_x2 = pos.right()
                face_y2 = pos.bottom()
                text_x1 = face_x1
                text_y1 = face_y1 - 3

                cv2.putText(img_array, "FACE " + str(i+1), (text_x1,text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1);
                cv2.rectangle(img_array, 
                         (face_x1, face_y1), 
                         (face_x2, face_y2), 
                         (0, 255, 0), 
                          2)
            '''
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            # Return code: 1=Frontal, 2=FrontRotLeft, 3=FronRotRight,
            #              4=ProfileLeft, 5=ProfileRight.
            my_cascade.findFace(gray, runFrontal=True, runFrontalRotated=False, runLeft=True, runRight=True, 
                                frontalScaleFactor=1.18, rotatedFrontalScaleFactor=1.18, leftScaleFactor=1.18, 
                                rightScaleFactor=1.18, minSizeX=70, minSizeY=70, rotationAngleCCW=30, 
                                rotationAngleCW=-30, lastFaceType=my_cascade.face_type)   

            face_x1 = my_cascade.face_x 
            face_y1 = my_cascade.face_y 
            face_x2 = my_cascade.face_x + my_cascade.face_w 
            face_y2 = my_cascade.face_y + my_cascade.face_h
            text_x1 = face_x1
            text_y1 = face_y1 - 3
            if(my_cascade.face_type == 1 or my_cascade.face_type == 2 or my_cascade.face_type == 3): cv2.putText(img_array, "FRONTAL", (text_x1,text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1);
            elif(my_cascade.face_type == 4): cv2.putText(img_array, "LEFT", (text_x1,text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1);
            elif(my_cascade.face_type == 5): cv2.putText(img_array, "RIGH", (text_x1,text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1);
            cv2.rectangle(img_array, 
                             (face_x1, face_y1), 
                             (face_x2, face_y2), 
                             (0, 255, 0),
                              2)

            is_face_detected = False
            STATE = "SHOW"

        # Show the image on a window and
        # draws faces and landmarks
        elif(STATE=="SHOW"):
            #Show the image and record the video
            #Yarp Bug? It is necessary to create a new object (img_array_bgr)
            #it is not possible to reuse img_array.
            img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            cv2.imshow('image',img_array_bgr)
            if(RECORD_VIDEO == True): video_out.write(img_array_bgr)
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
            elif key_pressed==105: #i=ICUB
                print("[STATE " + str(STATE) + "] " + "Button (i)Cub pressed..." + "\n")
                STATE = "ICUB"
            elif key_pressed==103: #g=GOOD
                print("[STATE " + str(STATE) + "] " + "Button (g)ood pressed..." + "\n")
            elif key_pressed==98: #b=BAD
                print("[STATE " + str(STATE) + "] " + "Button (b)ad pressed..." + "\n")
            elif key_pressed==114: #r=REWARD
                print("[STATE " + str(STATE) + "] " + "Button (r)eward pressed..." + "\n")
            elif key_pressed==112: #p=PUNISHMENT
                print("[STATE " + str(STATE) + "] " + "Button (p)unishment pressed..." + "\n")
                STATE = "PUNISHMENT"
            elif key_pressed==119: #w=WHICH object I'm looking
                print("[STATE " + str(STATE) + "] " + "Button (w)hich pressed..." + "\n")
                which_counter = 0                
                STATE = "WHICH"
            elif key_pressed==104: #w=WHICH object I'm looking
                print("[STATE " + str(STATE) + "] " + "Button (h)elp pressed..." + "\n")
                print("(q)uit, (n)ao, (g)ood, (b)ad, (w)hich, (h)elp" + "\n")
                STATE = "FIND"
            elif key_pressed==-1: #Nothing
                STATE = "FIND"
            else:
                print("[STATE " + str(STATE) + "] " + "Button pressed: " + str(key_pressed) + "\n")

        #Move the ICUB head
        elif(STATE=="ICUB"):
            print("[STATE " + str(STATE) + "] " + "Hey ICUB, look at me!" + "\n")
            #Reset the head position
            set_icub_head_pose(rpc_client, roll=0, pitch=0, yaw=0)
            if(USE_FESTIVAL_TTS == True):
                random_sentence = np.random.randint(4)                
                if(random_sentence == 0): say_something("I'm ready!")
                elif(random_sentence == 1): say_something("What can I do for you?")
                elif(random_sentence == 2): say_something("Ready to help.")
                elif(random_sentence == 3): say_something("How can I help?")
            STATE = "FIND"

        #Asking to ICUB to estimate the Teacher head pose
        #and to generate a head movement from the Hebbian net.
        #If there is a landmark close to the center of the camera
        #the ICUB says the name of the associated object.
        elif(STATE=="WHICH"):
            print("[STATE " + str(STATE) + "] " + "ICUB, which object I'm looking?" + "\n")
            #Reset the head position
            set_icub_head_pose(rpc_client, roll=0, pitch=-30, yaw=20)
            if(USE_FESTIVAL_TTS == True):
                random_sentence = np.random.randint(4)                
                if(random_sentence == 0): say_something("I think you are looking in this direction!")
                elif(random_sentence == 1): say_something("Are you looking there?")
                elif(random_sentence == 2): say_something("Maybe you are looking there.")
                elif(random_sentence == 3): say_something("Is this the correct direction?")
            #Swith to next state
            STATE = "FIND"

        #the ICUB says the name of the associated object.
        elif(STATE=="REWARD"):
            print("[STATE " + str(STATE) + "] " + "Reward given!" + "\n")
            STATE = "FIND"

        #the ICUB says the name of the associated object.
        elif(STATE=="PUNISHMENT"):
            print("[STATE " + str(STATE) + "] " + "Punishment given!" + "\n")
            if(USE_FESTIVAL_TTS == True):
                random_sentence = np.random.randint(4)                
                if(random_sentence == 0): say_something("I will do better nex time!")
                elif(random_sentence == 1): say_something("Sorry, I'm still learning.")
                elif(random_sentence == 2): say_something("Sorry, I'm still working on my skills.")
                elif(random_sentence == 3): say_something("I miss it!")
            #Reset the head position
            set_icub_head_pose(rpc_client, roll=0, pitch=0, yaw=0)
            STATE = "FIND"

        # QUIT State, unsubscribe from all the proxies
        # and QUIT.
        elif(STATE=="QUIT"):
            print("[STATE " + str(STATE) + "] " + "Closing the YARP ports..." + "\n")
            # Cleanup
            input_port.close()
            rpc_client.close()
            if(USE_FESTIVAL_TTS == True):
                print("[STATE " + str(STATE) + "] " + "Saying Goodbye..." + "\n")                
                say_something("Bye bye!")
            print("[STATE " + str(STATE) + "] " + "Terminating the script!" + "\n")
            
            return #QUIT

if __name__ == "__main__":
    main()
