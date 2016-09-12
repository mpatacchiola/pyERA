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
from head_pose_estimation import CnnHeadPoseEstimator
import tensorflow as tf

#It requires the pyERA library
from pyERA.som import Som
from pyERA.hebbian import HebbianNetwork

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
    DEBUG = False
    NAO_IP = "192.168.0.100"
    NAO_PORT = 9559
    VOICE_ENBLED = True #If True the robot speaks
    RECORD_VOICE = False #If True record all the sentences
    RECORD_VIDEO = False #If True record a video from the NAO camera
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
            which_counter_limit = 15 #Limit for the which counter

            #Init the deepgaze head pose estimator and 
            # launch the graph in a session.
            print("[STATE " + str(STATE) + "] " + "Deepgaze init" + "\n")
            sess = tf.Session()
            my_head_pose_estimator = CnnHeadPoseEstimator(sess)
            my_head_pose_estimator.allocate_yaw_variables()
            #my_head_pose_estimator.print_allocated_variables()
            my_head_pose_estimator.load_yaw_variables("./cnn_cccdd_30k")
            yaw_teacher = 0.0 #Init the teacher yaw variable
            yaw_robot = 0.0
            pitch_robot = 0.0

            #Init the Self-Organizing Maps
            print("[STATE " + str(STATE) + "] " + "Self-Organizing Maps init" + "\n")
            som_robot = Som(matrix_size=3, input_size=1) #the robot SOM is 8x8 matrix
            print("[STATE " + str(STATE) + "] " + "Loading the network from: som_nao_eight.npz" + "\n")
            #som_robot.load("som_nao_eight.npz")
            som_robot.set_unit_weights(-90.0, 0, 0); som_robot.set_unit_weights(-80.0, 0, 1); som_robot.set_unit_weights(-45.0, 0, 2); 
            som_robot.set_unit_weights(-25.0, 1, 0); som_robot.set_unit_weights(  0.0, 1, 1); som_robot.set_unit_weights(+25.0, 1, 2);
            som_robot.set_unit_weights(+45.0, 2, 0); som_robot.set_unit_weights(+80.0, 2, 1); som_robot.set_unit_weights(+90.0, 2, 2);
            #For simplicity we consider only the Yaw angle of the teacher
            #Instead of a SOM we store the values in a vector. We can have 7 discrete positions:
            #som_teacher = np.array([-80.0, -45.0, -25.0, 0.0, +25.0, +45.0, +80.0])
            som_teacher = Som(matrix_size=3, input_size=1)
            #We set the weights value manually
            som_teacher.set_unit_weights(-90.0, 0, 0); som_teacher.set_unit_weights(-80.0, 0, 1); som_teacher.set_unit_weights(-45.0, 0, 2); 
            som_teacher.set_unit_weights(-25.0, 1, 0); som_teacher.set_unit_weights(  0.0, 1, 1); som_teacher.set_unit_weights(+25.0, 1, 2);
            som_teacher.set_unit_weights(+45.0, 2, 0); som_teacher.set_unit_weights(+80.0, 2, 1); som_teacher.set_unit_weights(+90.0, 2, 2);

            #Init the Hebbian Network
            hebbian_network = HebbianNetwork("naonet")
            hebbian_network.add_node("som_robot", (3,3))
            hebbian_network.add_node("som_teacher", (3,3))
            hebbian_network.add_connection(0, 1) #connecting som_robot -> som_teacher
            hebbian_network.print_info()
          
            #Getting the nao proxies
            print("[STATE " + str(STATE) + "] " + "ALProxy init" + "\n")
            _al_motion_proxy = ALProxy("ALMotion", NAO_IP, int(NAO_PORT))
            _al_posture_proxy = ALProxy("ALRobotPosture", NAO_IP, int(NAO_PORT))
            _al_tts_proxy = ALProxy("ALTextToSpeech", NAO_IP, int(NAO_PORT))
            _al_landmark_proxy = ALProxy("ALLandMarkDetection", NAO_IP, int(NAO_PORT))
            _al_face_proxy = ALProxy("ALFaceDetection", NAO_IP, int(NAO_PORT))
            _al_memory_proxy = ALProxy("ALMemory", NAO_IP, int(NAO_PORT))
            _al_video_proxy = ALProxy("ALVideoDevice", NAO_IP, int(NAO_PORT))
            if(RECORD_VIDEO == True): _al_video_rec_proxy = ALProxy("ALVideoRecorder", NAO_IP, int(NAO_PORT))
            if(RECORD_VIDEO == True): _al_video_rec_proxy.stopRecording() #reset a dangling connection
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
                #Settings for resolution 1 (320x240)
                resolution_type = 1
                fps=15
                cam_w = 320
                cam_h = 240
                #Settigns for resolution 2 (320x240)
                #resolution_type = 2
                #fps = 15
                #cam_w = 640
                #cam_h = 480
                camera_name_id = _al_video_proxy.subscribeCamera("Test_Video", 0, resolution_type, 13, fps)
                print("[STATE " + str(STATE) + "] " + "Connected to the camera with resolution: " + str(cam_w) + "x" + str(cam_h) + "\n")
            except BaseException, err:
                print("[ERROR] connectToCamera: catching error " + str(err))
                return
            #Adding to the speech recognition proxy a vocabulary
            #_al_speechrecognition_proxy.setLanguage("English")
            #vocabulary = ["good", "bad", "nao"]
            #_al_speechrecognition_proxy.setVocabulary(vocabulary, False)
            #_al_speechrecognition_proxy.setVocabulary(vocabulary, False) #If you want to enable word spotting
            #_al_speechrecognition_proxy.subscribe("Test_ASR")

            #Initialise the OpenCV video recorder
            if(RECORD_VIDEO == True):
                print("[STATE " + str(STATE) + "] " + "Starting the video recorder..." + "\n")
                fourcc = cv2.cv.CV_FOURCC(*'XVID')
                video_out = cv2.VideoWriter("./output.avi", fourcc, 20.0, (cam_w,cam_h))
                #Record also the NAO session
                _al_video_rec_proxy.setResolution(2) #Resolution VGA  640*480
		_al_video_rec_proxy.setFrameRate(30)
		#_al_video_rec_proxy.setVideoFormat("MJPG")
		#self._video_proxy.startVideoRecord(complete_path)
		_al_video_rec_proxy.startRecording("/home/nao/recordings/cameras", "last_session", True) #It worked saving in this path!

            #Save all the sentences inside the NAO memory
            #it is usefull if you want a clean audio
            #to present in a video.
            if(RECORD_VOICE == True):
                print("[STATE " + str(STATE) + "] " + "Saving the voice in '/home/nao/recordings/microphones'" + "\n")
                _al_tts_proxy.sayToFile("Hello world!", "/home/nao/recordings/microphones/hello_wolrd.wav")
                _al_tts_proxy.sayToFile("I see only one object! It's a " , "/home/nao/recordings/microphones/i_see_only_one.wav")
                _al_tts_proxy.sayToFile("One object here! It's a " , "/home/nao/recordings/microphones/one_object.wav")
                _al_tts_proxy.sayToFile("There is one object! It's a " , "/home/nao/recordings/microphones/there_is_one.wav")
                _al_tts_proxy.sayToFile("I am sorry, I don't see any object!", "/home/nao/recordings/microphones/i_dont_see_any.wav")
                _al_tts_proxy.sayToFile("No objects here!", "/home/nao/recordings/microphones/no_objects.wav")
                _al_tts_proxy.sayToFile("There is nothing around!", "/home/nao/recordings/microphones/there_is_nothing.wav")
                _al_tts_proxy.sayToFile("I am doing better!", "/home/nao/recordings/microphones/im_doing_better.wav")
                _al_tts_proxy.sayToFile("Very good!", "/home/nao/recordings/microphones/very_good.wav")
                _al_tts_proxy.sayToFile("Catch it!", "/home/nao/recordings/microphones/catch_it.wav")
                _al_tts_proxy.sayToFile("I will do better nex time!", "/home/nao/recordings/microphones/i_will_do_better.wav")
                _al_tts_proxy.sayToFile("Sorry, I'm still learning.", "/home/nao/recordings/microphones/still_learning.wav")
                _al_tts_proxy.sayToFile("I miss it!", "/home/nao/recordings/microphones/miss_it.wav")

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

            #Hello world!!!
            if(VOICE_ENBLED==True): _al_tts_proxy.say("Hello world!")

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
                if(DEBUG==True): print("[NAO] Landmark detected!")
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
                    if(DEBUG==True): print("Mark: " + str(id_mark))
                    if(DEBUG==True): print("Alpha: " + str(alpha_mark))
                    if(DEBUG==True): print("Beta: " + str(beta_mark))
                    if(DEBUG==True): print("SizeX: " + str(size_x_mark))
                    if(DEBUG==True): print("SizeY: " + str(size_y_mark))
                    if(DEBUG==True): print("Heading: " + str(heading_mark))
                    if(DEBUG==True): print("")
            #else:
                 #print("No naomark detected...")
                 #is_landmark_detected = False

            #Get Data for Face Detection
            face_data = _al_memory_proxy.getData("FaceDetected", 0)
            if(face_data and isinstance(face_data, list) and len(face_data) > 0):
            #if(len(face_data) > 0):
                if(DEBUG==True): print("[NAO] Face detected!")
                face_vector = face_data[1]
                face_info = face_vector[0]
                is_face_detected = True
                alpha_face = face_info[0][1]
                beta_face = face_info[0][2]
                width_face = face_info[0][3]
                height_face = face_info[0][4]
                if(DEBUG==True): print("Alpha: " + str(alpha_face))
                if(DEBUG==True): print("Beta: " + str(beta_face))
                if(DEBUG==True): print("Width: " + str(width_face))
                if(DEBUG==True): print("Height: " + str(height_face))
                if(DEBUG==True): print("")
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

            #We use a copy of the original image to write over
            #in this way the original image is unchanged
            temp_img = np.copy(img)

            #Draw the face rectangle
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
                cv2.rectangle(temp_img, 
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
                    cv2.circle(temp_img, 
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

            #Show the image and record the video
            cv2.imshow('image',temp_img)
            if(RECORD_VIDEO == True): video_out.write(img)
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
            elif key_pressed==102: #f=FACE
                print("[STATE " + str(STATE) + "] " + "Button (f)ace pressed..." + "\n")
                STATE = "FACE"
            elif key_pressed==114: #r=REWARD
                print("[STATE " + str(STATE) + "] " + "Button (r)eward pressed..." + "\n")
                STATE = "REWARD"
            elif key_pressed==112: #p=PUNISHMENT
                print("[STATE " + str(STATE) + "] " + "Button (p)unishment pressed..." + "\n")
                STATE = "PUNISHMENT"
            elif key_pressed==119: #w=WHICH object I'm looking
                print("[STATE " + str(STATE) + "] " + "Button (w)hich pressed..." + "\n")
                which_counter = 0                
                STATE = "WHICH"
            elif key_pressed==104: #h=HELP
                print("[STATE " + str(STATE) + "] " + "Button (h)elp pressed..." + "\n")
                print("(q)uit, (n)ao, (r)eward, (p)unishment, (f)ace, (w)hich, (h)elp" + "\n")
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
            #Produce a sentence
            random_sentence = np.random.randint(3)
            if(random_sentence == 0 and VOICE_ENBLED==True): _al_tts_proxy.say("Hello!")
            elif(random_sentence == 1 and VOICE_ENBLED==True): _al_tts_proxy.say("I'm ready!")
            elif(random_sentence == 2 and VOICE_ENBLED==True): _al_tts_proxy.say("Yes?")
            STATE = "FIND"


        # Find the HEAD POSE given a FACE
        # it uses the deepgaze library
        elif(STATE=="FACE"):
            #Evaluate the YAW angle
            yaw_teacher = np.random.uniform(low=-90.0, high=90.0) #TODO remove this line
            '''
            if(is_face_detected == False):
                print("[STATE " + str(STATE) + "] " + "Head Pose Estimation: Failed because no face is present" + "\n")
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
                #Crop the face from the image
                if(face_h >= 64):
                    image_cropped = img[face_y1:face_y2, face_x1:face_x2] # Crop from x, y, w, h
                    image_resized = cv2.resize(image_cropped, (64, 64), interpolation = cv2.INTER_AREA)
                    #Show the cropped face
                    if(DEBUG==True): cv2.imshow('face', image_resized)
                    #Pass the croppend face to the Convolutional Neural Network
                    yaw_vector = my_head_pose_estimator.return_yaw(image_resized)
                    yaw_teacher = yaw_vector[0,0,0] #Set the global variable
                    print("[STATE " + str(STATE) + "] " + "Head Pose Estimation: Yaw = " + str(yaw_vector[0,0,0]) + "\n")
                    #if(VOICE_ENBLED==True): _al_tts_proxy.say("Catch it!")
                else:
                    print("[STATE " + str(STATE) + "] " + "Head Pose Estimation: Failed because image is less than 64x64 pixels" + "\n")
            '''
            #Switch to next state
            STATE = "FIND"

        #Asking to NAO to estimate the Teacher head pose
        #and to generate a head movement from the Hebbian net.
        #If there is a landmark close to the center of the camera
        #the NAO says the name of the associated object.
        #Before calling this state you have to call the state "FACE"
        #which set the SOMs variables.
        elif(STATE=="WHICH"):
            print("[STATE " + str(STATE) + "] " + "NAO, which object I'm looking?" + "\n")

            #1- Find the most similar head pose in the som_teacher
            #som_teacher_similarity_matrix = som_teacher.return_similarity_matrix(yaw_teacher)
            som_teacher_activation_matrix = som_teacher.return_activation_matrix(yaw_teacher)
            som_teacher_bmu_value = som_teacher.return_BMU_weights(yaw_teacher)
            print("[STATE " + str(STATE) + "] " + "CNN Yaw: " + str(yaw_teacher) + "; SOM Yaw: " + str(som_teacher_bmu_value) + "\n")

            #2- Backward pass from som_teacher to som_robot in the Hebbian Network
            hebbian_network.set_node_activations(1, som_teacher_activation_matrix)
            som_robot_hebbian_matrix = hebbian_network.compute_node_activations(0, set_node_matrix=False)
            max_row, max_col = np.unravel_index(som_robot_hebbian_matrix.argmax(), som_robot_hebbian_matrix.shape)
            print("Robot Hebbian Matrix: ")
            print(som_robot_hebbian_matrix)
            print("[STATE " + str(STATE) + "] " + "The Robot SOM activated unit is (" + str(max_row) + ", " + str(max_col) + ") with value: " + str(som_robot.get_unit_weights(max_row,max_col)) + "\n")
            print("")
            #3- The BMU of robot_som is returned and the robot head moves in that direction
            som_robot_bmu_weights = som_robot.get_unit_weights(max_row, max_col)
            yaw_robot = som_robot_bmu_weights[0] * (np.pi/180.0)
            pitch_robot = 0 #som_robot_bmu_weights[1] * (np.pi/180.0)
            _al_motion_proxy.setAngles("HeadYaw", yaw_robot, 0.2)
            _al_motion_proxy.setAngles("HeadPitch", pitch_robot, 0.2)

            #5- Check which landamrks are in the Field-of-view
            # and says the name of them
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

            #6-Swith to next state
            STATE = "FIND"

        #the NAO says the name of the associated object.
        elif(STATE=="REWARD"):

            print("[STATE " + str(STATE) + "] " + "Giving the Reward..." + "\n")

            #1- Compute the real distance_matrix
            input_vector = np.array([yaw_robot])
            som_robot_similarity_matrix = som_robot.return_similarity_matrix(input_vector)
            input_vector = np.array([yaw_teacher])
            som_teacher_activation_matrix = som_teacher.return_activation_matrix(input_vector)

            #2- Set the distance matrices as activation of the Hebbian Network
            print("Robot Activation Matrix: ")
            print(som_robot_similarity_matrix)
            print("")
            print("Teacher Activation Matrix: ")
            print(som_teacher_activation_matrix)
            print("")
            hebbian_network.set_node_activations(0, som_robot_similarity_matrix)
            hebbian_network.set_node_activations(1, som_teacher_activation_matrix)

            #3- Positive Learning!
            hebbian_network.learning(learning_rate=0.3, rule="hebb")

            print("[STATE " + str(STATE) + "] " + "Reward given!" + "\n")

            #Reset the head position
            _al_motion_proxy.setAngles("HeadPitch", 0.0, 0.3)
            _al_motion_proxy.setAngles("HeadYaw", 0.0, 0.3)
            #Produce a sentence
            random_sentence = np.random.randint(3)
            if(random_sentence == 0 and VOICE_ENBLED==True): _al_tts_proxy.say("I am doing better!")
            elif(random_sentence == 1 and VOICE_ENBLED==True): _al_tts_proxy.say("Very good!")
            elif(random_sentence == 2 and VOICE_ENBLED==True): _al_tts_proxy.say("Catch it!")
            STATE = "FIND"

        #the ICUB says the name of the associated object.
        elif(STATE=="PUNISHMENT"):

            print("[STATE " + str(STATE) + "] " + "Giving the Punishment..." + "\n")

            #1- Compute the real distance_matrix
            input_vector = np.array([yaw_robot])
            som_robot_similarity_matrix = som_robot.return_similarity_matrix(input_vector)
            input_vector = np.array([yaw_teacher])
            #som_teacher_similarity_matrix = som_teacher.return_similarity_matrix(input_vector)
            som_teacher_activation_matrix = som_teacher.return_activation_matrix(input_vector)

            #2- Set the distance matrices as activation of the Hebbian Network
            print("Robot Activation Matrix: ")
            print(som_robot_similarity_matrix)
            print("")
            print("Teacher Activation Matrix: ")
            print(som_teacher_activation_matrix)
            print("")
            hebbian_network.set_node_activations(0, som_robot_similarity_matrix)
            hebbian_network.set_node_activations(1, som_teacher_activation_matrix)

            #3- Negative Learning!
            hebbian_network.learning(learning_rate=0.3, rule="antihebb")

            print("[STATE " + str(STATE) + "] " + "Punishment given!" + "\n")

            #Reset the head position
            _al_motion_proxy.setAngles("HeadPitch", 0.0, 0.3)
            _al_motion_proxy.setAngles("HeadYaw", 0.0, 0.3)
            #Produce a sentence
            random_sentence = np.random.randint(3)
            if(random_sentence == 0 and VOICE_ENBLED==True): _al_tts_proxy.say("I will do better nex time!")
            elif(random_sentence == 1 and VOICE_ENBLED==True): _al_tts_proxy.say("Sorry, I'm still learning.")
            elif(random_sentence == 2 and VOICE_ENBLED==True): _al_tts_proxy.say("I miss it!")
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
            if(RECORD_VIDEO == True): _al_video_rec_proxy.stopRecording()
            return #QUIT

if __name__ == "__main__":
    main()
