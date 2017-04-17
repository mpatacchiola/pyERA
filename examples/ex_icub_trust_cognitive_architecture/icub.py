#!/usr/bin/python

# The MIT License (MIT)
#
# Copyright (c) 2017 Massimiliano Patacchiola
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Command to lunch in simulation mode
# yarpserver
# ./iCub_SIM
# ./iKinGazeCtrl --from configSim.ini
# yarpdev --device opencv_grabber
# yarp connect /grabber /icubSim/texture/screen
#
# For the cartesian controller of the left arm
# ./simCartesianControl
# ./iKinCartesianSolver --context simCartesianControl --part left_arm

import numpy as np
import cv2
import time
import yarp
import acapela
import subprocess
import csv
from deepgaze.color_classification import HistogramColorClassifier
from deepgaze.color_detection import BackProjectionColorDetector
from deepgaze.motion_detection import MogMotionDetector
from deepgaze.mask_analysis import BinaryMaskAnalyser
import threading
import random


class iCub:

    def __init__(self, icub_root='/icubSim'):
        # Global variables
        self.thread_movement_detection = threading.Thread(target=None)
        self.acapela_account_login = ''
        self.acapela_application_login = ''
        self.acapela_application_password = ''
        self.acapela_service_url = ''
        # Deepgaze variables
        self.object_list = list()
        self.histogram_classifier = HistogramColorClassifier(channels=[0, 1, 2], hist_size=[128, 128, 128],
                                                             hist_range=[0, 256, 0, 256, 0, 256], hist_type='BGR')

        # Init YARP
        yarp.Network.init()
        # Camera connection
        try:
            cam_w = 320  # 640
            cam_h = 240  # 480
            # Left camera
            print("[ICUB] Init: Waiting for " + icub_root + "/cam/left' ...")
            self.port_left_camera = yarp.Port()
            self.port_left_camera.open("/pyera-left-image-port")
            yarp.Network.connect(icub_root+"/cam/left", "/pyera-left-image-port")
            # right camera
            print("[ICUB] Init: Waiting for " + icub_root +  "/cam/right' ...")
            self.port_right_camera = yarp.Port()
            self.port_right_camera.open("/pyera-right-image-port")
            yarp.Network.connect(icub_root+"/cam/right", "/pyera-right-image-port")
            # Set the numpy array to fill with the image
            self.img_array = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            self.yarp_image = yarp.ImageRgb()
            self.yarp_image.resize(cam_w, cam_h)
            self.yarp_image.setExternal(self.img_array, self.img_array.shape[1], self.img_array.shape[0])
        except BaseException, err:
            print("[ICUB][ERROR] connect To Camera catching error " + str(err))
            return

        try:
            if icub_root.find("Sim") > -1:
                print("[ICUB] Simulation Mode, connecting grabber to texture/screen ")
                # yarp connect /grabber /icubSim/texture/screen
                yarp.Network.connect("/grabber", icub_root + "/texture/screen")
        except BaseException, err:
            print("[ICUB][ERROR] connecting /grabber to /texture/screen catching error " + str(err))
            return

        try:
            self.port_ikin_mono = yarp.Port()
            self.port_ikin_mono.open("/pyera-ikin-mono")
            yarp.Network.connect("/pyera-ikin-mono", "/iKinGazeCtrl/mono:i")
        except BaseException, err:
            print("[ICUB][ERROR] connect To iKinGazeCtrl/mono catching error " + str(err))
            return

        try:
            self.port_ikin_stereo = yarp.Port()
            self.port_ikin_stereo.open("/pyera-ikin-stereo")
            yarp.Network.connect("/pyera-ikin-stereo", "/iKinGazeCtrl/stereo:i")
        except BaseException, err:
            print("[ICUB][ERROR] connect To iKinGazeCtrl/stereo catching error " + str(err))
            return

        try:
            self.port_ikin_xd = yarp.Port()
            self.port_ikin_xd.open("/pyera-ikin-xd")
            yarp.Network.connect("/pyera-ikin-xd", "/iKinGazeCtrl/xd:i")
        except BaseException, err:
            print("[ICUB][ERROR] connect To iKinGazeCtrl/xd catching error " + str(err))
            return

        try:
            self.port_cart_leftarm = yarp.Port()
            self.port_cart_leftarm.open("/pyera-cart-leftarm")
            yarp.Network.connect("/pyera-cart-leftarm", "/cartesianSolver/left_arm/in")
        except BaseException, err:
            print("[ICUB][ERROR] connect To /cartesianSolver/left_arm/in catching error " + str(err))
            return

        self.rpc_client_head = yarp.RpcClient()
        self.rpc_client_head.addOutput(icub_root+"/head/rpc:i")

        self.rpc_client_head_ikin = yarp.RpcClient()
        self.rpc_client_head_ikin.addOutput("/iKinGazeCtrl/rpc")

    def close(self):
        """Close all the services

        """
        self.port_left_camera.close()
        self.port_right_camera.close()
        self.rpc_client_head.close()

    def check_connection(self):
        """Check if the internet connection is present or not
        
        @return: True if connected, otherwise False
        """
        import socket
        try:
            host = socket.gethostbyname("www.google.com")
            socket.create_connection((host, 80), 2)
            return True
        except:
            pass
        return False

    def return_left_camera_image(self, mode='RGB'):
        """Return a numpy array with the LEFT camera image

        @param mode the image to return (default RGB)
            RGB: Red Green Blue image
            BGR: Blue Green Red (OpenCV)
            GRAY: Grayscale image
        """
        self.port_left_camera.read(self.yarp_image)
        if(mode=='BGR'):
            return cv2.cvtColor(self.img_array, cv2.COLOR_RGB2BGR)
        elif(mode=='RGB'):
            return self.img_array
        elif(mode=='GRAY'):
            return cv2.cvtColor(self.img_array, cv2.COLOR_BGR2GRAY)
        else:
            return self.img_array

    def return_right_camera_image(self, mode='RGB'):
        """Return a numpy array with the RIGHT camera image

        @param mode the image to return (default RGB)
            RGB: Red Green Blue image
            BGR: Blue Green Red (OpenCV)
            GRAY: Grayscale image
        """
        self.port_right_camera.read(self.yarp_image)
        if(mode=='BGR'):
            return cv2.cvtColor(self.img_array, cv2.COLOR_RGB2BGR)
        elif(mode=='RGB'):
            return self.img_array
        elif(mode=='GRAY'):
            return cv2.cvtColor(self.img_array, cv2.COLOR_BGR2GRAY)
        else:
            return self.img_array

    def _set_pose_left_hand(self, x, y, z, ax, ay, az, theta):
        """ This is a low level function which must be used carefully.
        
        It allows setting the position and orientation of the left hand.
        @param x: the x position (negative to move in front of the robot)
        @param y: the y position (negative to move on the left side of the robot)
        @param z: the z position (positive to move up)
        @param ax: the x orientation (zero for hand touching the left lef)
        @param ay: the y orientation (zero for hand touching left leg)
        @param az: the z orientation (zero for hand touching the left leg)
        @param theta: the angle theta
        """
        bottle = yarp.Bottle()
        bottle.clear()
        bottle.addString('xd')
        tmp0 = bottle.addList()
        tmp0.addDouble(x)
        tmp0.addDouble(y)
        tmp0.addDouble(z)
        tmp0.addDouble(ax)
        tmp0.addDouble(ay)
        tmp0.addDouble(az)
        tmp0.addDouble(theta)
        self.port_cart_leftarm.write(bottle)

    #TODO: This function must be implemented for a safe object manipulation
    # def move_left_hand_to_position(self, x, y, z):
    #    self._set_pose_left_hand(x, y, z, 0, 0, 0, 0)


    def move_head_to_target_mono(self, type, u, v, z):
        """ given a point in the image (mono) it moves the head
        to that point. 
        
        WARNING: it requires iKinGazeCtrl to run.
        @param type: left or right image (string)
        @param u: the x coordinate of the point
        @param v: the y coordinate of the point
        @param z: the estimated depth in the eye coord frame
        """
        bottle = yarp.Bottle()
        bottle.clear()
        bottle.addString(type)
        bottle.addDouble(u)
        bottle.addDouble(v)
        bottle.addDouble(z)
        self.port_ikin_mono.write(bottle)

    def move_head_to_target_stereo(self, u_left, v_left, u_right, v_right):
        """ Move the head to a point defined in the two image plane (stereo)
        
        @param u_left: x coord in the left image
        @param v_left: y coord in the left image
        @param u_right: x coord in the right image
        @param v_right: y coord in the right image
        """
        bottle = yarp.Bottle()
        bottle.clear()
        bottle.addDouble(u_left)
        bottle.addDouble(v_left)
        bottle.addDouble(u_right)
        bottle.addDouble(v_right)
        self.port_ikin_stereo.write(bottle)

    def move_head_to_point(self, x, y, z):
        """ Given a point in the space it moves the head 
        in the direction of the point.
        
        @param x: the x coord of the point
        @param y: the y coord of the point
        @param z: the z coord of the point
        """
        bottle = yarp.Bottle()
        bottle.clear()
        bottle.addDouble(x)
        bottle.addDouble(y)
        bottle.addDouble(z)
        self.port_ikin_xd.write(bottle)

    def _move_head_random(self, delay=1.0):
        t = threading.currentThread()
        while getattr(t, "do_run", True):
            roll = 0
            pitch = random.randint(a=-30, b=+30)
            yaw = random.randint(a=-20, b=+20)
            bottle = yarp.Bottle()
            result = yarp.Bottle()
            # Set ROLL
            bottle.clear()
            bottle.addString("set")
            bottle.addString("pos")
            bottle.addInt(1)  # Joint
            bottle.addInt(roll)  # Angle
            self.rpc_client_head.write(bottle, result)  # Send
            # Set PITCH
            bottle.clear()
            bottle.addString("set")
            bottle.addString("pos")
            bottle.addInt(0)  # Joint
            bottle.addInt(pitch)  # Angle
            self.rpc_client_head.write(bottle, result)  # Send
            # Set YAW
            bottle.clear()
            bottle.addString("set")
            bottle.addString("pos")
            bottle.addInt(2)  # Joint
            bottle.addInt(yaw)  # Angle
            self.rpc_client_head.write(bottle, result)  # Send
            time.sleep(delay)

    def _track_movement(self, template_path, delay=0.5):
        """ Given a colour template it tracks the 
        
        @param delay: 
        @return: 
        """
        my_mask_analyser = BinaryMaskAnalyser()
        t = threading.currentThread()
        template = cv2.imread(template_path)  # Load the image
        my_back_detector = BackProjectionColorDetector()  # Defining the deepgaze color detector object
        my_back_detector.setTemplate(template)  # Set the template
        #cv2.namedWindow('filtered')
        while getattr(t, "do_run", True):
            #img_array = np.zeros((360,240,3), dtype=np.uint8)
            img_array = self.return_left_camera_image(mode='BGR')
            image_filtered = my_back_detector.returnFiltered(img_array, morph_opening=True,
                                                             blur=True, kernel_size=7, iterations=2)
            cx, cy = my_mask_analyser.returnMaxAreaCenter(image_filtered)
            if cx is not None:
                cv2.circle(image_filtered,(cx,cy), 5, (0, 0, 255), -1)
                bottle = yarp.Bottle()
                bottle.clear()
                bottle.addString('left')
                bottle.addDouble(cx)
                bottle.addDouble(cy)
                bottle.addDouble(1.0)
                self.port_ikin_mono.write(bottle)
            #images_stack = np.hstack((img_array, image_filtered))
            #cv2.imshow('filtered', images_stack)
            #cv2.waitKey(100) #waiting 50 msec
            time.sleep(0.1)
        #cv2.destroyWindow('filtered')

    def start_movement_detection(self, template_path, delay=1.0):
        try:
            if not self.thread_movement_detection.isAlive():
                self.thread_movement_detection = threading.Thread(target=self._track_movement,
                                                                  args=(template_path, 0.5,))
                self.thread_movement_detection.start()
                print "[ICUB] Head control thread started!"
        except:
            print "[ICUB][ERROR] unable to start head control thread"

    def stop_movement_detection(self):
        try:
            if self.thread_movement_detection.isAlive():
                self.thread_movement_detection.do_run = False
                self.thread_movement_detection.join()
                self.reset_head_pose()  # reset the head
                print "[ICUB] Head control thread stopped!"
        except:
            print "[ICUB][ERROR] unable to stop head control thread. Is it running?"

    def is_movement_detection(self):
        """Check if the movement tracking is active
        
        @return: return True if the movement tracking is active 
        """
        return self.thread_movement_detection.isAlive()

    def set_head_pose(self, roll, pitch, yaw):
        """It sets the icub head using the RPC port
           HEAD axes: 0=Pitch, 1=Roll, 2=Yaw

        @param roll (degree) int
        @param pitch (degree) int
        @param yaw (degree) int
        """
        bottle = yarp.Bottle()
        result = yarp.Bottle()
        return_tuple = [False, False, False]
        #Set ROLL
        bottle.clear()
        bottle.addString("set")
        bottle.addString("pos")
        bottle.addInt(1) #Joint
        bottle.addInt(roll) #Angle
        self.rpc_client_head.write(bottle, result) #Send
        if result == "[OK]":
            return_tuple[0] = True
        else:
            return_tuple[0] = False
        # Set PITCH
        bottle.clear()
        bottle.addString("set")
        bottle.addString("pos")
        bottle.addInt(0)  # Joint
        bottle.addInt(pitch)  # Angle
        self.rpc_client_head.write(bottle, result)  # Send
        if result == "[OK]":
            return_tuple[1] = True
        else:
            return_tuple[1] = False
        # Set YAW
        bottle.clear()
        bottle.addString("set")
        bottle.addString("pos")
        bottle.addInt(2)  # Joint
        bottle.addInt(yaw)  # Angle
        self.rpc_client_head.write(bottle, result)  # Send
        if result == "[OK]":
            return_tuple[2] = True
        else:
            return_tuple[2] = False

    def reset_head_pose(self):
        """Reset the eyes and head position to 0,0,0

        """
        bottle = yarp.Bottle()
        result = yarp.Bottle()
        #Set ROLL
        bottle.clear()
        bottle.addString("set")
        bottle.addString("pos")
        bottle.addInt(1) #Joint
        bottle.addInt(0) #Angle
        self.rpc_client_head.write(bottle, result) #Send
        # Set PITCH
        bottle.clear()
        bottle.addString("set")
        bottle.addString("pos")
        bottle.addInt(0)  # Joint
        bottle.addInt(0)  # Angle
        self.rpc_client_head.write(bottle, result)  # Send
        # Set YAW
        bottle.clear()
        bottle.addString("set")
        bottle.addString("pos")
        bottle.addInt(2)  # Joint
        bottle.addInt(0)  # Angle
        self.rpc_client_head.write(bottle, result)  # Send
        # Set EYE-YAW
        bottle.clear()
        bottle.addString("set")
        bottle.addString("pos")
        bottle.addInt(3)  # Joint
        bottle.addInt(0)  # Angle
        self.rpc_client_head.write(bottle, result)  # Send
        # Set EYE-PITCH
        bottle.clear()
        bottle.addString("set")
        bottle.addString("pos")
        bottle.addInt(4)  # Joint
        bottle.addInt(0)  # Angle
        self.rpc_client_head.write(bottle, result)  # Send

    def get_3d_mono_angles(self, type, u, v, z):
        """ returns the 3D point whose projected pixel coordinates (u,v) 
        in the image plane <type> ["left"|"right"] along with third 
        component <z> in the eye's reference frame are given.
        It requires iKinGaze to run. In the simulator mode should
        be activated with: ./iKinGazeCtrl --from configSim.ini
        
        WARNING: has been hard to find the way for adding a list
        in a bottle, the icub documentation should be improved.
        The trick is: tmp_var = bottle.addList()
        @param type: 'let' or 'right' camera
        @param u: pixel coordinate x
        @param v: pixel coordinate y
        @param z: third component point in front of the robot (eye reference frame)
        @return: the 3D point (x,y,z) coordinates
        """
        bottle = yarp.Bottle()
        result = yarp.Bottle()
        bottle.clear()
        bottle.addString('get')
        bottle.addString('3D')
        bottle.addString('mono')
        tmp0 = bottle.addList()
        tmp0.addString('left')
        tmp0.addInt(35)
        tmp0.addInt(35)
        tmp0.addInt(35)
        self.rpc_client_head_ikin.write(bottle, result)
        list_bottle = result.get(1).asList()
        list_return = []
        for i in range(list_bottle.size()):
            list_return.append(list_bottle.get(i).asDouble())
        return list_return

    def get_3d_stereo_angles(self, u_left, v_left, u_right, v_right):
        """ returns the 3D point whose projected pixel coordinates (u,v) 
        in the image plane <type> ["left"|"right"] along with third 
        component <z> in the eye's reference frame are given.
        It requires iKinGaze to run. In the simulator mode should
        be activated with: ./iKinGazeCtrl --from configSim.ini

        WARNING: has been hard to find the way for adding a list
        in a bottle, the icub documentation should be improved.
        The trick is: tmp_var = bottle.addList()
        @param type: 'let' or 'right' camera
        @param u: pixel coordinate x
        @param v: pixel coordinate y
        @return: the 3D point (x,y,z) coordinates
        """
        bottle = yarp.Bottle()
        result = yarp.Bottle()
        bottle.clear()
        bottle.addString('get')
        bottle.addString('3D')
        bottle.addString('stereo')
        tmp0 = bottle.addList()
        tmp0.addInt(u_left)
        tmp0.addInt(v_left)
        tmp0.addInt(u_right)
        tmp0.addInt(v_right)
        self.rpc_client_head_ikin.write(bottle, result)
        list_bottle = result.get(1).asList()
        list_return = []
        for i in range(list_bottle.size()):
            list_return.append(list_bottle.get(i).asDouble())
        return list_return

    def set_acapela_credential(self, csv_path):
        '''Load the ACAPELA config parameters

        The first line of the CSV must contain:
        account_login, application_login, 
        application_password, service_url.
        @param csv_path the path to the config file
        '''
        with open(csv_path, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                self.acapela_account_login = row[0]
                self.acapela_application_login = row[1]
                self.acapela_application_password = row[2]
                self.acapela_service_url = row[3]

    def get_acapela_credential(self):
        '''Return the ACAPELA config parameters

        '''
        return self.acapela_account_login, self.acapela_application_login, \
               self.acapela_application_password, self.acapela_service_url

    def say_something(self, text, directory='/tmp/', in_background=True):
        """It says something using ACAPELA tts

        @param text the string to say
        @param in_background run the process in background
        """
        print("[ICUB][ACAPELA] Downloading the mp3 file...")

        tts_acapela = acapela.Acapela(self.acapela_account_login, self.acapela_application_login,
                                      self.acapela_application_password, self.acapela_service_url,
                                      quality='22k', directory=directory)
        tts_acapela.prepare(text=text, lang='US', gender='M', intonation='NORMAL')
        output_filename = tts_acapela.run()
        print "[ICUB][ACAPELA] Recorded TTS to %s" % output_filename
        subprocess.Popen(["play","-q",directory + str(output_filename)])
        print "[ICUB][PLAY] reproducing the acapela file"

    def learn_object_from_histogram(self, template, name):
        """Using the deepgaze histogram classifier to save an object.
        
        @param template: the image template to store
        @param name: the name of the model (must be a unique ID)
        """
        self.histogram_classifier.addModelHistogram(template, name)

    def remove_object_from_histogram(self, name):
        """Given an object remove it from the list
        
        @param name: the name of the object. 
        @return: True if the object has been deleted
        """
        return self.histogram_classifier.removeModelHistogramByName(name)

    def recall_object_from_histogram(self, template):
        """Return the name of the object with the closest similarity to the template.
        
        @param template: the image to recall
        @return: the name of the object with closest similarity
        """
        if self.histogram_classifier.returnSize() == 0:
            return None
        else:
            return self.histogram_classifier.returnBestMatchIndexName(template, method="intersection")

"""
def main():
    my_cub = iCub()
    #my_cub.return_left_camera_image()
    #print(my_cub.get_3d_mono_angles(type='left', u=0, v=0, z=0).toString())
    #print(my_cub.get_3d_mono_angles(type='left', u=0, v=0, z=0).size())
    #result = my_cub.get_3d_stereo_angles(u_left=23, v_left=20, u_right=35, v_right=20)
    #print(result.size())
    #print (result.get(1).isList())
    #print(result.get(1).get(0).asDouble())
    #rint(yarpListToTuple(result.get(1).asList()))

    #my_cub.move_head_to_target(type='left', u=50, v=50, z=0.5)
    my_cub.reset_head_pose()
    #my_cub.move_head_to_target_mono(type='left', u=0, v=0, z=0.5)
    #time.sleep(2)
    #my_cub.start_movement_detection()
    #time.sleep(30)
    #my_cub.stop_movement_detection()
    my_cub._set_pose_left_hand(-0.1, -0.4, 0.0, 0, 0, 0, 0.0)
    #my_cub.move_left_hand_to_position(-0.3, -0.3, 0.3)

if __name__ == "__main__":
    main()
"""