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

import numpy as np
import cv2
import time
import sys
import yarp
import acapela
import subprocess
import csv
import deepgaze
from deepgaze.color_classification import HistogramColorClassifier
from deepgaze.motion_detection import MogMotionDetector
from deepgaze.mask_analysis import BinaryMaskAnalyser
import thread
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
        self.rpc_client_head = yarp.RpcClient()
        self.rpc_client_head.addOutput(icub_root+"/head/rpc:i")

        self.rpc_client_head_ikin = yarp.RpcClient()
        self.rpc_client_head_ikin.addOutput(icub_root+"/iKinGazeCtrl/rpc")

    def close(self):
        """Close all the services

        """
        self.port_left_camera.close()
        self.port_right_camera.close()
        self.rpc_client_head.close()

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

    def _track_movement(self):
        delay = 0.5
        pitch = 0
        yaw = 0
        bottle = yarp.Bottle()
        result = yarp.Bottle()
        my_mog_detector = MogMotionDetector()
        my_mask_analyser = BinaryMaskAnalyser()
        t = threading.currentThread()
        while getattr(t, "do_run", True):
            self.port_left_camera.read(self.yarp_image)
            mog_mask = my_mog_detector.returnMask(self.img_array)
            cx, cy = my_mask_analyser.returnMaxAreaCenter(mog_mask)
            #TODO here from the centre it is necessary to estimate
            #a point where to look in yaw/pitch coords
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

    def start_movement_detection(self, delay=1.0):
        try:
            if not self.thread_movement_detection.isAlive():
                self.thread_movement_detection = threading.Thread(target=self._move_head_random, args=(1.5,))
                self.thread_movement_detection.start()
                print "[ICUB] Head control thread started!"
        except:
            print "[ICUB][ERROR] unable to start head control thread"

    def stop_movement_detection(self):
        try:
            if self.thread_movement_detection.isAlive():
                self.thread_movement_detection.do_run = False
                self.thread_movement_detection.join()
                self.set_head_pose(0, 0, 0) #reset the head
                print "[ICUB] Head control thread stopped!"
        except:
            print "[ICUB][ERROR] unable to stop head control thread. Is it running?"

    def set_head_pose_ikin(self, roll, pitch, yaw):
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
        self.rpc_client_head_ikin.write(bottle, result) #Send
        if(result == "[OK]"): return_tuple[0] = True
        else: return_tuple[0] = False
        #Set PITCH
        bottle.clear()
        bottle.addString("set")
        bottle.addString("pos")
        bottle.addInt(0) #Joint
        bottle.addInt(pitch) #Angle
        self.rpc_client_head_ikin.write(bottle, result) #Send
        if(result == "[OK]"): return_tuple[1] = True
        else: return_tuple[1] = False
        #Set YAW
        bottle.clear()
        bottle.addString("set")
        bottle.addString("pos")
        bottle.addInt(2) #Joint
        bottle.addInt(yaw) #Angle
        self.rpc_client_head_ikin.write(bottle, result) #Send
        if result == "[OK]":
            return_tuple[2] = True
        else:
            return_tuple[2] = False

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
        return self.acapela_account_login, self.acapela_application_login, self.acapela_application_password, self.acapela_service_url

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

