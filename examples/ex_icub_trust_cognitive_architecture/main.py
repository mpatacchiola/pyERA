#!/usr/bin/python

#The MIT License (MIT)
#
#Copyright (c) 2016 Massimiliano Patacchiola
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.


import init
from icub import iCub
import Image

def main():

    #1- Load acapela configuration from file
    ACCOUNT_LOGIN, APPLICATION_LOGIN, APPLICATION_PASSWORD, SERVICE_URL = init.load_acapela_config("./acapela_config.csv")
    print("[ACAPELA]Acapela configuration parameters:")
    print("Account Login: " +str(ACCOUNT_LOGIN))
    print("Application Login: " +str(APPLICATION_LOGIN))
    print("Account Password: " +str(APPLICATION_PASSWORD))
    print("Service URL: " +str(SERVICE_URL))
    print("")

    #2-Test ACAPELA
    #icub.say_something("Hello World!", ACCOUNT_LOGIN, APPLICATION_LOGIN, APPLICATION_PASSWORD, SERVICE_URL, '/tmp/', in_background=True)

    #3-iCub Initialisation
    my_icub = iCub(icub_root='/icubSim')
    my_icub.set_head_pose(roll=0, pitch=0, yaw=0)

    image_left = my_icub.return_left_camera_image()
    Image.fromarray(image_left).show()

    my_icub.close()








if __name__ == "__main__":
    main()
