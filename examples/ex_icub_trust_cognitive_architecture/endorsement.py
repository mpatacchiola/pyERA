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

#ATTENTION: to work it requires to lunch the iCub world:
# yarpserver
# ./iCub_SIM
# ./iKinGazeCtrl --from configSim.ini
# yarpdev --device opencv_grabber
# yarp connect /grabber /icubSim/texture/screen
#
# For the cartesian controller of the left arm
# ./simCartesianControl
# ./iKinCartesianSolver --context simCartesianControl --part left_arm

# PocketSphinx valid Commands are:
# The prefix [iCub] or [hey] is optional
# learn <object name>
# this is a <object name>
# forget <object name>
# what is this
# find the <object name>
# stop detection
# look at me

from speech_recognition import SpeechRecognizer
from icub import iCub
import cv2
import random
import time
import os
import sys

def initialise():
    # Initialise the speech recognition engine and the iCub controller
    my_speech = SpeechRecognizer(
        hmm_path="/home/massimiliano/pyERA/examples/ex_icub_trust_cognitive_architecture/sphinx/model/en-us/en-us",
        language_model_path="/home/massimiliano/pyERA/examples/ex_icub_trust_cognitive_architecture/sphinx/model/en-us/en-us.lm.bin",
        dictionary_path="/home/massimiliano/pyERA/examples/ex_icub_trust_cognitive_architecture/sphinx/data/icub.dic",
        grammar_path="/home/massimiliano/pyERA/examples/ex_icub_trust_cognitive_architecture/sphinx/data/icub.gram",
        rule_name='icub.basicCmd',
        fsg_name="icub")
    # iCub initialization
    my_icub = iCub(icub_root='/icubSim')
    # Load acapela configuration from file
    my_icub.set_acapela_credential("./acapela_config.csv")
    account_login, application_login, application_password, service_url = my_icub.get_acapela_credential()
    print("[ACAPELA]Acapela configuration parameters:")
    print("Account Login: " + str(account_login))
    print("Application Login: " + str(application_login))
    print("Account Password: " + str(application_password))
    print("Service URL: " + str(service_url))
    print("")
    # Return the objects
    return my_speech, my_icub


def speech_to_action(speech_string):
    """ Take the sentence from the speech recognition and plan an action
    <action> = (learn new object | watch | inspect | find | search | look | what | start | stop);
    <target> = (ball | cup | book | dog | chair | table | at me | is this | movement detection);
    @param speech_string: 
    @return: 
    """
    if speech_string.find('learn') > -1 or speech_string.find('this is a') > -1:
        response_list = ['I like to learn! This is a ',
                         'Ok, this is a ',
                         'I learned a new object, ',
                         '']
        object_name = speech_string.rsplit(None, 1)[-1]
        response_string = response_list[random.randint(0, len(response_list)-1)] + object_name
        state = 'learn'
    elif speech_string.find('what is this') > -1:
        response_string = ""
        state = 'what'
    elif speech_string.find('find the') > -1 or speech_string.find('search the') > -1:
        object_name = speech_string.rsplit(None, 1)[-1]
        object_path = "./objects/" + str(object_name) + ".png"
        if not os.path.isfile(object_path):
            print("[SPEECH-TO-ACTION][WARNING] " + "this file does not exist: " + str(object_path) + "\n")
            response_string = "Sorry I do not know this object!"
            state = 'key'
        else:
            response_list = ["Ok, now I'm looking for a ",
                             'Ok I will track the ',
                             'Ready to track the ']
            response_string = response_list[random.randint(0, len(response_list)-1)] + object_name
            state = 'movedetect on'
    elif speech_string.find('stop detection') > -1:
        response_list = ["Ok, no more movements",
                         'Ok I will stop it',
                         "I'm gonna stop it!"]
        response_string = response_list[random.randint(0, len(response_list)-1)]
        state = 'movedetect off'
    elif speech_string.find('look at me') > -1:
        response_list = ["Ok!",
                         'Sure!']
        response_string = response_list[random.randint(0, len(response_list)-1)]
        state = 'look'
    else:
        response_list = ["Sorry I did not understand.",
                         'Sorry, can you repeat?',
                         'Repeat again please.']
        response_string = response_list[random.randint(0,len(response_list)-1)]
        state = 'key'
    return response_string, state


def main():
    inputfile = ''
    outputfile = ''
    informant_name = ''
    if len(sys.argv) == 1 or len(sys.argv) > 4:
        print("python familiarization.py <inputfile> <outputfilename> <informant_name>")
    elif len(sys.argv) == 4:
        inputfile = sys.argv[1]
        outputfile = sys.argv[2]
        informant_name = sys.argv[3]

    print("Input file: " + str(inputfile))
    print("Output file: " + str(outputfile))
    print("Informant Name: " + str(informant_name))

    STATE = 'show'
    speech_string = ""
    fovea_offset = 40 # side of the fovea square
    my_speech, my_icub = initialise()
    is_connected = my_icub.check_connection()
    if is_connected:
        print("[STATE Init] intenet connection present.")
    else:
        print("[STATE Init][ERROR] internet connection not present!!!")
    my_icub.say_something(text="I'm ready!")
    cv2.namedWindow('main')


    while True:
        if STATE == 'record':
            #image = my_icub.return_left_camera_image(mode='BGR')
            my_speech.record_audio("/tmp/audio.wav", seconds=3, extension='wav', harddev='3,0')
            raw_file_path = my_speech.convert_to_raw(file_name="/tmp/audio.wav", file_name_raw="/tmp/audio.raw", extension='wav')
            speech_string = my_speech.return_text_from_audio("/tmp/audio.raw")
            print("[STATE " + str(STATE) + "] " + "Speech recognised: " + speech_string)
            STATE = 'understand'

        elif STATE == 'understand':
            response_string, local_state = speech_to_action(speech_string)
            print("[STATE " + str(STATE) + "] " + "Speech recognised: " + speech_string)
            print("[STATE " + str(STATE) + "] " + "Next state: " + local_state)
            my_icub.say_something(text=response_string)
            STATE = local_state

        elif STATE == 'show':
            left_image = my_icub.return_left_camera_image(mode='BGR')
            img_cx = int(left_image.shape[1] / 2)
            img_cy = int(left_image.shape[0] / 2)
            cv2.rectangle(left_image,
                          (img_cx-fovea_offset, img_cy-fovea_offset),
                          (img_cx+fovea_offset, img_cy+fovea_offset),
                          (0, 255, 0), 1)
            cv2.imshow('main', left_image)
            STATE = 'key'

        elif STATE == 'movedetect on':
            object_name = response_string.rsplit(None, 1)[-1]
            print("[STATE " + str(STATE) + "] " + "start tracking of: " + str(object_name) + "\n")
            object_path = "./objects/" + str(object_name) + ".png"
            if my_icub.is_movement_detection():
                    my_icub.stop_movement_detection()
                    time.sleep(0.5)
                    my_icub.start_movement_detection(template_path=object_path, delay=1.0)
            else:
                    my_icub.start_movement_detection(template_path=object_path, delay=1.0)
            STATE = 'key'

        elif STATE == 'movedetect off':
            print("[STATE " + str(STATE) + "] " + "stop movement tracking" + "\n")
            my_icub.stop_movement_detection()
            time.sleep(0.5)
            my_icub.reset_head_pose()
            STATE = 'key'

        elif STATE == 'look':
            print("[STATE " + str(STATE) + "] " + "gaze reset" + "\n")
            my_icub.reset_head_pose()
            STATE = 'key'

        elif STATE == 'learn':
            object_name = response_string.rsplit(None, 1)[-1]
            print("[STATE " + str(STATE) + "] " + "Learning new object: " + object_name + "\n")
            left_image = my_icub.return_left_camera_image(mode='BGR')
            #left_image = image
            img_cx = int(left_image.shape[1] / 2)
            img_cy = int(left_image.shape[0] / 2)
            left_image = left_image[img_cy-fovea_offset:img_cy+fovea_offset,
                                    img_cx-fovea_offset:img_cx+fovea_offset]
            my_icub.learn_object_from_histogram(left_image, object_name)
            print("[STATE " + str(STATE) + "] " + "Writing new template in ./objects/" + object_name + ".png" + "\n")
            cv2.imwrite('./objects/' + str(object_name) + '.png', left_image)
            STATE = 'key'

        elif STATE == 'what':
            print("[STATE " + str(STATE) + "] " + "Recalling object from memory..." + "\n")
            left_image = my_icub.return_left_camera_image(mode='BGR')
            #left_image = image
            img_cx = int(left_image.shape[1] / 2)
            img_cy = int(left_image.shape[0] / 2)
            left_image = left_image[img_cy-25:img_cy+25, img_cx-25:img_cx+25]
            object_name = my_icub.recall_object_from_histogram(left_image)
            if object_name is None:
                my_icub.say_something("My memory is empty. Teach me something!")
            else:
                print("[STATE " + str(STATE) + "] " + "Name returned: " + str(object_name) + "\n")
                response_list = ["Let me see. I think this is a ",
                                 "Let me think. It's a ",
                                 "Just a second. It may be a ",
                                 "It should be a "]
                response_string = response_list[random.randint(0, len(response_list) - 1)]
                my_icub.say_something(response_string + str(object_name))
            STATE = 'key'

        elif STATE == 'key':
            key_pressed = cv2.waitKey(10) # delay in millisecond
            if key_pressed==113:  #q=QUIT
                print("[STATE " + str(STATE) + "] " + "Button (q)uit pressed..." + "\n")
                STATE = "close"
            elif key_pressed==110: #n=
                print("[STATE " + str(STATE) + "] " + "Button (n) pressed..." + "\n")
            elif key_pressed==102: #f=
                print("[STATE " + str(STATE) + "] " + "Button (f) pressed..." + "\n")
            elif key_pressed == 114:  # r=RECORD
                print("[STATE " + str(STATE) + "] " + "Button (r)ecord pressed..." + "\n")
                STATE = "record"
            else:
                STATE = 'show'

        elif STATE == 'close':
            my_icub.say_something(text="See you soon, bye bye!")
            my_icub.stop_movement_detection()
            my_icub.close()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
