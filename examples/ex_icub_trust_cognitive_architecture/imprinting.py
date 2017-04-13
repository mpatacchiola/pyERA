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

from speech_recognition import SpeechRecognizer
from icub import iCub
import cv2
import random
import thread
import time

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
    if 'find' or 'search' in speech_string:
        response_list = ['All right! I will find the ',
                         'Ok, I will look for the ',
                         'Searching the ',
                         'I am looking for a ']
        response_string = response_list[random.randint(0,len(response_list)-1)] + speech_string.rsplit(None, 1)[-1]
        state = 'key'
    elif 'learn new object' in speech_string:
        response_list = ['I like to learn! This is a ',
                         'Ok, this is a ',
                         '']
        response_string = response_list[random.randint(0,len(response_list)-1)] + speech_string.rsplit(None, 1)[-1]
        state = 'key'
    elif 'what is this' in speech_string:
        response_list = ['Let me see',
                         'Let me think.',
                         'Just a second']
        response_string = response_list[random.randint(0,len(response_list)-1)]
        state = 'key'
    elif 'start movement detection' in speech_string:
        response_list = ["Ok, now I'm looking for moving objects",
                         'Ok I will track movements',
                         'Ready to track!']
        response_string = response_list[random.randint(0, len(response_list)-1)]
        state = 'movedetect on'
    elif 'stop movement detection' in speech_string:
        response_list = ["Ok, no more movements",
                         'Ok I will stop it',
                         "I'm gonna stop it!"]
        response_string = response_list[random.randint(0, len(response_list)-1)]
        state = 'movedetect off'
    else:
        response_list = ["Sorry I did not understand what you said.",
                         'Sorry, What did you say?',
                         'Repeat again please.']
        response_string = response_list[random.randint(0,len(response_list)-1)]
        state = 'key'

    return response_string, state


def main():
    STATE = 'show'
    speech_string = ""
    my_speech, my_icub = initialise()
    my_icub.say_something(text="I'm ready!")

    while True:
        if STATE == 'record':
            my_speech.record_audio("/tmp/audio.wav", seconds=4, extension='wav', harddev='3,0')
            raw_file_path = my_speech.convert_to_raw(file_name="/tmp/audio.wav", file_name_raw="/tmp/audio.raw", extension='wav')
            speech_string = my_speech.return_text_from_audio("/tmp/audio.raw")
            print("[STATE " + str(STATE) + "] " + "Speech recognised: " + speech_string)
            STATE = 'understand'

        elif STATE == 'understand':
            response_string, local_state = speech_to_action(speech_string)
            my_icub.say_something(text=response_string)
            STATE = local_state

        elif STATE == 'show':
            left_image = my_icub.return_left_camera_image(mode='RGB')
            cv2.imshow('left image', left_image)
            STATE = 'key'

        elif STATE == 'move':
            my_icub.set_head_pose(0, -40, random.randint(a=-20, b=+20))
            STATE = 'key'

        elif STATE == 'movedetect on':
            print("[STATE " + str(STATE) + "] " + "starting movement tracking" + "\n")
            my_icub.start_movement_detection(delay=1.0)
            STATE = 'key'

        elif STATE == 'movedetect off':
            print("[STATE " + str(STATE) + "] " + "stopping movement tracking" + "\n")
            my_icub.stop_movement_detection()
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
