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


import init
from icub import iCub
from reinforcement_learning import ReinforcementLearner
import Image
import random

def main():
    # 1- Load acapela configuration from file
    account_login, application_login, application_password, service_url = init.load_acapela_config("./acapela_config.csv")
    print("[ACAPELA]Acapela configuration parameters:")
    print("Account Login: " + str(account_login))
    print("Application Login: " + str(application_login))
    print("Account Password: " + str(application_password))
    print("Service URL: " + str(service_url))
    print("")

    # 2-iCub Initialisation
    # my_icub = iCub(icub_root='/icubSim')
    # my_icub.say_something("Hello World!",
    #                       account_login,
    #                       application_login,
    #                       application_password,
    #                       service_url,
    #                       directory='/tmp/',
    #                       in_background=True)
    # my_icub.set_head_pose(roll=0, pitch=0, yaw=0)
    # image_left = my_icub.return_left_camera_image()
    # Image.fromarray(image_left).show()
    # my_icub.close()

    # 3-Initialisation of the learning categories
    informant_list = [[1, 1000], [1, 1], [1, 1]]
    dict_informants = {'CAREGIVER': 0, 'RELIABLE': 1, 'UNRELIABLE': 2}
    dict_actions = {'REJECT': 0, 'ACCEPT': 1}
    list_reliability = ['UNRELIABLE', 'RELIABLE']
    my_learner = ReinforcementLearner(tot_images=12, tot_labels=12, tot_actions=2, informant_vector=informant_list)
    object_name_list = ['cup', 'book', 'ball', 'shoe',
                        'dog', 'chair', 'loma', 'mido',
                        'wug', 'dax', 'blicket', 'dawnoo']
    dict_images = {'CUP': 0, 'BOOK': 1, 'BALL': 2, 'SHOE': 3,
                   'DOG': 4, 'CHAIR': 5, 'LOMA': 6, 'MIDO': 7,
                   'WUG': 8,'DAX': 9, 'BLICKET': 10, 'DAWNOO': 11}
    dict_labels = {'cup': 0, 'book': 1, 'ball': 2, 'shoe': 3,
                   'dog': 4, 'chair': 5, 'loma': 6, 'mido': 7,
                   'wug': 8, 'dax': 9, 'blicket': 10, 'dawnoo': 11}

    # 4- Imprinting phase, the caregiver teach the name of the objects (both positive and negative associations)
    print("####### IMPRINTING ########")
    positive_imprinting = [(dict_images['CUP'], dict_labels['cup'], dict_informants['CAREGIVER'], dict_actions['ACCEPT']),
                          (dict_images['BOOK'], dict_labels['book'], dict_informants['CAREGIVER'], dict_actions['ACCEPT']),
                          (dict_images['BALL'], dict_labels['ball'], dict_informants['CAREGIVER'], dict_actions['ACCEPT']),
                          (dict_images['SHOE'], dict_labels['shoe'], dict_informants['CAREGIVER'], dict_actions['ACCEPT']),
                          (dict_images['DOG'], dict_labels['dog'], dict_informants['CAREGIVER'], dict_actions['ACCEPT']),
                          (dict_images['CHAIR'], dict_labels['chair'], dict_informants['CAREGIVER'], dict_actions['ACCEPT'])]

    negative_imprinting = [(dict_images['CUP'], dict_labels['book'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['CUP'], dict_labels['ball'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['CUP'], dict_labels['shoe'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['CUP'], dict_labels['dog'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['CUP'], dict_labels['chair'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['BOOK'], dict_labels['cup'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['BOOK'], dict_labels['ball'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['BOOK'], dict_labels['shoe'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['BOOK'], dict_labels['dog'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['BOOK'], dict_labels['chair'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['BALL'], dict_labels['cup'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['BALL'], dict_labels['book'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['BALL'], dict_labels['shoe'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['BALL'], dict_labels['dog'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['BALL'], dict_labels['chair'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['SHOE'], dict_labels['cup'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['SHOE'], dict_labels['book'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['SHOE'], dict_labels['ball'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['SHOE'], dict_labels['dog'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['SHOE'], dict_labels['chair'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['DOG'], dict_labels['cup'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['DOG'], dict_labels['book'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['DOG'], dict_labels['ball'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['DOG'], dict_labels['shoe'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['DOG'], dict_labels['chair'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['CHAIR'], dict_labels['cup'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['CHAIR'], dict_labels['book'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['CHAIR'], dict_labels['ball'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['CHAIR'], dict_labels['shoe'], dict_informants['CAREGIVER'], dict_actions['REJECT']),
                          (dict_images['CHAIR'], dict_labels['dog'], dict_informants['CAREGIVER'], dict_actions['REJECT'])]

    my_learner.training(dataset=positive_imprinting, repeat=15, model='4yo')
    my_learner.training(dataset=negative_imprinting, repeat=15, model='4yo')

    # Testing the network on known random objects
    for _ in range(8):
        rand_object_index = random.randint(a=0,b=11)  # in python randint b is included in the sampling
        object_label_index = my_learner.predict_object_name(object_image_index=rand_object_index)
        print("SHOWING: " + str(object_name_list[rand_object_index]))
        print("PREDICTED: " + str(object_name_list[object_label_index]))
        print("--------")

    print("QUESTION: Is it the CAREGIVER reliable or unreliable?")
    caregiver_reliability = my_learner.predict_informant_reliability(dict_informants['CAREGIVER'])
    print("ANSWER: " + str(list_reliability[caregiver_reliability]))


    # 2- FAMILIARISATION: a set of known objects is presented
    # The reliable informant always gives the correct label
    # The unreliable informant always gives the wrong label
    print("")
    print("####### FAMILIARISATION ########")
    reliable_familiarisation = [(dict_images['BALL'], dict_labels['ball'], dict_informants['RELIABLE'], dict_actions['ACCEPT']),
                                (dict_images['CUP'], dict_labels['cup'], dict_informants['RELIABLE'], dict_actions['ACCEPT']),
                                (dict_images['BOOK'], dict_labels['book'], dict_informants['RELIABLE'], dict_actions['ACCEPT'])]

    my_learner.training(dataset=reliable_familiarisation, repeat=2, model='4yo')
    print("QUESTION: Is it the RELIABLE informant reliable or unreliable?")
    caregiver_reliability = my_learner.predict_informant_reliability(dict_informants['RELIABLE'])
    print("ANSWER: " + str(list_reliability[caregiver_reliability]))

    unreliable_familiarisation = [(dict_images['BALL'], dict_labels['shoe'], dict_informants['UNRELIABLE'], dict_actions['ACCEPT']),
                                (dict_images['CUP'], dict_labels['dog'], dict_informants['UNRELIABLE'], dict_actions['ACCEPT']),
                                (dict_images['BOOK'], dict_labels['chair'], dict_informants['UNRELIABLE'], dict_actions['ACCEPT'])]

    my_learner.training(dataset=unreliable_familiarisation, repeat=2, model='4yo')
    print("QUESTION: Is it the UNRELIABLE informant reliable or unreliable?")
    caregiver_reliability = my_learner.predict_informant_reliability(dict_informants['UNRELIABLE'])
    print("ANSWER: " + str(list_reliability[caregiver_reliability]))

if __name__ == "__main__":
    main()
