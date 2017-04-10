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


import cv2
import sys
import yarp
import acapela
import subprocess
import csv
from deepgaze.color_classification import HistogramColorClassifier


#ACCOUNT_LOGIN = ''
#APPLICATION_LOGIN = ''
#APPLICATION_PASSWORD = ''
#SERVICE_URL = 'http://vaas.acapela-group.com/Services/Synthesizer'

def load_acapela_config(csv_path):
    '''Load the ACAPELA config parameters

    The first line of the CSV must contain:
    account_login, application_login, 
    application_password, service_url.
    @param csv_path the path to the config file
    '''
    with open(csv_path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
          conf_account_login = row[0]
          conf_application_login = row[1]
          conf_application_password = row[2]
          conf_service_url = row[3]
          return conf_account_login, conf_application_login, conf_application_password, conf_service_url


def learn_object_from_folder(images_folder, image_extension='jpg'):
    '''Given a folder with images of objects on uniform background
       and different colours fingerprints, it associate the colours
       to the object name.

    The folder must contain images named:
    1_firstobjectname.jpg, 2_secondobjectname.jpg, ...
    When username=None the object is associated
    with an empty object and not used.
    '''
    my_classifier = HistogramColorClassifier(channels=[0, 1, 2], 
                                             hist_size=[128, 128, 128], 
                                             hist_range=[0, 256, 0, 256, 0, 256], 
                                             hist_type='BGR')
    for filename in os.listdir(images_folder):
        if filename.endswith(image_extension): 
            model = cv2.imread(filename)
            my_classifier.addModelHistogram(model)
    return my_classifier




