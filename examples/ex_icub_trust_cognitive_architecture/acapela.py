#!/usr/bin/python
# -*- coding: utf-8 -*-

# acapela.py - Python wrapper for text-to-speech synthesis with Acapela
# Copyright (C) 2012 Arezqui Belaid <areski@gmail.com>
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import os.path
from optparse import OptionParser

if sys.version_info < (3, 0):
    import urllib as request
    parse = request
    import urllib2
    for method in dir(urllib2):
        setattr(request, method, getattr(urllib2, method))
    import cookielib as cookiejar
else:
    from http import cookiejar
    from urllib import parse, request

# Version Python-Acapela
__version__ = '0.2.6'

ACCOUNT_LOGIN = 'EVAL_XXXX'
APPLICATION_LOGIN = 'EVAL_XXXXXXX'
APPLICATION_PASSWORD = 'XXXXXXXX'

SERVICE_URL = 'http://vaas.acapela-group.com/Services/Synthesizer'
LANGUAGE = 'EN'
QUALITY = '22k'  # 22k, 8k, 8ka, 8kmu
DIRECTORY = '/tmp/'

USAGE = \
    """\nUsage: acapela.py -a <accountlogin> -n <applicationlogin> -p <password> -t <text> [-l <language>] [-q <quality>] [-d <directory>] [-url <service_url>] [-h]"""


def validate_options(accountlogin, applicationlogin, password, text):
    """Perform sanity checks on threshold values"""

    if not accountlogin or len(accountlogin) == 0:
        print 'Error: Warning the option accountlogin should contain a string.'
        print USAGE
        sys.exit(3)

    if not applicationlogin or len(applicationlogin) == 0:
        print 'Error: Warning the option applicationlogin should contain a string.'
        print USAGE
        sys.exit(3)

    if not password or len(password) == 0:
        print 'Error: Warning the option password should contain a string.'
        print USAGE
        sys.exit(3)

    if not text or len(text) == 0:
        print 'Error: Warning the option text should contain a string.'
        print USAGE
        sys.exit(3)


class Acapela(object):
    # Properties
    TTS_ENGINE = None
    ACCOUNT_LOGIN = None
    APPLICATION_LOGIN = None
    APPLICATION_PASSWORD = None
    SERVICE_URL = None
    QUALITY = None
    DIRECTORY = ''

    # Available voices list
    # http://www.acapela-vaas.com/ReleasedDocumentation/voices_list.php

    langs = {
        'EN': {'W': {'NORMAL': 'lucy'}, 'M': {'NORMAL': 'peter'}},
        'US': {'W': {'NORMAL': 'nelly'}, 'M': {'NORMAL': 'kenny'}},
        'ES': {'W': {'NORMAL': 'ines'}, 'M': {'NORMAL': 'antonio'}},
        'FR': {'W': {'NORMAL': 'alice'}, 'M': {'NORMAL': 'antoine'}},
        'PT': {'W': {'NORMAL': 'celia'}},
        'BR': {'W': {'NORMAL': 'marcia'}},
    }

    data = {}
    filename = None
    cache = True

    def __init__(self, account_login, application_login, application_password, service_url, quality, directory=''):
        """construct Acapela TTS"""
        self.TTS_ENGINE = 'ACAPELA'
        self.ACCOUNT_LOGIN = account_login
        self.APPLICATION_LOGIN = application_login
        self.APPLICATION_PASSWORD = application_password
        self.SERVICE_URL = service_url
        self.QUALITY = quality
        self.DIRECTORY = directory

    def prepare(self, text, lang, gender, intonation):
        """Prepare Acapela TTS"""
        lang = lang.upper()
        concatkey = '%s-%s-%s-%s' % (text, lang, gender, intonation)
        key = self.TTS_ENGINE + '' + str(hash(concatkey))
        try:
            req_voice = self.langs[lang][gender][intonation] \
                + self.QUALITY
        except:
            req_voice = 'lucy22k'

        self.data = {
            'cl_env': 'PYTHON_2.X',
            'req_snd_id': key,
            'cl_login': self.ACCOUNT_LOGIN,
            'cl_vers': '1-30',
            # req_err_as_id3 is depreciated
            # 'req_err_as_id3': 'yes',
            'req_voice': req_voice,
            'cl_app': self.APPLICATION_LOGIN,
            'prot_vers': '2',
            'cl_pwd': self.APPLICATION_PASSWORD,
            'req_asw_type': 'STREAM',
        }
        self.filename = '%s-%s.mp3' % (key, lang)
        self.data['req_text'] = '\\vct=100\\ \\spd=160\\ %s' \
            % text.encode('utf-8')

    def set_cache(self, value=True):
        """
        Enable Cache of file, if files already stored return this filename
        """

        self.cache = value

    def run(self):
        """run will call acapela API and and produce audio"""

        # check if file exists

        if self.cache and os.path.isfile(self.DIRECTORY
                + self.filename):
            return self.filename
        else:
            encdata = parse.urlencode(self.data)
            request.urlretrieve(self.SERVICE_URL, self.DIRECTORY
                                + self.filename, data=encdata)
            return self.filename


def _main():
    """
    Parse options and process text to Acapela
    """

    # Parse arguments
    parser = OptionParser()
    parser.add_option('-a', '--acclogin', dest='acclogin',
                      help='accountlogin for authentication')
    parser.add_option('-n', '--applogin', dest='applogin',
                      help='applicationlogin for authentication')
    parser.add_option('-p', '--password', dest='password',
                      help='Password for authentication')
    parser.add_option('-t', '--text', dest='text',
                      help='text to synthesize')
    parser.add_option('-l', '--language', dest='language',
                      help='language')
    parser.add_option('-q', '--quality', dest='quality',
                      help='quality of synthesizer (22k, 8k, 8ka, 8kmu)'
                      )
    parser.add_option('-d', '--directory', dest='directory',
                      help='directory to store the file')
    parser.add_option('-u', '--url', dest='url', help='web service url')
    (options, args) = parser.parse_args()
    acclogin = options.acclogin
    applogin = options.applogin
    password = options.password
    text = options.text
    language = options.language
    quality = options.quality
    directory = options.directory
    url = options.url

    # Perform sanity checks on options
    validate_options(acclogin, applogin, password, text)

    if not quality:
        quality = QUALITY

    if not directory:
        directory = DIRECTORY

    if not url:
        url = SERVICE_URL

    if not language:
        language = LANGUAGE

    tts_acapela = Acapela(acclogin, applogin, password, url, quality, directory)
    gender = 'W'
    intonation = 'NORMAL'
    tts_acapela.set_cache(False)
    tts_acapela.prepare(text, language, gender, intonation)
    output_filename = tts_acapela.run()

    print 'Recorded TTS to %s%s' % (directory, output_filename)


if __name__ == '__main__':
    _main()
