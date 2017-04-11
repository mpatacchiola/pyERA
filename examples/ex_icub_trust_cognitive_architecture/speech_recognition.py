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

from os import environ, path
from sys import stdout
from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *


class SpeechRecognizer:
    def __init__(self, data_path='./sphinx/data', model_path='./sphinx/model'):
        self.data_path = data_path
        # Create a decoder with certain model
        config = Decoder.default_config()
        config.set_string('-hmm', path.join(model_path, 'en-us/en-us'))
        config.set_string('-lm', path.join(data_path, 'turtle.lm.bin'))
        config.set_string('-dict', path.join(data_path, 'turtle.dic'))
        self.decoder = Decoder(config)
        # Switch to JSGF grammar
        jsgf = Jsgf(path.join(data_path, 'goforward.gram'))
        rule = jsgf.get_rule('goforward.move2')
        fsg = jsgf.build_fsg(rule, self.decoder.get_logmath(), 7.5)
        fsg.writefile('goforward.fsg')

    def return_text_from_audio(self, file_name):
        """Given an audio file in .raw format returns the text.
        
        @param file_name: audio file in .raw format 
        @return: the text (string) 
        """
        self.decoder.start_utt()
        stream = open(path.join(self.data_path, file_name), 'rb')
        while True:
            buf = stream.read(1024)
            if buf:
                self.decoder.process_raw(buf, False, False)
            else:
                break
        self.decoder.end_utt()
        return self.decoder.hyp().hypstr
