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
import subprocess
import os

class SpeechRecognizer:
    """
    
    Spynx is based on some standard file that it is necessary to provide:
    
    1- JSpeech Grammar Format (JSF): platform-independent, vendor-independent textual representation of 
    grammars for use in speech recognition. Grammars are used by speech recognizers to determine 
    what the recognizer should listen for, and so describe the utterances a user may say. 
    JSGF adopts the style and conventions of the Java Programming Language in addition to use of 
    traditional grammar notations.
    Example:
        grammar hello;
        public <greet> = (good morning | hello) ( bhiksha | evandro | paul | philip | rita | will );
    http://cmusphinx.sourceforge.net/doc/sphinx4/edu/cmu/sphinx/jsgf/JSGFGrammar.html
    
    """
    def __init__(self, hmm_path, language_model_path, dictionary_path, grammar_path, rule_name, fsg_name):
        # Create a decoder with certain model
        config = Decoder.default_config()
        config.set_string('-hmm', hmm_path)
        #config.set_string('-lm', path.join(data_path, 'turtle.lm.bin')) #language model
        config.set_string('-lm',  language_model_path)
        config.set_string('-dict', dictionary_path) #dictionary
        self.decoder = Decoder(config)
        # Switch to JSGF grammar
        jsgf = Jsgf(grammar_path)
        rule = jsgf.get_rule(rule_name)
        fsg = jsgf.build_fsg(rule, self.decoder.get_logmath(), 7.5)
        fsg.writefile(fsg_name + '.fsg')

        self.decoder.set_fsg(fsg_name, fsg)
        self.decoder.set_search(fsg_name)

    def convert_to_raw(self, file_name, file_name_raw="./audio.raw"):
        """ It uses linux 'sox' to convert an mp3 file to .raw file.
        
        It is necessary to convert to raw before passing the file to other methods
        @param file_name: The path to the file to convert
        @param file_name_raw: The path and file name (.raw) for the file produced
        @return: the path to the raw file created
        """
        # Before processing audio must be converted to PCM format. Recommended format is 16khz 16bit
        # little-endian mono. If you are decoding telephone quality audio you can also decode 8khz 16bit
        # little-endian mono, but you usually need to reconfigure the decoder to input 8khz audio.
        # For example, pocketsphinx has -samprate 8000 #option in configuration.
        # E.g. use sox to convert mp3 to raw file: sox input.mp3 output.raw rate 16000
        # sox --endian little --bits 16000 member.mp3 member.raw rate 16000 channels 1
        os.system('sox --endian little --bits 16000 ' + file_name + ' ' + file_name_raw + ' rate 16000 channels 1')
        return file_name_raw

    def return_text_from_audio(self, file_name):
        """Given an audio file in .raw format returns the text.
        
        @param file_name: audio file in .raw format 
        @return: the text (string) 
        """
        self.decoder.start_utt()
        stream = open(file_name, 'rb')
        while True:
            buf = stream.read(1024)
            if buf:
                self.decoder.process_raw(buf, False, False)
            else:
                break
        self.decoder.end_utt()
        return self.decoder.hyp().hypstr


def main():
    my_speech= SpeechRecognizer(hmm_path="/home/massimiliano/pyERA/examples/ex_icub_trust_cognitive_architecture/sphinx/model/en-us/en-us",
                                language_model_path="/home/massimiliano/pyERA/examples/ex_icub_trust_cognitive_architecture/sphinx/model/en-us/en-us.lm.bin",
                                dictionary_path="/home/massimiliano/pyERA/examples/ex_icub_trust_cognitive_architecture/sphinx/data/icub.dic",
                                grammar_path="/home/massimiliano/pyERA/examples/ex_icub_trust_cognitive_architecture/sphinx/data/icub.gram",
                                rule_name='icub.basicCmd',
                                fsg_name="icub")
    #text = my_speech.return_text_from_audio("/home/massimiliano/pyERA/examples/ex_icub_trust_cognitive_architecture/sphinx/data/goforward.raw")
    raw_file_path = my_speech.convert_to_raw(file_name="/home/massimiliano/pyERA/examples/ex_icub_trust_cognitive_architecture/sphinx/data/icub.mp3")
    text = my_speech.return_text_from_audio(raw_file_path)
    print(text)
if __name__ == "__main__":
    main()
