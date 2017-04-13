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

# Implementation of pocketsphinx Speech Recognition based on a grammar.
# It requires a dictionary of world and a grammar file. There are also
# methods for audio recording (based on linux arecord) and file format
# conversion (based on linux sox, lame, oggenc)

from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *
import os

class SpeechRecognizer:
    """
    
    Spynx is based on some standard file that it is necessary to provide:
    
    1- JSpeech Grammar extension (JSF): platform-independent, vendor-independent textual representation of 
    grammars for use in speech recognition. Grammars are used by speech recognizers to determine 
    what the recognizer should listen for, and so describe the utterances a user may say. 
    JSGF adopts the style and conventions of the Java Programming Language in addition to use of 
    traditional grammar notations.
    Example:
        grammar hello;
        public <greet> = (good morning | hello) ( bhiksha | evandro | paul | philip | rita | will );
    http://cmusphinx.sourceforge.net/doc/sphinx4/edu/cmu/sphinx/jsgf/JSGFGrammar.html
    2- A dictionary of allowed words, all the words used in the grammar must be present in this file.
    3- The model for the language, the Pocketsphinx default en-us folder is a good choice.
    """
    def __init__(self, hmm_path, language_model_path, dictionary_path, grammar_path, rule_name, fsg_name):
        """Initiliase a SpeechDetector object. It requires a grammar in order to work.
        
        @param hmm_path: the hidden markov model path
        @param language_model_path: the language model path (.bin)
        @param dictionary_path: the path to the dictionary used (.dic)
        @param grammar_path: path to the grammar file (.gram)
        @param rule_name: the rule to pick up from the grammar file
        @param fsg_name: the fsg name (can be something like: mygrammar)
        """
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

    def record_audio(self, destination_path, seconds=3, extension='ogg', harddev='wav'):
        """Record an audio file for the amount of time specified.

        It requires to install the following packages:
        oggenc: sudo apt-get install vorbis-tools
        lame: sudo apt-get install lame
        @param destination_path: the path were the object is saved
        @param seconds: time in seconds
        @param extension: the extension of the produced file (mp3, ogg, wav)
        @param harddev: to see all the microphones on your laptop type "arecord --list-devices"
            this parameter must be a string containing 'card,device' returned by the command above.
            e.g. card 3: AK5371 [AK5371], device 0: USB Audio [USB Audio]
            For this microphone the harddev parameter must be: '3,0'
        @return: the path to the file created or an empty string in case of errors
        """
        if harddev == '':
            if extension == 'mp3':
                command = "arecord -f cd -d " + str(seconds) + " -t raw | lame -x -r - "  + destination_path
            elif extension == 'ogg':
                command = "arecord -f cd -d " + str(seconds) + " -t raw | oggenc - -r -o " + destination_path
            elif extension == 'wav':
                command = "arecord -f cd -d " + str(seconds) + " " + destination_path
        else:
            if extension == 'mp3':
                command = command = "arecord -f cd -D hw:" + str(harddev) + " -d " + str(seconds) + " -t raw | lame -x -r - "  + destination_path
            elif extension == 'ogg':
                command = "arecord -f cd -D hw:" + str(harddev) + " -d " + str(seconds) + " -t raw | oggenc - -r -o " + destination_path
            elif extension == 'wav':
                command = "arecord -f cd -D hw:" + str(harddev) + " -d " + str(seconds) + " " + destination_path
        try:
            returned = os.system(command)
        except:
            print("Exception when executing arecord command to record audio.")
        if returned == 0:
            return destination_path
        else:
            print("[SPEECH RECOGNITION][ERROR] problem with arecord command, check if extension and harddev are correct.")
            return ''

    def convert_to_raw(self, file_name, file_name_raw="./audio.raw", extension='wav'):
        """ It uses linux 'sox' to convert an mp3 file to raw file.
        
        It is necessary to convert to raw before passing the file to other methods
        @param extension: the extension of the input file (wav, mp3)
        @param file_name: The path to the file to convert
        @param file_name_raw: The path and file name (.raw) for the file produced
        @return: the path to the raw file created
        """
        # Before processing audio must be converted to PCM extension. Recommended extension is 16khz 16bit
        # little-endian mono. If you are decoding telephone quality audio you can also decode 8khz 16bit
        # little-endian mono, but you usually need to reconfigure the decoder to input 8khz audio.
        # For example, pocketsphinx has -samprate 8000 #option in configuration.
        # E.g. use sox to convert mp3 to raw file: sox input.mp3 output.raw rate 16000
        # sox --endian little --bits 16000 member.mp3 member.raw rate 16000 channels 1
        if extension == 'mp3':
            os.system("sox --endian little --bits 16000 " + file_name + " '" + file_name_raw + "' rate 16000 channels 1")
        elif extension == 'wav':
            os.system("sox " + file_name + " " + file_name_raw + " rate 16000 channels 1")
        return file_name_raw

    def return_text_from_audio(self, file_name):
        """Given an audio file in raw extension returns the text.
        
        @param file_name: audio file in .raw extension 
        @return: the text (string) decoded or an empty string if nothing is found
        """
        string_to_return = ""
        self.decoder.start_utt()
        try:
            stream = open(file_name, 'rb')
        except IOError:
            print("[SPEECH RECOGNITION][ERROR] Could not find the audio file :" + str(file_name))
        while True:
            buf = stream.read(1024)
            if buf:
                self.decoder.process_raw(buf, False, False)
            else:
                break
        try:
            self.decoder.end_utt()
            string_to_return = self.decoder.hyp().hypstr
        except:
            print("[SPEECH RECOGNITION][ERROR] The audio file does not respect the grammar rules")
        return string_to_return
