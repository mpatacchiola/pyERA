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

# this is the FIRST PHASE of the Koenig-Harris experiment.
# The child is presented with 3 known objects and two informants.
# The informants say the name of the objects. One is alway reliable,
# the other always unreliable.

# This file store the learning matrix for a single informant. It should be
# used with both of them.

import sys


def main():
    #print 'Number of arguments:', len(sys.argv)
    #print 'Argument List:', str(sys.argv)
    inputfile = ''
    informant_name = ''
    if len(sys.argv) == 1 or len(sys.argv) >3:
        print("python familiarization.py <inputfile> <informant_name>")
    if len(sys.argv) == 2:
        inputfile = sys.argv[1]
    elif len(sys.argv) == 3:
        inputfile = sys.argv[1]
        informant_name = sys.argv[2]

    print("Input file: " + str(inputfile))
    print("Informant Name: " + str(informant_name))



if __name__ == "__main__":
    main()

