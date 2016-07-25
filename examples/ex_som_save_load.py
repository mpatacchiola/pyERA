#!/usr/bin/python

## Massimiliano Patacchiola, Plymouth University 2016
#
# This code shows how to store and load a SOM.
#


#Add the pyERA package
import sys
sys.path.insert(0, "../pyERA")

import numpy as np

#It requires the pyERA library
from pyERA.som import Som


def main():


    #Init the SOM
    print("Initializing SOM of size 512x512")
    my_som = Som(matrix_size=512, input_size=3, low=0, high=1, round_values=False)

    #Storing the som in a compressed file
    path = "./"
    name = "my_som"
    print("Saving the network in: " + path + name)
    my_som.save(path, name)

    #Loading the stored file
    print("Loading the network from: " + path + name)
    my_som.load(path + name + ".npz")
 


if __name__ == "__main__":
    main()
