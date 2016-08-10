#!/usr/bin/python

## Massimiliano Patacchiola, Plymouth University 2016
#
# It requires the file "examples/som_colours.npz" generated from the example "ex_som_colours"
# In this example a pre-trained SOM is used to show how the weights are related to an input.
# Given 6 random samples of the 6 random colours it is generated a normalised image of the 
# distances between the input and the weights.
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#It requires the pyERA library
from pyERA.som import Som

def main():

    #Loading the pretrained SOM
    som_size = 512
    my_som = Som(matrix_size=som_size, input_size=3, low=0, high=1, round_values=False)
    my_som.load("./some_colours_512_2.npz")
    print("")

    #Saving the image of the SOM weights
    print("ORIGINAL IMAGE")
    img = np.rint(my_som.return_weights_matrix()*255)
    plt.axis("off")
    plt.imshow(img)
    print("Saving in: ./original.png")
    plt.savefig("./original.png", dpi=None, facecolor='black')

    #Variables
    draw_range = 5 #The colour range of the re square
    colour = 0.6 #The intensity of the byte in the array
    img = np.zeros((som_size,som_size,3), 'uint8')

    #Saving the activation units for RED
    print("")
    som_input_vector = np.array([colour, 0, 0]) #RED
    print("INPUT RED: " + str(som_input_vector*255))
    #For each color an image of the differences between input/weights is generated.
    #To have a normalised matrix of weights it is possible to use the function: return_similarity_matrix
    #The weights are then returned in a normalized range [0,1] and multiplied times 255 to have valid pixels.
    som_similarity_matrix = my_som.return_similarity_matrix(som_input_vector) * 255
    img[:,:,0] = som_similarity_matrix
    img[:,:,1] = som_similarity_matrix
    img[:,:,2] = som_similarity_matrix
    bmu_index = my_som.return_BMU_index(som_input_vector)
    #In the black and withe image a RED spot is drawn on the BMU.
    img[bmu_index[0]-draw_range:bmu_index[0]+draw_range, bmu_index[1]-draw_range:bmu_index[1]+draw_range, 0] = 200
    img[bmu_index[0]-draw_range:bmu_index[0]+draw_range, bmu_index[1]-draw_range:bmu_index[1]+draw_range, 1] = 0
    img[bmu_index[0]-draw_range:bmu_index[0]+draw_range, bmu_index[1]-draw_range:bmu_index[1]+draw_range, 2] = 0
    #Create a matplotlib graph and save the image
    plt.axis("off")
    plt.imshow(img)
    print("Saving in: ./red.png")
    plt.savefig("./red.png", dpi=None, facecolor='black')

    #Saving the activation units for GREEN
    print("")
    som_input_vector = np.array([0, colour, 0]) #GREEN
    print("INPUT GREEN: " + str(som_input_vector*255))
    som_similarity_matrix = my_som.return_similarity_matrix(som_input_vector) * 255
    img[:,:,0] = som_similarity_matrix
    img[:,:,1] = som_similarity_matrix
    img[:,:,2] = som_similarity_matrix
    bmu_index = my_som.return_BMU_index(som_input_vector)
    img[bmu_index[0]-draw_range:bmu_index[0]+draw_range, bmu_index[1]-draw_range:bmu_index[1]+draw_range, 0] = 200
    img[bmu_index[0]-draw_range:bmu_index[0]+draw_range, bmu_index[1]-draw_range:bmu_index[1]+draw_range, 1] = 0
    img[bmu_index[0]-draw_range:bmu_index[0]+draw_range, bmu_index[1]-draw_range:bmu_index[1]+draw_range, 2] = 0
    plt.axis("off")
    plt.imshow(img)
    print("Saving in: ./gree.png")
    plt.savefig("./green.png", dpi=None, facecolor='black')

    #Saving the activation units for BLUE
    print("")
    som_input_vector = np.array([0, 0, colour]) #BLUE
    print("INPUT BLUE: " + str(som_input_vector*255))
    som_similarity_matrix = my_som.return_similarity_matrix(som_input_vector) * 255
    img[:,:,0] = som_similarity_matrix
    img[:,:,1] = som_similarity_matrix
    img[:,:,2] = som_similarity_matrix
    bmu_index = my_som.return_BMU_index(som_input_vector)
    img[bmu_index[0]-draw_range:bmu_index[0]+draw_range, bmu_index[1]-draw_range:bmu_index[1]+draw_range, 0] = 200
    img[bmu_index[0]-draw_range:bmu_index[0]+draw_range, bmu_index[1]-draw_range:bmu_index[1]+draw_range, 1] = 0
    img[bmu_index[0]-draw_range:bmu_index[0]+draw_range, bmu_index[1]-draw_range:bmu_index[1]+draw_range, 2] = 0
    plt.axis("off")
    plt.imshow(img)
    print("Saving in: ./blue.png")
    plt.savefig("./blue.png", dpi=None, facecolor='black')

    #Saving the activation units for YELLOW
    print("")
    som_input_vector = np.array([colour,colour, 0]) #YELLOW
    print("INPUT YELLOW: " + str(som_input_vector*255))
    som_similarity_matrix = my_som.return_similarity_matrix(som_input_vector) * 255
    img[:,:,0] = som_similarity_matrix
    img[:,:,1] = som_similarity_matrix
    img[:,:,2] = som_similarity_matrix
    bmu_index = my_som.return_BMU_index(som_input_vector)
    img[bmu_index[0]-draw_range:bmu_index[0]+draw_range, bmu_index[1]-draw_range:bmu_index[1]+draw_range, 0] = 200
    img[bmu_index[0]-draw_range:bmu_index[0]+draw_range, bmu_index[1]-draw_range:bmu_index[1]+draw_range, 1] = 0
    img[bmu_index[0]-draw_range:bmu_index[0]+draw_range, bmu_index[1]-draw_range:bmu_index[1]+draw_range, 2] = 0
    plt.axis("off")
    plt.imshow(img)
    print("Saving in: ./yellow.png")
    plt.savefig("./yellow.png", dpi=None, facecolor='black')

    #Saving the activation units for LIGHT-BLUE
    print("")
    som_input_vector = np.array([0, colour, colour]) #LIGHT-BLUE
    print("INPUT LIGHT-BLUE: " + str(som_input_vector*255))
    som_similarity_matrix = my_som.return_similarity_matrix(som_input_vector) * 255
    img[:,:,0] = som_similarity_matrix
    img[:,:,1] = som_similarity_matrix
    img[:,:,2] = som_similarity_matrix
    bmu_index = my_som.return_BMU_index(som_input_vector)
    img[bmu_index[0]-draw_range:bmu_index[0]+draw_range, bmu_index[1]-draw_range:bmu_index[1]+draw_range, 0] = 200
    img[bmu_index[0]-draw_range:bmu_index[0]+draw_range, bmu_index[1]-draw_range:bmu_index[1]+draw_range, 1] = 0
    img[bmu_index[0]-draw_range:bmu_index[0]+draw_range, bmu_index[1]-draw_range:bmu_index[1]+draw_range, 2] = 0
    plt.axis("off")
    plt.imshow(img)
    print("Saving in: ./light-blue.png")
    plt.savefig("./light-blue.png", dpi=None, facecolor='black')

    #Saving the activation units for PURPLE
    print("")
    som_input_vector = np.array([colour, 0, colour]) #PURPLE
    print("INPUT PURPLE: " + str(som_input_vector*255))
    som_similarity_matrix = my_som.return_similarity_matrix(som_input_vector) * 255
    img[:,:,0] = som_similarity_matrix
    img[:,:,1] = som_similarity_matrix
    img[:,:,2] = som_similarity_matrix
    bmu_index = my_som.return_BMU_index(som_input_vector)
    img[bmu_index[0]-draw_range:bmu_index[0]+draw_range, bmu_index[1]-draw_range:bmu_index[1]+draw_range, 0] = 200
    img[bmu_index[0]-draw_range:bmu_index[0]+draw_range, bmu_index[1]-draw_range:bmu_index[1]+draw_range, 1] = 0
    img[bmu_index[0]-draw_range:bmu_index[0]+draw_range, bmu_index[1]-draw_range:bmu_index[1]+draw_range, 2] = 0
    plt.axis("off")
    plt.imshow(img)
    print("Saving in: ./purple.png")
    plt.savefig("./purple.png", dpi=None, facecolor='black')


if __name__ == "__main__":
    main()
