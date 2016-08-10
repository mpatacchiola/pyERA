#!/usr/bin/python

## Massimiliano Patacchiola, Plymouth University 2016
#
# In this example a SOM with 2D input vectors is trained to morph a b/w image. 
# Each pixel of the b/w image is represented with a tuple (x, y) representing its location.
# Random pixels are then sampled from a probability distribution built on the image pixel
# intensity map. Darker pixels have higher probability of being sampled.
# The SOM adapt its weights to describe that random distribution.
#

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#It requires the pyERA library
from pyERA.som import Som
from pyERA.utils import ExponentialDecay


def main():

    #Opening the image in greyscale
    img_size = 512
    img_original = Image.open('./original.jpg')
    img = Image.open('./filtered.jpg').convert("L") #greyscale
    img_matrix = np.asarray(img, dtype=np.float32)

    #Normalising the pixel values to sum up to 1.0
    #The image can be seen as a probability distribution
    #where darker pixels have higher probability to be sampled
    img_prob = (255.0 - img_matrix) / 255.0
    img_prob = img_prob / np.sum(img_prob, dtype=np.float32)

    #This list is a flatten array containing np.array([row,col])
    #which identify the index of the pixel in the image
    index_list = list()
    for row in range(0,img_size):
        for col in range(0,img_size):
            index_list.append(np.array([row,col]))

    #The single index array is an array containing a single
    #identifier for the pixel to take. The function random.choice
    #can take samples from the single_index given a probability
    #distribution. The probability distribution in our case is the
    #normalised image.
    single_index_array = np.arange(0, img_size*img_size)

    #Creating a SOM with weights in the rage [0, img_size]
    #Each weight codifies a position in the original image
    som_size = 128
    batch_size = 64
    my_som = Som(matrix_size=som_size, input_size=2, low=0, high=img_size-1, round_values=True)
    tot_epochs = 5000
    my_learning_rate = ExponentialDecay(starter_value=0.9, decay_step=50, decay_rate=0.9, staircase=True)
    my_radius = ExponentialDecay(starter_value=np.rint(som_size/5), decay_step=80, decay_rate=0.95, staircase=True)

    for epoch in range(0, tot_epochs):

        #Getting a random input
        input_vector_list = list()
        sorted_index_list = np.random.choice(single_index_array, batch_size, p=img_prob.flatten())
        for i in range(0, batch_size): 
            input_vector_list.append(index_list[sorted_index_list[i]])

        #Updating the learning rate and the radius
        learning_rate = my_learning_rate.return_decayed_value(global_step=epoch)
        radius = my_radius.return_decayed_value(global_step=epoch)
        
        if(epoch % 1 == 0):
            #Generate the image from the SOM weights
            som_img = np.full((img_size, img_size, 3), 255, dtype=np.uint8) #np.zeros((img_size, img_size, 3), dtype=np.uint8)
            som_weights_matrix = my_som.return_weights_matrix()
            for row in range(0, som_weights_matrix.shape[0]):
                for col in range(0, som_weights_matrix.shape[1]):
                    x = int(som_weights_matrix[row, col, 0])
                    y = int(som_weights_matrix[row, col, 1])
                    som_img[x, y, 0] = 0
                    som_img[x, y, 1] = 0
                    som_img[x, y, 2] = 0

            #Saving the image
            fig = plt.figure()
            a=fig.add_subplot(1,2,1)
            imgplot = plt.imshow(img_original)
            plt.axis("off")
            b=fig.add_subplot(1,2,2)
            imgplot = plt.imshow(som_img)
            plt.axis("off")
            plt.savefig("./images/" + str(epoch) + ".png", dpi=200, facecolor='black')
            plt.close('all')

        #Learning step (batch learning)
        my_som.training_batch_step(input_vector_list, learning_rate=learning_rate, radius=radius, weighted_distance=True)

        print("")
        print("Epoch: " + str(epoch))
        print("Learning Rate: " + str(learning_rate))
        print("Radius: " + str(radius))
        print("Sorted index: " + str(sorted_index_list))


    #Saving the network
    file_name = "./som_marilyn.npz"
    print("Saving the network in: " + str(file_name))
    my_som.save(path="./", name="some_marilyn")
   

if __name__ == "__main__":
    main()


