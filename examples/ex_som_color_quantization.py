#!/usr/bin/python

## Massimiliano Patacchiola, Plymouth University 2016
#
# Color quantization reduces the number of colors used in an image; 
# this is important for displaying images on devices that support a limited number of colors 
# and for efficiently compressing certain kinds of images [https://en.wikipedia.org/wiki/Quantization_(image_processing)]
#
# In this example an image is passed as input. A batch of pixel is sampled and given as dataset to
# a Self-Organizing Map (SOM). The SOM will find the best colour vectors representing the image.
# The idea is to describe the same image with a lower number of colors. If the SOM has size 4
# then the resulting number of pixel used will be 4*4 = 16. In comparison the total possible 
# combination of colours in the RGB format is 255*255*255 = 16,581,375
# To get the total number of colors in an image it is possible to use the command:
# identify -verbose -unique image_name.jpg

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os

#It requires the pyERA library
from pyERA.som import Som
from pyERA.utils import ExponentialDecay
from pyERA.utils import LinearDecay

def main():

    #Variables to play with
    som_size = 4
    batch_size = 32
    tot_epoch = 200
    image_name = "chameleon.jpg" #the file must be in the "./input" folder

    #Set to True if you want to save the SOM images inside a folder.
    SAVE_IMAGE = True
    output_path = "./output/" #Change this path to save in a different forlder
    if not os.path.exists(output_path):
        os.makedirs(output_path) 
    img_original = Image.open("./input/" + image_name)
    img_input_matrix = np.asarray(img_original, dtype=np.float32)
    img_rows = img_input_matrix.shape[0]
    img_cols = img_input_matrix.shape[1]

    #Init the SOM
    my_som = Som(matrix_size=som_size, input_size=3, low=0, high=255, round_values=True)
    
    #Init the parameters
    my_learning_rate = ExponentialDecay(starter_value=0.5, decay_step=10, decay_rate=0.8, staircase=True)
    my_radius = ExponentialDecay(starter_value=np.rint(2.0), decay_step=10, decay_rate=0.5, staircase=True)
    #my_radius = LinearDecay(starter_value=30, decay_rate=0.02, allow_negative=False)

    #Starting the Learning
    for epoch in range(1, tot_epoch):

        #Iterates the elements in img_output_matrix and
        #assign the closest value contained in SOM
        img_output_matrix = np.zeros((img_rows, img_cols, 3))

        #Iterates through the original image and find the BMU for
        #each single pixel.
        for row in range (0, img_rows):
            for col in range(0, img_cols):
                input_vector =  np.array(img_input_matrix[row, col, :])
                bmu_index = my_som.return_BMU_index(input_vector)
                bmu_weights = my_som.get_unit_weights(bmu_index[0], bmu_index[1])
                img_output_matrix[row, col, :] = (bmu_weights - 255) * -1 #renormalise to show the right colours           

        #Saving the image associated with the SOM weights
        if(SAVE_IMAGE == True):
            #Saving the image
            fig = plt.figure()
            a=fig.add_subplot(1,2,1)
            imgplot = plt.imshow(img_original)
            plt.axis("off")
            b=fig.add_subplot(1,2,2)
            img_output = np.rint(img_output_matrix)
            imgplot = plt.imshow(img_output)
            plt.axis("off")
            plt.savefig(output_path + str(epoch) + ".png", dpi=200, facecolor='black')
            plt.close('all')
            
        #Updating learning rate and radius
        learning_rate = my_learning_rate.return_decayed_value(global_step=epoch)
        radius = my_radius.return_decayed_value(global_step=epoch)

        #Generating input vectors from random sampling
        input_vector_list = list()
        for i in range(0, batch_size):
            row_index = np.random.randint(img_rows)
            col_index = np.random.randint(img_cols)
            input_vector =  np.array(img_input_matrix[row_index, col_index, :])
            input_vector_list.append(input_vector)

        #Learning step (batch learning)
        my_som.training_batch_step(input_vector_list, learning_rate=learning_rate, radius=radius, weighted_distance=True) 

        #Learning step      
        #my_som.training_single_step(input_vector, units_list=bmu_neighborhood_list, learning_rate=learning_rate, radius=radius, weighted_distance=False)
        my_som.training_batch_step(input_vector_list, learning_rate=learning_rate, radius=radius, weighted_distance=True)

        print("")
        print("Epoch: " + str(epoch))
        print("Learning Rate: " + str(learning_rate))
        print("Radius: " + str(radius))
        print("Input vector: " + str(input_vector))
        print("BMU index: " + str(bmu_index))
        print("BMU weights: " + str(bmu_weights))

    #Saving the network
    file_name = output_path + "som_color_quantization.npz"
    print("Saving the network in: " + str(file_name))
    my_som.save(path=output_path, name="som_color_quantization")

    #Saving the final image
    img_output = ((img_output_matrix- 255) * -1 ).astype(np.uint8)
    img_to_save = Image.fromarray(img_output, "RGB")
    img_to_save.save(output_path + image_name)

    #Showing the SOM weights
    #img = np.rint(my_som.return_weights_matrix())
    #plt.axis("off")
    #plt.imshow(img)
    #plt.show()

if __name__ == "__main__":
    main()
