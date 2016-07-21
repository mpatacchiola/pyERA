#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import pyERA.som
from pyERA.som import Som
from pyERA.utils import ExponentialDecay

def main():

    #Init the SOM
    som_size = 64
    my_som = Som(matrix_size=som_size, input_size=3, low=0, high=1, round_values=False)

    img = np.rint(my_som.return_weights_matrix()*255)
    plt.axis("off")
    plt.imshow(img)
    plt.show()

    #Init the parameters
    tot_epoch = 1000
    my_learning_rate = ExponentialDecay(starter_value=0.9, decay_step=50, decay_rate=0.9, staircase=True)
    my_ray = ExponentialDecay(starter_value=20, decay_step=100, decay_rate=0.7, staircase=True)

    #Starting the Learning
    for epoch in range(1, tot_epoch):

        #Updating the learning rate and the ray
        learning_rate = my_learning_rate.return_decayed_value(global_step=epoch)
        ray = my_ray.return_decayed_value(global_step=epoch)

        #Generating random input vectors
        colour_selected = np.random.randint(0, 3)
        colour_range = np.random.randint(80, 255)
        colour_range = float(colour_range) / 255.0
        #colour_range = np.random.random_sample()
        #colour_range = 1
        if(colour_selected == 0): input_vector = np.array([colour_range, 0, 0])
        if(colour_selected == 1): input_vector = np.array([0, colour_range, 0])
        if(colour_selected == 2): input_vector = np.array([0, 0, colour_range])

        #Estimating the BMU coordinates
        bmu_index = my_som.return_BMU_index(input_vector)
        bmu_weights = my_som.get_unit_weights(bmu_index[0], bmu_index[1])
        bmu_neighborhood_list = my_som.return_unit_square_neighborhood(bmu_index[0], bmu_index[1], ray=ray)  

        #Learning step      
        my_som.training_single_step(input_vector, units_list=bmu_neighborhood_list, learning_rate=learning_rate, ray=ray)

        print("")
        print("Epoch: " + str(epoch))
        print("Learning Rate: " + str(learning_rate))
        print("Ray: " + str(ray))
        print("Input vector: " + str(input_vector*255))
        print("BMU index: " + str(bmu_index))
        print("BMU weights: " + str(bmu_weights*255))
        #print("BMU NEIGHBORHOOD: " + str(bmu_neighborhood_list))

    #Taking the weight matrix and showing
    #it like a picture. Each element inside
    #the matrix is an RBG array representing
    # a colour.
    img = np.rint(my_som.return_weights_matrix()*255)
    plt.axis("off")
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    main()
