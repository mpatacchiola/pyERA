#!/usr/bin/python

import numpy as np


class Som:
    """Som Class

    This is an implementation of the Self-Organizing Map (SOM).
    It provides low level funcion and utilities for assembling
    diffeent type of som. It uses only the numpy library.
    """

    def __init__(self, matrix_size, input_size, low=0, high=1, round_values=False):
        """Init function.

        @param matrix_size It defines the matrix size 
        @param input_size it defines the vector input size.
        @param low boundary for the random initialization 
        @param high boundary for the random initialization
        @param round_values it is possible to initialize the 
        weights to the closest integer value.
        """
        self._matrix_size = matrix_size
        self._input_size = input_size
        self._weights_matrix = np.random.uniform(low=low, high=high, size=(matrix_size, matrix_size, input_size))

        if (round_values == True):
            self._weights_matrix = np.rint(self._weights_matrix)

    def return_weights_matrix(self):
        return self._weights_matrix

    def get_unit_weights(self, row, col):
        """Return the weights associated with a given unit.

        """
        return self._weights_matrix[row, col, :]


    def set_unit_weights(self, weights_vector, row, col):
        self._weights_matrix[row, col, :] = weights_vector


    def return_unit_square_neighborhood(self, row, col, ray):
         output_list = list()
         if(ray <= 0): output_list.append((row, col, 0)); return output_list #return empty if ray=0
         ray = int(ray) #Precaution against float ray
         row_range_min = row - ray
         if(row_range_min < 0): row_range_min = 0
         row_range_max = row + ray
         if(row_range_max >= self._matrix_size): row_range_max = self._matrix_size - 1

         col_range_min = col - ray
         if(col_range_min < 0): col_range_min = 0
         col_range_max = col + ray
         if(col_range_max >= self._matrix_size): col_range_max = self._matrix_size - 1

         for row_iter in range(row_range_min, row_range_max+1):
             for col_iter in range(col_range_min, col_range_max+1):
                 #Finding the distances from the BMU
                 col_distance = np.abs(col - col_iter)
                 row_distance = np.abs(row - row_iter)
                 if(col_distance >= row_distance): distance = col_distance
                 else: distance = row_distance
                 #Storing (row, col, distance)
                 output_list.append((row_iter, col_iter, distance))

         return output_list

    def return_unit_round_neighborhood(self, row, col, ray):
         output_list = list()
         if(ray <= 0): output_list.append((row, col, 0)); return output_list #return empty if ray=0

         #Finding the square around the unit
         #with wide=ray using the ceil of ray
         row_range_min = row - int(np.ceil(ray))
         if(row_range_min < 0): row_range_min = 0
         row_range_max = row + int(np.ceil(ray))
         if(row_range_max >= self._matrix_size): row_range_max = self._matrix_size - 1
         col_range_min = col - int(np.ceil(ray))
         if(col_range_min < 0): col_range_min = 0
         col_range_max = col + int(np.ceil(ray))
         if(col_range_max >= self._matrix_size): col_range_max = self._matrix_size - 1

         for row_iter in range(row_range_min, row_range_max+1):
             for col_iter in range(col_range_min, col_range_max+1):
                 #Finding the distances from the BMU
                 col_distance = np.abs(col - col_iter)
                 row_distance = np.abs(row - row_iter)
                 #Pitagora's Theorem to estimate distance
                 distance = np.sqrt( np.power(col_distance,2) + np.power(row_distance,2) )
                 #Store the unit only if the distance is
                 #less than the ray
                 if(distance <= ray): output_list.append((row_iter, col_iter, distance))

         return output_list

    def return_euclidean_distance(self, a, b):
        return np.linalg.norm(a-b)


    def return_BMU_index(self, input_vector):
        output_matrix = np.zeros((self._matrix_size, self._matrix_size))
        it = np.nditer(output_matrix, flags=['multi_index'])
        while not it.finished:
            #print "%d <%s>" % (it[0], it.multi_index),
            dist = self.return_euclidean_distance(input_vector, self._weights_matrix[it.multi_index[0], it.multi_index[1], :])
            output_matrix[it.multi_index[0], it.multi_index[1]] = dist
            it.iternext()
        row, col = np.unravel_index(output_matrix.argmin(), output_matrix.shape)
        return (row, col)


    def return_BMU_weights(self, input_vector):
        output_matrix = np.zeros((self._matrix_size, self._matrix_size))
        it = np.nditer(output_matrix, flags=['multi_index'])
        while not it.finished:
            #print "%d <%s>" % (it[0], it.multi_index),
            dist = self.return_euclidean_distance(input_vector, self._weights_matrix[it.multi_index[0], it.multi_index[1], :])
            output_matrix[it.multi_index[0], it.multi_index[1]] = dist
            it.iternext()
        row, col = np.unravel_index(output_matrix.argmin(), output_matrix.shape)
        return self._weights_matrix[row, col, :]

    def training_single_step(self, input_vector, units_list, learning_rate, ray):
        for unit in units_list:
            row = unit[0]
            col = unit[1]
            dist = unit[2]

            #The distance_rate take into account the distance of the unit
            #from the BMU and permits regulating the updating of the weights
            #with more decision for units that are close to the BMU.
            #distance_rate = np.exp(- np.power(dist,2)/(2*np.power(ray,2)))

            #Update the weights of the neighborood units
            #The new weight is equal to the old one plus a fracion
            # of the difference between the input_vector and the
            # unit weights
            #self._weights_matrix[row, col, :] = self._weights_matrix[row, col, :] + distance_rate * learning_rate * (input_vector - self._weights_matrix[row, col, :])
            self._weights_matrix[row, col, :] = self._weights_matrix[row, col, :] + learning_rate * (input_vector - self._weights_matrix[row, col, :])








