#!/usr/bin/python

import numpy as np

class HebbianConnection:

    def __init__(self, input_shape, output_shape):
        """Initialize the hebbian connections between two networks.

        @param input_shape, it can be a square matrix or a numpy vector. The vector must be initialized as a 1 row (or 1 column) numpy matrix.
        @param output_shape, it can be a square matrix or a numpy vector. The vector must be initialized as a 1 row (or 1 column) numpy matrix.
        """
        if(len(input_shape.shape != 2) or len(output_shape.shape != 2): raise ValueError('hebbian_connection: error the input-outpu matrix shape is != 2')

        self._input_shape = input_shape
        self._output_shape = output_shape

        #The weight matrix is created from the shape of the input/output matrices
        rows = self._input_shape[0] * self._input_shape[1]
        cols = self._output_shape[0] * self._output_shape[1]
        self._weights_matrix = np.zeros((rows * cols))


    def learning_hebb_rule(self, input_activations, output_activations, learning_rate):
        """Single step learning using the Hebbian update rule.

        The standard Hebbian rule: If two neurons on either side of a synapse (connection) are activated simultaneously, 
        then the strength of that synapse is selectively increased.
        @param input_activations a vector or a bidimensional matrix representing the activation of the input units
        @param output_activations a vector or a bidimensional matrix representing the activation of the output units
        @param learning_rate it is costant that defines the learning step
        """
        input_activation = input_activation.flatten()
        output_activation = output_activation.flatten()

        it = np.nditer(self._weights_matrix, flags=['multi_index'])
        while not it.finished:
            #print "%d <%s>" % (it[0], it.multi_index)
            #Applying the Hebbian Rule:
            delta_weight = learning_rate * input_activation[it.multi_index[0]] * output_activation[it.multi_index[1]]
            self._weights_matrix[it.multi_index[0], it.multi_index[1]] += delta_weight
            it.iternext()


    def learning_oja_rule(self, input_activations, output_activations, learning_rate):
        """Single step learning using the Oja's update rule.

        The Oja's rule normalizes the weights between 0 and 1, trying  to stop the weights increasing indefinitely
        @param input_activations a vector or a bidimensional matrix representing the activation of the input units
        @param output_activations a vector or a bidimensional matrix representing the activation of the output units
        @param learning_rate it is costant that defines the learning step
        """
        input_activation = input_activation.flatten()
        output_activation = output_activation.flatten()

        it = np.nditer(self._weights_matrix, flags=['multi_index'])
        while not it.finished:
            #print "%d <%s>" % (it[0], it.multi_index)
            #Applying the Oja's Rule:
            delta_weight = (learning_rate * input_activation[it.multi_index[0]] * output_activation[it.multi_index[1]]) - \
                           (learning_rate * output_activation[it.multi_index[1]] * output_activation[it.multi_index[1]] * self._weights_matrix[it.multi_index[0], it.multi_index[1]] )
            self._weights_matrix[it.multi_index[0], it.multi_index[1]] += delta_weight




