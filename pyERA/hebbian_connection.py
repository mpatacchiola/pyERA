#!/usr/bin/python

import numpy as np

class HebbianConnection:

    def __init__(self, input_shape, output_shape):
        """Initialize the hebbian connections between two networks.

        The weights of the connection are adjusted using a learning rule.
        @param input_shape, it can be a square matrix or a numpy vector. The vector must be initialized as a 1 row (or 1 column) numpy matrix.
        @param output_shape, it can be a square matrix or a numpy vector. The vector must be initialized as a 1 row (or 1 column) numpy matrix.
        """
        if(len(input_shape) != 2 or len(output_shape) != 2): raise ValueError('hebbian_connection: error the input-outpu matrix shape is != 2')

        self._input_shape = input_shape
        self._output_shape = output_shape

        #The weight matrix is created from the shape of the input/output matrices
        rows = self._input_shape[0] * self._input_shape[1]
        cols = self._output_shape[0] * self._output_shape[1]
        self._weights_matrix = np.zeros((rows, cols))


    def learning_hebb_rule(self, input_activation_matrix, output_activation_matrix, learning_rate):
        """Single step learning using the Hebbian update rule.

        The standard Hebbian rule: If two neurons on either side of a synapse (connection) are activated simultaneously, 
        then the strength of that synapse is selectively increased.
        @param input_activations a vector or a bidimensional matrix representing the activation of the input units
        @param output_activations a vector or a bidimensional matrix representing the activation of the output units
        @param learning_rate (positive) it is costant that defines the learning step
        """
        if(learning_rate <=0): raise ValueError('hebbian_connection: error the learning rate used for the hebbian rule must be >0')

        input_activation_vector = input_activation_matrix.flatten()
        output_activation_vector = output_activation_matrix.flatten()

        it = np.nditer(self._weights_matrix, flags=['multi_index'])
        while not it.finished:
            #print "%d <%s>" % (it[0], it.multi_index)
            #Applying the Hebbian Rule:
            delta_weight = learning_rate * input_activation_vector[it.multi_index[0]] * output_activation_vector[it.multi_index[1]]
            self._weights_matrix[it.multi_index[0], it.multi_index[1]] += delta_weight
            it.iternext()


    def learning_anti_hebb_rule(self, input_activations, output_activations, learning_rate):
        """Single step learning using the Anti-Hebbian update rule.

        The Anti-Hebbian rule: If two neurons on either side of a synapse (connection) are activated simultaneously, 
        then the strength of that synapse is selectively decreased.
        @param input_activations a vector or a bidimensional matrix representing the activation of the input units
        @param output_activations a vector or a bidimensional matrix representing the activation of the output units
        @param learning_rate (negative) it is costant that defines the decreasing step
        """
        if(learning_rate >=0): raise ValueError('hebbian_connection: error the learning rate used for the anti-hebbian rule must be <0')

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
            it.iternext()

    def return_output_activation_matrix(self, input_activation_matrix):
        """It returns the activation matrix of the output network

        @param input_activation_matrix a vector or a bidimensional matrix representing the activation of the input units
        """
        input_activation_vector = input_activation_matrix.flatten()
        output_activation_matrix = np.zeros(self._output_shape)
        output_activation_vector = output_activation_matrix.flatten()

        it = np.nditer(self._weights_matrix, flags=['multi_index'])
        while not it.finished:
            output_activation_vector[it.multi_index[1]] +=  input_activation_vector[it.multi_index[0]] * self._weights_matrix[it.multi_index[0], it.multi_index[1]]
            it.iternext()

        output_activation_matrix = output_activation_vector.reshape(self._output_shape)
        return output_activation_matrix





