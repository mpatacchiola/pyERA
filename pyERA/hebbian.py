#!/usr/bin/python

##
# Massimiliano Patacchiola, Plymouth University (2016)
#
# Implementation of hebbian connection and hebbian network class
#


import numpy as np

class HebbianNetwork:
    """HebbianNetwork

    This is an implementation of the hebbian network class.
    """
    def __init__(self, name):
        """Initialize the hebbian network.

        @param name the name of the network
        """
        self.name = name
        #The node_list contains the name and structure
        #of each node inside the network
        self._node_list = list()
        self._connection_list = list()


    def add_node(self, name, shape):
        """Add the node to the network.

        The nodes are added following an incremental index.
        To access the node properties it is necessary to have
        the index associated to it.
        @param name
        @param shape
        """
        rows = shape[0]
        cols = shape[1]
        if(rows <= 0 or cols<=0): raise ValueError('hebbian_network: the widht and the height cannot be negative or null.')
        temp_maptrix = np.zeros((rows, cols))
        dict = {'Name': name, 'Rows': rows, 'Cols': cols, 'Matrix': temp_maptrix}
        self._node_list.append(dict.copy()) #append a swallow copy of the dict

    #TODO remove all the connections associated with
    #the removed node.
    def remove_node(self, index):
        """Remove the node to the network.

        The nodes are added following an incremental index.
        To remove the node it is necessary to have the index 
        associated to it.
        @param index the numeric node index
        """
        self._node_list.remove(index)

    def set_node_activation_matrix(self, index, matrix):
        """Set the activation matrix associated with a node.

        The nodes are added following an incremental index.
        To remove the node it is necessary to have the index 
        associated to it.
        @param index the numeric node index
        """
        self._node_list[index]['Matrix'] = matrix

    def get_node_activation_matrix(self, index):
        """Get the activation matrix associated with a node.

        The nodes are added following an incremental index.
        To remove the node it is necessary to have the index 
        associated to it.
        @param index the numeric node index
        """     
        return self._node_list[index]['Matrix']

    #TODO check if the connection between the two nodes 
    #already exists.
    def add_connection(self, input_node_index, output_node_index):
        """Add a connection between two nodes.

        @param input_node_index
        @param output_node_index
        """
        if(input_node_index< 0 or output_node_index<0 or input_node_index>=len(self._node_list) or output_node_index>=len(self._node_list) or input_node_index==output_node_index): 
            raise ValueError('hebbian_network: there is a conflict in the index.')
        input_shape = (self._node_list[input_node_index]['Rows'], self._node_list[input_node_index]['Cols'])
        output_shape = (self._node_list[output_node_index]['Rows'], self._node_list[output_node_index]['Cols'])
        temp_connection = HebbianConnection(input_shape, output_shape)
        dict = {'input_index': input_node_index, 'output_index': output_node_index, 'connection': temp_connection }
        self._connection_list.append(dict.copy())

    def return_total_nodes(self):
        """Return the total number of nodes in the network

        """
        return len(self._node_list)

    def return_total_connections(self):
        """Return the total number of connections in the network

        """
        return len(self._connection_list)

    def print_info(self):
        """Print on the terminal some info about the network

        """
        print("")
        print("Name ..... " + str(self.name))
        print("Total Nodes ..... " + str(self.return_total_nodes()))
        print("Total Connections ..... " + str(self.return_total_connections()))
        print("")


class HebbianConnection:
    """HebbianConnecttion

    This is an implementation of an hebbian connection
    between two layers.
    """
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
        #The number of rows in weights_matris is equal to the number of elements (rows*cols) in input_matrix
        #The number of cols in weights_matrix is equal to the number of elements (rows*cols) in output_matrix
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


    def compute_activation(self, activation_matrix, reverse=False):
        """It returns the activation matrix of input/output layer

        @param input_activation_matrix a vector or a bidimensional matrix representing the activation of the input units
        @param reverse it defines the computation direction, False=Forward (input > Ouptu), True=Backward (Input < Output)
        """
        #Forward activation
        if(reverse == False):

            input_activation_vector = activation_matrix.flatten()
            output_activation_matrix = np.zeros(self._output_shape)
            output_activation_vector = output_activation_matrix.flatten()

            #Iterates the elements in weights_matrix and use the row index for
            #accessing the element of the flatten input matrix.
            it = np.nditer(self._weights_matrix, flags=['multi_index'])
            while not it.finished:
                output_activation_vector[it.multi_index[1]] +=  input_activation_vector[it.multi_index[0]] * self._weights_matrix[it.multi_index[0], it.multi_index[1]]
                it.iternext()

            output_activation_matrix = output_activation_vector.reshape(self._output_shape)
            return output_activation_matrix

        #Backward activation
        elif(reverse == True):

            output_activation_vector = activation_matrix.flatten()
            input_activation_matrix = np.zeros(self._input_shape)
            input_activation_vector = input_activation_matrix.flatten()

            #Iterates the elements in weights_matrix and use the col index for
            #accessing the element of the flatten output matrix.
            it = np.nditer(self._weights_matrix, flags=['multi_index'])
            while not it.finished:
                input_activation_vector[it.multi_index[0]] +=  output_activation_vector[it.multi_index[1]] * self._weights_matrix[it.multi_index[0], it.multi_index[1]]
                it.iternext()

            input_activation_matrix = input_activation_vector.reshape(self._input_shape)
            return input_activation_matrix


