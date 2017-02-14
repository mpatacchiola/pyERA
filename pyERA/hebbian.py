#!/usr/bin/python

##
# Massimiliano Patacchiola, Plymouth University (2016)
#
# Implementation of hebbian connection and hebbian network classes.
#


import numpy as np

class HebbianNetwork:
    """HebbianNetwork

    This is an implementation of the hebbian network class.
    The Hebbian Network is considered as an integrator of nodes.
    The nodes are references, some kind of abstract labels.
    Each node can be a Self-Organizing Map, the output vector of
    another type of artificial network or the output of a generic system.
    In the first phase the network can be organised step by step adding and removing nodes.
    In a second phase it is possible to connect nodes using Hebbian Connections.
    In a third phase it is possible to compute the activation of each node.
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
        @param name the name of the node (it is different from the index)
        @param shape the shape is a list that indentifies total rows and num cols
        """
        rows = shape[0]
        cols = shape[1]
        if(rows <= 0 or cols<=0): raise ValueError('HebbianNetwork: the widht and the height cannot be negative or null.')
        temp_maptrix = np.zeros((rows, cols))
        dict = {'Name': name, 'Rows': rows, 'Cols': cols, 'Matrix': temp_maptrix}
        self._node_list.append(dict.copy()) #append a swallow copy of the dict

    def remove_node(self, index):
        """Remove the node from the network and all the connections associated.

        The nodes are added following an incremental index.
        To remove the node it is necessary to have the index 
        associated to it.
        @param index the numeric node index
        """
        del self._node_list[index]

        element_counter = 0
        remove_list = list()
        for connection_dict in self._connection_list:
            if(connection_dict['Start']==index or connection_dict['End']==index):
                remove_list.append(element_counter)
                element_counter += 1

        #Code for multiple elements removal
        self._connection_list = [v for i, v in enumerate(self._connection_list) if i not in remove_list]

    def set_node_activations(self, index, matrix):
        """Set the activation matrix associated with a node.

        The nodes are added following an incremental index.
        To remove the node it is necessary to have the index 
        associated to it.
        @param index the numeric node index
        """
        self._node_list[index]['Matrix'] = matrix

    def get_node_activations(self, index):
        """Get the activation matrix associated with a node.

        The nodes are added following an incremental index.
        To remove the node it is necessary to have the index 
        associated to it.
        @param index the numeric node index
        """     
        return self._node_list[index]['Matrix']

    def reset_node_activations(self, index):
        """Reset the activation matrix associated with a node to numpy.zeros

        The nodes are added following an incremental index.
        To remove the node it is necessary to have the index 
        associated to it.
        @param index the numeric node index
        """
        self._node_list[index]['Matrix'] = np.zeros((self._node_list[index]['Rows'], self._node_list[index]['Cols']))

    def return_node_connection_list(self, index):
        """Return a list containing all the nodes connected to the index

        @param index the numeric node index
        """
        node_connection_list = list()
        for connection_dict in self._connection_list:
            if(connection_dict['Start']==index):
                node_connection_list.append(connection_dict['End'])
            elif(connection_dict['End']==index):
                node_connection_list.append(connection_dict['Start'])
        return node_connection_list

    def return_node_incoming_connection_list(self, index):
        """Return a list containing all the nodes that project a connection to the index node

        @param index the numeric node index
        """
        node_connection_list = list()
        for connection_dict in self._connection_list:
            if(connection_dict['End']==index):
                node_connection_list.append(connection_dict['End'])
        return node_connection_list

    def return_node_outgoing_connection_list(self, index):
        """Return a list containing all the nodes that receive a connection from the index node

        @param index the numeric node index
        """
        node_connection_list = list()
        for connection_dict in self._connection_list:
            if(connection_dict['Start']==index):
                node_connection_list.append(connection_dict['Start'])
        return node_connection_list

    def compute_node_activations(self, index, set_node_matrix=True):
        """Compute the node activations and return the activation matrix. 
       
        Based on the activation matrices set with previous call to set_node_activations,
        it computes the index node output matrix multiplying the commections weights with
        the input/output activation matrices of the other nodes afferent to the index.
        @param index the index of the node to compute
        @param set_node_matrix if True before return it assigns the internal output matrix to the index node.
        """
        node_activation_matrix = np.zeros((self._node_list[index]['Rows'], self._node_list[index]['Cols']))

        #Iterate through all the connections looking for the 
        #incoming and outgoing connections to the index node
        for connection_dict in self._connection_list:
            #The node is a starting node then to compute its activation
            #It is necessary to go in reverse from: input < output
            if(connection_dict['Start']==index):
                activation_matrix = self._node_list[connection_dict['End']]['Matrix']               
                node_activation_matrix = np.add(node_activation_matrix, connection_dict['Connection'].compute_activation(activation_matrix, reverse=True)) 
                #node_activation_matrix += connection_dict['Connection'].compute_activation(activation_matrix, reverse=True)
            #The node is an ending node then to compute its activation
            #it is necessary to go directly from: input > output
            if(connection_dict['End']==index):
                activation_matrix = self._node_list[connection_dict['Start']]['Matrix']
                node_activation_matrix = np.add(node_activation_matrix, connection_dict['Connection'].compute_activation(activation_matrix, reverse=False))
                #node_activation_matrix += connection_dict['Connection'].compute_activation(activation_matrix, reverse=False)

        if(set_node_matrix==True): self.set_node_activations(index, node_activation_matrix)
        return node_activation_matrix

    def learning(self, learning_rate=0.01, rule="hebb"):
        """One step learning for all the connections. 
       
        Calling this function the network updates the connection values 
        based on the connection properties and the specific learning rule.
        There are three possible learning rules: hebb, antihebb and oja (see HebbianConnection class)
        @param learning_rate
        @return True if operation succeeded, False if the connection already exists
        """
        #Check if the learning_rate is negative
        if(learning_rate <= 0): raise ValueError("HebbianNetork: Error the learning rate must be positive not null.")

        if(rule!="hebb" and rule!="antihebb" and rule!="oja"):
            raise ValueError('HebbianNetwork: the learning rule does not exist. Available rules are hebb, antihebb and oja.')

        #Cycling thorugh each connection and
        #applying the learning rule.
        for connection_dict in self._connection_list:
            input_index = connection_dict['Start']
            output_index = connection_dict['End']
            input_activation_matrix = self.get_node_activations(input_index)
            output_activation_matrix = self.get_node_activations(output_index)
            if(rule == "hebb"):
                connection_dict['Connection'].learning_hebb_rule(input_activation_matrix, output_activation_matrix, learning_rate)
            elif(rule == "antihebb"):
                connection_dict['Connection'].learning_anti_hebb_rule(input_activation_matrix, output_activation_matrix, learning_rate)
            elif(rule == "oja"):
                connection_dict['Connection'].learning_oja_rule(input_activation_matrix, output_activation_matrix, learning_rate)
            else:
                raise ValueError("HebbianNetork: Error the learning rule specified does not exist.")


    def add_connection(self, first_node_index, second_node_index):
        """Add a connection between two nodes.
       
        @param first_node_index
        @param second_node_index
        @return True if operation succeeded, False if the connection already exists
        """
        if(first_node_index< 0 or second_node_index<0 or first_node_index>=len(self._node_list) or second_node_index>=len(self._node_list) or first_node_index==second_node_index): 
            raise ValueError('HebbianNetwork: there is a conflict in the index.')

        node_connection_list = self.return_node_connection_list(second_node_index)
        if(first_node_index in node_connection_list):
            #print("False, the connection already exists")
            return False

        first_shape = (self._node_list[first_node_index]['Rows'], self._node_list[first_node_index]['Cols'])
        second_shape = (self._node_list[second_node_index]['Rows'], self._node_list[second_node_index]['Cols'])
        temp_connection = HebbianConnection(first_shape, second_shape, add_gaussian_noise=True)
        dict = {'Start': first_node_index, 'End': second_node_index, 'Connection': temp_connection }
        self._connection_list.append(dict.copy())
        return True

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
        print("Net Name ..... " + str(self.name))
        print("Total Nodes ..... " + str(self.return_total_nodes()))
        print("Total Connections ..... " + str(self.return_total_connections()))
        print "Nodes Name: ", 
        for node_dict in self._node_list:
            print node_dict['Name'] + ";",
        print("")


class HebbianConnection:
    """HebbianConnecttion

    This is an implementation of an hebbian connection
    between two nodes.
    @param rule it specifies the learning rule associted with this connection (hebb, antihebb, oja)
    """
    def __init__(self, input_shape, output_shape, add_gaussian_noise=False):
        """Initialize the hebbian connections between two networks.

        The weights of the connection are adjusted using a learning rule.
        @param input_shape, it can be a square matrix or a numpy vector. The vector must be initialized as a 1 row (or 1 column) numpy matrix.
        @param output_shape, it can be a square matrix or a numpy vector. The vector must be initialized as a 1 row (or 1 column) numpy matrix.
        """
        if(len(input_shape) != 2 or len(output_shape) != 2): raise ValueError('HebbianConnection: error the input-outpu matrix shape is != 2')

        self._input_shape = input_shape
        self._output_shape = output_shape

        #The weight matrix is created from the shape of the input/output matrices
        #The number of rows in weights_matris is equal to the number of elements (rows*cols) in input_matrix
        #The number of cols in weights_matrix is equal to the number of elements (rows*cols) in output_matrix
        rows = self._input_shape[0] * self._input_shape[1]
        cols = self._output_shape[0] * self._output_shape[1]

        #Add gaussian noise to each element of the matrix
        if(add_gaussian_noise==True):
            self._weights_matrix = np.random.normal(loc=0.0, scale=0.01, (rows, cols))
        else:
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


    def learning_anti_hebb_rule(self, input_activation_matrix, output_activation_matrix, learning_rate):
        """Single step learning using the Anti-Hebbian update rule.

        The Anti-Hebbian rule: If two neurons on either side of a synapse (connection) are activated simultaneously, 
        then the strength of that synapse is selectively decreased.
        @param input_activations a vector or a bidimensional matrix representing the activation of the input units
        @param output_activations a vector or a bidimensional matrix representing the activation of the output units
        @param learning_rate (positive converted internally to negative) it is costant that defines the decreasing step
        """
        if(learning_rate <=0): raise ValueError('hebbian_connection: error the learning rate used for the anti-hebbian rule must be >0')

        learning_rate = -learning_rate #The antihebb has negative learning_rate
        input_activation_vector = input_activation_matrix.flatten()
        output_activation_vector = output_activation_matrix.flatten()

        it = np.nditer(self._weights_matrix, flags=['multi_index'])
        while not it.finished:
            #print "%d <%s>" % (it[0], it.multi_index)
            #Applying the Hebbian Rule:
            delta_weight = learning_rate * input_activation_vector[it.multi_index[0]] * output_activation_vector[it.multi_index[1]]
            self._weights_matrix[it.multi_index[0], it.multi_index[1]] += delta_weight
            it.iternext()


    def learning_oja_rule(self, input_activation_matrix, output_activation_matrix, learning_rate):
        """Single step learning using the Oja's update rule.

        The Oja's rule normalizes the weights between 0 and 1, trying  to stop the weights increasing indefinitely
        @param input_activations a vector or a bidimensional matrix representing the activation of the input units
        @param output_activations a vector or a bidimensional matrix representing the activation of the output units
        @param learning_rate it is costant that defines the learning step
        """
        if(learning_rate <=0): raise ValueError('hebbian_connection: error the learning rate used for the oja rule must be >0')

        input_activation_vector = input_activation_matrix.flatten()
        output_activation_vector = output_activation_matrix.flatten()

        it = np.nditer(self._weights_matrix, flags=['multi_index'])
        while not it.finished:
            #print "%d <%s>" % (it[0], it.multi_index)
            #Applying the Oja's Rule:
            delta_weight = (learning_rate * input_activation_vector[it.multi_index[0]] * output_activation_vector[it.multi_index[1]]) - \
                           (learning_rate * output_activation_vector[it.multi_index[1]] * output_activation_vector[it.multi_index[1]] * self._weights_matrix[it.multi_index[0], it.multi_index[1]] )
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


