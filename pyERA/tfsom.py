#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2018 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import tensorflow as tf

class tfSom:
    """tfSom Class

    This is an implementation of a Self-Organizing Map (SOM) in Tensorflow.
    It provides low level funcion and utilities for assembling diffeent type of som.
    """

    def __init__(self, tot_rows, tot_cols, input_shape, low=0.0, high=1.0, verbose=True):
        """Init function.

        @param matrix_size It defines the matrix size as [row,col]
        @param input_size it defines the vector input size.
        @param low boundary for the random initialization 
        @param high boundary for the random initialization
        @param round_values it is possible to initialize the 
        weights to the closest integer value.
        """
        self.tot_rows = tot_rows
        self.tot_cols = tot_cols
        self.input_shape = input_shape
        #self._weights_matrix = np.random.uniform(low=low, high=high, size=(matrix_size, matrix_size, input_size))
        #self.weights_matrix = tf.random_uniform(shape=matrix_size+input_size, minval=low, maxval=high, dtype = tf.float32, name="weights")
        self.index2coord_dict = {}
        self.euclidean_list = []

        input_placeholder = tf.placeholder(tf.float32, shape=input_shape, name="input_placeholder")
        if(verbose): print input_placeholder
        counter = 0
        for row in range(tot_rows):
            for col in range(tot_cols):
                index = (row, col)
                weight = tf.random_uniform(shape=input_shape, minval=low, maxval=high, dtype = tf.float32, name="weight_" + str(row) + "-" + str(col))
                euclidean = tf.norm(tf.subtract(input_placeholder, weight), ord='euclidean', name="euclidean_" + str(row) + "-" + str(col))
                if(verbose): print weight
                if(verbose): print euclidean
                self.euclidean_list.append(euclidean)
                self.index2coord_dict[counter] = (row,col)
                counter += 1

        weight = tf.random_uniform(shape=[tot_rows, tot_cols, input_shape[1]], minval=low, maxval=high, dtype = tf.float32, name="weight")
        self.distance_matrix = tf.norm(tf.subtract(input_placeholder, weight), ord='euclidean', axis=2, name="euclidean_matrix")
        self.distance_argmin = tf.argmin(self.distance_matrix)
        self.elements_gather = tf.gather(self.distance_matrix, self.distance_argmin)

    def return_BMU_index(self, sess, input_array):
        """Return the coordinates of the BMU.

        @param sess the tensorflow session
        @param input_array a numpy array
        """
        output = sess.run(self.euclidean_list, feed_dict={"input_placeholder:0": input_array})
        return np.argmax(output)

    def return_BMU_coord(self, sess, input_array):
        """Return the coordinates of the BMU.

        @param sess the tensorflow session
        @param input_array a numpy array
        """
        output = sess.run(self.euclidean_list, feed_dict={"input_placeholder:0": input_array})
        index = np.argmax(output)
        return self.index2coord_dict[index]

    def return_BMU_value_fast(self, sess, input_array):
        """Return the coordinates of the BMU.

        @param sess the tensorflow session
        @param input_array a numpy array
        """
        output = sess.run(self.elements_gather, feed_dict={"input_placeholder:0": input_array})
        return output

    def return_BMU_coord_fast(self, sess, input_array):
        """Return the coordinates of the BMU.

        @param sess the tensorflow session
        @param input_array a numpy array
        """
        output = sess.run([self.distance_matrix,self.distance_argmin], feed_dict={"input_placeholder:0": input_array})
        #index = np.argmax(output)
        #return self.index2coord_dict[index]
        return output

def main():

    #TEST time
    #input_shape = (1,32*32*3)
    #tot_rows=25, tot_cols=25
    #for i in range(1000)

    #my_som.return_BMU_coord:       18.9095129967 seconds
    #my_som.return_BMU_value_fast:  11.027312994  seconds

    import time

    #Init the SOM
    print("Initializing SOM...")
    input_shape = (1,32*32*3)
    start = time.time()
    my_som = tfSom(tot_rows=25, tot_cols=25, input_shape=input_shape, low=0.0, high=1.0, verbose=False)
    end = time.time()
    print("Finished in: " + str(end - start))

    sess = tf.Session()

    print("Estimating BMU...")
    start = time.time()
    for i in range(1000):
        input_array = np.random.uniform(size=(1,32*32*3))
        output = my_som.return_BMU_value_fast(sess, input_array)
        #output = my_som.return_BMU_coord(sess, input_array)
        #print index[0].shape
        #print index[1]
    #print output
    end = time.time()
    print("Finished in: " + str(end - start))


if __name__ == "__main__":
    main()


