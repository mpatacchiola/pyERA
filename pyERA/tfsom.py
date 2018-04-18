#!/usr/bin/env python

#The MIT License (MIT)
#Copyright (c) 2018 Massimiliano Patacchiola
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
#MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
#CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#TEST time
#input_shape = (1,32*32*3)
#tot_rows=25, tot_cols=25
#for i in range(1000)
#
#On CPU:
#my_som.return_BMU_coord:       18.90 seconds
#my_som.return_BMU_value_fast:  11.02 seconds
#
#On GPU:
#my_som.return_BMU_coord:       85.60 seconds
#my_som.return_BMU_value_fast:  1.730 seconds

import numpy as np
import tensorflow as tf

class tfSom:
    """tfSom Class

    This is an implementation of a Self-Organizing Map (SOM) in Tensorflow.
    It provides low level funcion and utilities for assembling diffeent type of som.
    """

    def __init__(self, tot_rows, tot_cols, depth, low=0.0, high=1.0, verbose=True):
        """Init function.

        @param tot_rows
        @param tot_cols
        @param depth
        @param low boundary for the random initialization 
        @param high boundary for the random initialization
        @param verbose
        """
        self.tot_rows = tot_rows
        self.tot_cols = tot_cols
        self.depth = depth
        #self._weights_matrix = np.random.uniform(low=low, high=high, size=(matrix_size, matrix_size, input_size))
        #self.weights_matrix = tf.random_uniform(shape=matrix_size+input_size, minval=low, maxval=high, dtype = tf.float32, name="weights")
        self.index2coord_dict = {}
        self.euclidean_list = []

        self.input_placeholder = tf.placeholder(tf.float32, shape=depth, name="input_placeholder")
        self.learning_rate_placeholder = tf.placeholder(tf.float32, shape=None, name="learning_rate_placeholder")
        if(verbose): print self.input_placeholder
        counter = 0
        #for row in range(tot_rows):
        #    for col in range(tot_cols):
        #        index = (row, col)
        #        weight = tf.random_uniform(shape=input_shape, minval=low, maxval=high, dtype = tf.float32, name="weight_" + str(row) + "-" + str(col))
        #        euclidean = tf.norm(tf.subtract(input_placeholder, weight), ord='euclidean', name="euclidean_" + str(row) + "-" + str(col))
        #        if(verbose): print weight
        #        if(verbose): print euclidean
        #        self.euclidean_list.append(euclidean)
        #        self.index2coord_dict[counter] = (row,col)
        #        counter += 1
        #weight_initializer = tf.random_uniform(shape=[tot_rows, tot_cols, input_shape[1]], minval=low, maxval=high, dtype = tf.float32, name="weight")

        #Variables and saver
        self.weight = tf.get_variable("weights", [tot_rows, tot_cols, depth], initializer=tf.random_uniform_initializer(minval=low, maxval=high))
        self.tf_saver = tf.train.Saver({"weights": self.weight})
        #initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        #Distance estimation
        difference = tf.subtract(self.input_placeholder, self.weight)
        self.distance_matrix = tf.norm(difference, ord='euclidean', axis=2, name="euclidean_matrix")
        #Train operations
        delta = tf.multiply(self.learning_rate_placeholder, difference)
        self.train = self.weight.assign(tf.add(self.weight, delta))
        #Additional calls
        self.distance_mean = tf.reduce_mean(self.distance_matrix)
        distance_matrix_flatten = tf.reshape(self.distance_matrix, [-1])
        self.distance_argmin = tf.argmin(distance_matrix_flatten)
        self.elements_gather = tf.gather(distance_matrix_flatten, self.distance_argmin)

    #def return_BMU_index(self, sess, input_array):
    #    """Return the coordinates of the BMU.
    #
    #    @param sess the tensorflow session
    #    @param input_array a numpy array
    #    """
    #    output = sess.run(self.euclidean_list, feed_dict={"input_placeholder:0": input_array})
    #    return np.argmax(output)

    #def return_BMU_coord(self, sess, input_array):
    #    """Return the coordinates of the BMU.
    #
    #    @param sess the tensorflow session
    #    @param input_array a numpy array
    #    """
    #    output = sess.run(self.euclidean_list, feed_dict={"input_placeholder:0": input_array})
    #    index = np.argmax(output)
    #    return self.index2coord_dict[index]

    def return_BMU_value(self, sess, input_array):
        """Return the coordinates of the BMU.

        @param sess the tensorflow session
        @param input_array a numpy array
        """
        output = sess.run(self.elements_gather, feed_dict={self.input_placeholder: input_array})
        return output

    def return_BMU_coord(self, sess, input_array):
        """Return the coordinates of the BMU.

        @param sess the tensorflow session
        @param input_array a numpy array
        """
        output = sess.run([self.distance_matrix,self.distance_argmin], feed_dict={self.input_placeholder: input_array})
        index = output[1] #flatten index
        row = index/self.tot_cols
        col = index -(row*self.tot_cols)
        return index, (row,col)

    def training_single_step(self, sess, input_array, learning_rate, radius=None):
        """A single step of the training procedure.

        It updates the weights using the Kohoen learning rule.
        @param sess the tensorflow session
        @param input_array the vector to use for the comparison.
        @param learning_rate
        @param radius (optional) positive real, used to update the weights based on distance.
        @return the average distance of the weights from the input array
        """
        output = sess.run([self.train, self.distance_mean], 
                           feed_dict={self.input_placeholder: input_array, 
                                      self.learning_rate_placeholder: learning_rate})
        return output[1]
        

    def save(self, sess, save_path="./log/model.ckpt", verbose=True):
        """It saves the SOM parameters in a tf file.

        @param sess the tensorflow session
        @param save_path
        @param verbose
        """
        if(verbose): print("Saving model in: " + str(save_path))
        save_path = self.tf_saver.save(sess, save_path)
        if(verbose): print("Done!")

    def load(self, sess, file_path, verbose=True):
        """It saves the SOM parameters in a tf file.

        @param sess the tensorflow session
        @param file_path
        @param verbose
        """
        if(verbose): print("Loading model from: " + str(file_path))
        self.tf_saver.restore(sess, file_path)
        if(verbose): print("Done!")

def main():

    import time

    #Init the SOM
    print("Initializing SOM...")
    input_shape = (32*32*3)
    start = time.time()
    my_som = tfSom(tot_rows=25, tot_cols=25, depth=input_shape, low=0.0, high=1.0, verbose=False)
    end = time.time()
    print("Finished in: " + str(end - start))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #my_som.load(sess, file_path="./log/model.ckpt", verbose=True)

    print("Estimating BMU...")
    start = time.time()
    for i in range(1):
        input_array = np.random.uniform(size=input_shape)
        print("Input: ")
        print(input_array)
        print("")
        print("Weights: ")
        print(sess.run(my_som.weight))
        print("")
        print("Distance: ")
        print(sess.run(my_som.distance_matrix, feed_dict={"input_placeholder:0": input_array}))
        output = my_som.return_BMU_value(sess, input_array)
        print("")
        print("Distance mean: ")
        print(sess.run(my_som.distance_mean, feed_dict={"input_placeholder:0": input_array}))
        print("")
        print("Value: ")
        print output
        output = my_som.return_BMU_coord(sess, input_array)
        print("")
        print("Coord: ")
        print output
    end = time.time()
    print("Finished in: " + str(end - start))

    start = time.time()
    input_array = np.random.uniform(size=input_shape)
    for i in range(10):
        #input_array = np.random.uniform(size=input_shape)
        average_distance = my_som.training_single_step(sess, input_array, learning_rate=0.1)
        print("Step number: " + str(i))
        print("Average distance: " + str(average_distance))
        print("")
    end = time.time()
    print("Finished in: " + str(end - start))

    #my_som.save(sess, save_path="./log/model.ckpt", verbose=True)


if __name__ == "__main__":
    main()


