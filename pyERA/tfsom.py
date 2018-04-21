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
        #-Global variables
        self.tot_rows = tot_rows
        self.tot_cols = tot_cols
        self.depth = depth
        #-Placeholders
        self.input_placeholder = tf.placeholder(tf.float32, shape=depth, name="input_placeholder")
        self.learning_rate_placeholder = tf.placeholder(tf.float32, shape=None, name="learning_rate_placeholder")
        self.radius_placeholder = tf.placeholder(tf.float32, shape=None, name="radius_placeholder")
        #-Constants
        indices_matrix = np.zeros((tot_rows, tot_cols, 2))
        for row in range(tot_rows):
            for col in range(tot_cols):
                indices_matrix[row,col,:] = (row, col)
        #print indices_matrix
        #print "Check...."
        #a = np.array([1,2])
        #print(np.linalg.norm(a-indices_matrix, axis=2))
        #return
        grid_matrix = tf.constant(indices_matrix, shape=[tot_rows, tot_cols, 2])

        #-Variables and saver
        self.weight = tf.get_variable("weights", [tot_rows, tot_cols, depth], initializer=tf.random_uniform_initializer(minval=low, maxval=high))
        self.tf_saver = tf.train.Saver({"weights": self.weight})
        #initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        #-Distance estimation
        difference = tf.subtract(self.input_placeholder, self.weight)
        self.distance_matrix = tf.norm(difference, ord='euclidean', axis=2, name="euclidean_matrix")
        distance_matrix_flatten = tf.reshape(self.distance_matrix, [-1])
        self.softmax_distance_matrix = tf.reshape(tf.nn.softmax(tf.multiply(distance_matrix_flatten, -1.0)), shape=[tot_rows,tot_cols])

        #-Train operations
        #find the index of the best matching unit
        self.distance_argmin = tf.argmin(distance_matrix_flatten)
        self.distance_min = tf.gather(distance_matrix_flatten, self.distance_argmin)
        #generate a tensor with the best matching unit coord
        grid_matrix_flatten = tf.reshape(grid_matrix, [-1,2])
        bmu_coord = tf.gather(grid_matrix_flatten, self.distance_argmin, axis=0)
        #difference between the coord tensor and the constant grid matrix
        difference_neighborhood = tf.subtract(bmu_coord, grid_matrix)
        self.neighborhood_matrix = tf.norm(difference_neighborhood, ord='euclidean', axis=2, name="neighborhood_matrix")
        #pass the neighborhood_matrix through a linear function
        #after this step the BMU having distance 0 has distance 1
        coefficient = tf.divide(-1.0, self.radius_placeholder)
        self.neighborhood_matrix = tf.add(tf.multiply(coefficient, tf.cast(self.neighborhood_matrix, tf.float32)), 1.0) #linearly scale the distance
        self.neighborhood_matrix = tf.clip_by_value(self.neighborhood_matrix, clip_value_min=0.0, clip_value_max=1.0)

        #-Evaluate the delta
        self.weighted_learning_rate_matrix = tf.multiply(self.learning_rate_placeholder, self.neighborhood_matrix)
        self.weighted_learning_rate_matrix = tf.expand_dims(self.weighted_learning_rate_matrix, axis=2)
        delta = tf.multiply(self.weighted_learning_rate_matrix, difference)
        self.train = self.weight.assign(tf.add(self.weight, delta))

        #-Error measures
        self.distance_mean = tf.reduce_mean(self.distance_matrix)
        weight_flatten = tf.reshape(self.weight, [-1, depth])
        self.bmu_array = tf.gather(weight_flatten, self.distance_argmin)
        #self.reconstruction_error = tf.norm(tf.subtract(self.input_placeholder, self.bmu_array), ord='euclidean')

    def return_distance(self, sess, input_array, softmax=False):
        """Return a matrix of distances between the input
           array and all the units of the SOM.

        @param sess the tensorflow session
        @param input_array a numpy array
        @param softmax (default: False) when True the sum of the elements
            in the distance matrix is 1.0
        """
        if(softmax):
            output = sess.run(self.softmax_distance_matrix, feed_dict={self.input_placeholder: input_array})
        else:
            output = sess.run(self.distance_matrix, feed_dict={self.input_placeholder: input_array})
        return output

    def return_BMU_distance(self, sess, input_array):
        """Return the coordinates of the BMU.

        @param sess the tensorflow session
        @param input_array a numpy array
        """
        output = sess.run(self.distance_min, feed_dict={self.input_placeholder: input_array})
        return output

    def return_BMU_coord(self, sess, input_array):
        """Return the coordinates of the BMU.

        @param sess the tensorflow session
        @param input_array a numpy array
        @return the index on the flat array and (row,col) in the matrix
        """
        output = sess.run([self.distance_matrix,self.distance_argmin], feed_dict={self.input_placeholder: input_array})
        index = output[1] #flatten index
        row = index/self.tot_cols
        col = index - (row*self.tot_cols)
        return index, (row,col)

    def training_single_step(self, sess, input_array, learning_rate, radius):
        """A single step of the training procedure.

        It updates the weights using the Kohoen learning rule.
        @param sess the tensorflow session
        @param input_array the vector to use for the comparison.
        @param learning_rate
        @param radius positive real, used to update the weights based on distance.
            all the weights < radius are updated
        @return the average distance of the weights from the input array, the min distance
        """
        output = sess.run([self.train, self.distance_mean, self.distance_min], 
                           feed_dict={self.input_placeholder: input_array, 
                                      self.learning_rate_placeholder: learning_rate,
                                      self.radius_placeholder: radius})
        return output[1], output[2]
        

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
    input_shape = (4)
    start = time.time()
    my_som = tfSom(tot_rows=5, tot_cols=5, depth=input_shape, low=0.0, high=1.0, verbose=False)
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
        print("Distance matrix: ")
        print(my_som.return_distance(sess, input_array, softmax=False))
        print("")
        print("Softmax Distance matrix: ")
        print(my_som.return_distance(sess, input_array, softmax=True))
        print("")
        print("Learning rate matrix: ")
        print(sess.run(my_som.weighted_learning_rate_matrix, feed_dict={"input_placeholder:0": input_array,
                                                                        "learning_rate_placeholder:0": 0.1,
                                                                        "radius_placeholder:0": 2.0}))
        print("")
        print("Neighborhood matrix: ")
        print(sess.run(my_som.neighborhood_matrix, feed_dict={"input_placeholder:0": input_array,
                                                              "learning_rate_placeholder:0": 0.1,
                                                              "radius_placeholder:0": 2.0}))
        print("")
        print("Distance mean: ")
        print(sess.run(my_som.distance_mean, feed_dict={"input_placeholder:0": input_array}))
        print("")
        print("BMU distance: ")
        output = my_som.return_BMU_distance(sess, input_array)
        print output        
        print("")
        print("BMU Coord: ")
        output = my_som.return_BMU_coord(sess, input_array)
        print output
        print("")
        print("BMU array: ")
        print(sess.run(my_som.bmu_array, feed_dict={"input_placeholder:0": input_array}))
        print("")
        print("update...")
        my_som.training_single_step(sess, input_array, learning_rate=0.1, radius=2.0)
        print("Weights updated: ")
        print(sess.run(my_som.weight))
        print("")
        #print("Reconstruction error: ")
        #print(sess.run(my_som.reconstruction_error, feed_dict={"input_placeholder:0": input_array}))
        #print("")
    end = time.time()
    print("Finished in: " + str(end - start))

    start = time.time()
    input_array = np.random.uniform(size=input_shape)
    for i in range(1):
        #input_array = np.random.uniform(size=input_shape)
        average_distance, min_distance = my_som.training_single_step(sess, input_array, learning_rate=0.1, radius=2.0)
        print("Step number: " + str(i))
        print("Average distance: " + str(average_distance))
        print("Minimum distance: " + str(min_distance))
        print("")
    end = time.time()
    print("Finished in: " + str(end - start))

    #my_som.save(sess, save_path="./log/model.ckpt", verbose=True)


if __name__ == "__main__":
    main()


