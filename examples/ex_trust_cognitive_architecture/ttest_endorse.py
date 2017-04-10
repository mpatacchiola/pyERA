#!/usr/bin/python

## Massimiliano Patacchiola, Plymouth University 2017
# https://mpatacchiola.github.io
# This script print the t-test results for the
# endorse test with robots.
#
# Endorse: two informants (reliable and unreliable) give different labels
# to 3 new objects. The child has to say the name of the object.
# As in the original experiment of Harris et al. we assume a
# probability of 1/2 to choose the correct label for each object.

from scipy import stats
import numpy as np

#This random list represent the distribution obtained when choosing
#randomly between two option in each one of the 3 trials.
random_list = list()
for _ in range(10000):
    random_list.append(np.sum(np.random.randint(2, size=3)))

print("----- Random list -----")
print("Mean: " + str(np.mean(random_list)))
print("Std: " + str(np.std(random_list)))
print("")

#Experimental distribtuions obtained through simulated agents
#Learning rate 0.1, informant reputation counter not updated when child_confidence==0
#sample_4yo_array = np.array([1, 3, 3, 1, 2, 1, 2, 3, 3, 2, 3, 2, 3, 0, 2, 2, 3, 2, 0, 1, 2, 1, 1, 3, 2]) #25 samples
#sample_3yo_array = np.array([2, 1, 1, 2, 2, 2, 1, 1, 1, 3, 1, 2, 2, 3, 3, 0, 2, 0, 1, 2, 2, 1, 2, 1, 2]) #25 samples
#sample_4yo_array = np.array([1, 3, 3, 1, 2, 1, 2, 3, 3, 2, 3, 2, 3, 0, 2, 2, 3, 2, 0, 1]) #20 samples 
#sample_3yo_array = np.array([2, 1, 1, 2, 2, 2, 1, 1, 1, 3, 1, 2, 2, 3, 3, 0, 2, 0, 1, 2]) #20 samples

sample_3yo_array = np.array([1, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 0, 1, 1, 3, 2, 3, 2, 2])
sample_4yo_array = np.array([2, 2, 3, 3, 1, 1, 2, 1, 2, 3, 2, 3, 0, 3, 3, 3, 1, 2, 1, 1, 0, 1, 3, 3, 2])

t, prob = stats.ttest_ind(sample_4yo_array, random_list, equal_var=True)
print("----- 4-years-old -----")
print("Mean: " + str(np.mean(sample_4yo_array)))
print("Std: " + str(np.std(sample_4yo_array)))
print("t = " + str(t))
print("prob = " + str(prob))
print("")

t, prob = stats.ttest_ind(sample_3yo_array, random_list, equal_var=True)
print("----- 3-years-old -----")
print("Mean: " + str(np.mean(sample_3yo_array)))
print("Std: " + str(np.std(sample_3yo_array)))
print("t = " + str(t))
print("prob = " + str(prob))
print("")





