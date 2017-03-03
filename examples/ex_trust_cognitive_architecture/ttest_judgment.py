#!/usr/bin/python

## Massimiliano Patacchiola, Plymouth University 2017
# https://mpatacchiola.github.io
# This script print the t-test results for the
# endorse test with robots.
#
# Judgment: two informants (reliable and unreliable) give different labels
# to 3 new objects. The child has to say which informant was not good
# at answering questions. Correct if child indicates unreliable informant.

from scipy import stats
import numpy as np

#This random list represent the distribution obtained when choosing
#randomly between two option in a single trial.
random_list = np.random.randint(2, size=10000)

print("----- Random list -----")
print("Mean: " + str(np.mean(random_list)))
print("Std: " + str(np.std(random_list)))
print("")

#Experimental distribtuions obtained through simulated agents
#Learning rate 0.1, informant reputation counter not updated when child_confidence==0

sample_3yo_array = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]) #25 samples
sample_4yo_array = np.array([0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1])

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

