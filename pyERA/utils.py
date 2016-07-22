#!/usr/bin/python

import numpy as np

class ExponentialDecay:

    def __init__(self, starter_value, decay_step, decay_rate, staircase=False):
        """Function that applyes an exponential decay to a value.

        @param starting_value the value to decay
        @param global_step the global step to use for decay (positive integer)
        @param decay_step (positive integer)
        @param decay_rate 
        """
        self.starter_value = starter_value
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.last_value = starter_value

    def return_decayed_value(self, global_step):
        """Returns the decayed value.

        decayed_value = starting_value * decay_rate ^ (global_step / decay_steps)
        @param global_step the global step to use for decay (positive integer)
        """
        if(self.staircase == False):      
            decayed_value = self.last_value * np.power(self.decay_rate, (global_step / self.decay_step))
            self.last_value = decayed_value
            return decayed_value

        else:
            if(global_step % self.decay_step == 0):
                decayed_value = self.starter_value * np.power(self.decay_rate, (global_step / self.decay_step))
                self.last_value = decayed_value
                return decayed_value
            else:
                return self.last_value 


class LinearDecay:

    def __init__(self, starter_value, decay_rate, allow_negative=True):
        """Function that applyes an exponential decay to a value.

        @param starting_value the value to decay
        @param decay_rate 
        """
        self.starter_value = starter_value
        self.decay_rate = decay_rate
        self.allow_negative = allow_negative

    def return_decayed_value(self, global_step):
        """Returns the decayed value.

        decayed_value = starting_value * decay_rate ^ (global_step / decay_steps)
        @param global_step the global step to use for decay (positive integer)
        """      
        decayed_value = self.starter_value - (self.decay_rate * global_step)
        if(self.allow_negative == False and decayed_value < 0): return 0.0
        else: return decayed_value








