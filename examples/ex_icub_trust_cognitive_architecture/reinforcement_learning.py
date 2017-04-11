#!/usr/bin/python

# The MIT License (MIT)
#
# Copyright (c) 2017 Massimiliano Patacchiola
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np


class ReinforcementLearner:
    def __init__(self, tot_images, tot_labels, tot_actions, informant_vector):
        """Init method.
        
        @param tot_images: number of images to consider 
        @param tot_labels: number of labels to consider
        @param tot_actions: number of actions (accept,reject)
        @param informant_vector: a list of list containing a binomial distribution
            which represents the reliability of the informant. For example:
            informant_vector=[[1,100],[1,1],[1,1]] there are 3 informants
            where the first is very reliable and the other two have uniform distribution.
        """
        # Declaring variables
        self.tot_images = tot_images
        self.tot_labels = tot_labels
        self.tot_actions = tot_actions
        self.informant_vector = np.array(informant_vector)
        # Building the CRITIC
        # The critic is a vector of (tot_images * tot_labels)
        self.critic_vector = np.zeros((1, tot_images * tot_labels))
        # - Building the ACTOR
        # The actor is a matrix of  tot_actions * (tot_images * tot_labels)
        self.actor_matrix = np.zeros((tot_actions, tot_images * tot_labels))

    def _softmax(self, x):
        """Compute softmax values of array x.

        @param x the input array
        @return the softmax array
        """
        return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

    def _return_cost(self, child_confidence, informant_reputation, child_action, informant_action, value="3yo"):
        """This function return the cost of taking an action in accordance or not with the informant.
        It is a Bayesian Network (Forward Step) where a Conditional Probability Table is used.
        
        @param child_confidence: 1-0 value representing the confidence sampled from distribution
        @param informant_reputation: the reputation sampled from the informant distribution
        @param child_action: the action (1=accept; 0=reject) of the child
        @param informant_action: the action (1=accept; 0=reject) of the informant
        @param value: a string representing the model used (for the moment only 3yo=3-year-old)
        @return: the cost of taking that action in those conditions
        """
        if value == '3yo':
            if child_confidence == 1 and informant_reputation == 1 and child_action == 1 and informant_action == 1:
                cost = -1.0  # (knowledge, knowledge, accept, accept) = reinforce
            elif (child_confidence == 1 and informant_reputation == 1 and child_action == 0 and informant_action == 1):
                cost = +0.5  # (knowledge, knowledge, reject, accept) = slightly punish
            elif (child_confidence == 1 and informant_reputation == 1 and child_action == 1 and informant_action == 0):
                cost = +0.5  # (knowledge, knowledge, accept, reject) = slightly punish
            elif (child_confidence == 1 and informant_reputation == 1 and child_action == 0 and informant_action == 0):
                cost = -1.0  # (knowledge, knowledge, reject, reject) = reinforce
            elif (child_confidence == 0 and informant_reputation == 1 and child_action == 1 and informant_action == 1):
                cost = -1.0  # (non-knowledge, knowledge, accept, accept) = reinforce
            elif (child_confidence == 0 and informant_reputation == 1 and child_action == 0 and informant_action == 0):
                cost = -1.0  # (non-knowledge, knowledge, reject, reject) = reinforce
            elif (child_confidence == 0 and informant_reputation == 1 and child_action == 0 and informant_action == 1):
                cost = +1.0  # (non-knowledge, knowledge, reject, accept) = reinforce
            elif (child_confidence == 0 and informant_reputation == 1 and child_action == 1 and informant_action == 0):
                cost = +1.0  # (non-knowledge, knowledge, accept, reject) = punish
            elif (child_confidence == 1 and informant_reputation == 0 and child_action == 1 and informant_action == 1):
                cost = -1.0  # (knowledge, non-knowledge, accept, accept) = reinforce
            elif (child_confidence == 1 and informant_reputation == 0 and child_action == 0 and informant_action == 1):
                cost = +0.5  # (knowledge, non-knowledge, reject, accept) = slightly punish
            elif (child_confidence == 1 and informant_reputation == 0 and child_action == 1 and informant_action == 0):
                cost = +0.5  # (knowledge, non-knowledge, accept, reject) = slightly punish
            elif (child_confidence == 1 and informant_reputation == 0 and child_action == 0 and informant_action == 0):
                cost = -1.0  # (knowledge, non-knowledge, reject, reject) = reinforce
            elif (child_confidence == 0 and informant_reputation == 0 and child_action == 1 and informant_action == 1):
                cost = -1.0  # (non-knowledge, non-knowledge, accept, accept) = reinforce
            elif (child_confidence == 0 and informant_reputation == 0 and child_action == 0 and informant_action == 1):
                cost = +1.0  # (non-knowledge, non-knowledge, reject, accept) = punish
            elif (child_confidence == 0 and informant_reputation == 0 and child_action == 1 and informant_action == 0):
                cost = +1.0  # (non-knowledge, non-knowledge, accept, reject) = punish
            elif (child_confidence == 0 and informant_reputation == 0 and child_action == 0 and informant_action == 0):
                cost = -1.0  # (non-knowledge, non-knowledge, reject, reject) = reinforce
            else:
                raise ValueError("ERROR: the '3yo' Bayesian Networks input values are out of range")
            return cost

        elif value == '4yo':
            if (child_confidence == 1 and informant_reputation == 1 and child_action == 1 and informant_action == 1):
                cost = -1.0  # (knowledge, knowledge, accept, accept) = reinforce
            elif (child_confidence == 1 and informant_reputation == 1 and child_action == 0 and informant_action == 1):
                cost = +0.5  # (knowledge, knowledge, reject, accept) = slight punish
            elif (child_confidence == 1 and informant_reputation == 1 and child_action == 1 and informant_action == 0):
                cost = +0.5  # (knowledge, knowledge, accept, reject) = slight punish
            elif (child_confidence == 1 and informant_reputation == 1 and child_action == 0 and informant_action == 0):
                cost = -1.0  # (knowledge, knowledge, reject, reject) = reinforce
            elif (child_confidence == 0 and informant_reputation == 1 and child_action == 1 and informant_action == 1):
                cost = -1.0  # (non-knowledge, knowledge, accept, accept) = reinforce
            elif (child_confidence == 0 and informant_reputation == 1 and child_action == 0 and informant_action == 0):
                cost = -1.0  # (non-knowledge, knowledge, reject, reject) = reinforce
            elif (child_confidence == 0 and informant_reputation == 1 and child_action == 0 and informant_action == 1):
                cost = +1.0  # (non-knowledge, knowledge, reject, accept) = punish
            elif (child_confidence == 0 and informant_reputation == 1 and child_action == 1 and informant_action == 0):
                cost = +1.0  # (non-knowledge, knowledge, accept, reject) = punish
            elif (child_confidence == 1 and informant_reputation == 0 and child_action == 1 and informant_action == 1):
                cost = 0.0  # (knowledge, non-knowledge, accept, accept) =
            elif (child_confidence == 1 and informant_reputation == 0 and child_action == 0 and informant_action == 1):
                cost = 0.0  # (knowledge, non-knowledge, reject, accept) =
            elif (child_confidence == 1 and informant_reputation == 0 and child_action == 1 and informant_action == 0):
                cost = 0.0  # (knowledge, non-knowledge, accept, reject) =
            elif (child_confidence == 1 and informant_reputation == 0 and child_action == 0 and informant_action == 0):
                cost = 0.0  # (knowledge, non-knowledge, reject, reject) =
            elif (child_confidence == 0 and informant_reputation == 0 and child_action == 1 and informant_action == 1):
                cost = 0.0  # (non-knowledge, non-knowledge, accept, accept) = zero_cost
            elif (child_confidence == 0 and informant_reputation == 0 and child_action == 0 and informant_action == 1):
                cost = 0.0  # (non-knowledge, non-knowledge, reject, accept) = zero_cost
            elif (child_confidence == 0 and informant_reputation == 0 and child_action == 1 and informant_action == 0):
                cost = 0.0  # (non-knowledge, non-knowledge, accept, reject) = zero_cost
            elif (child_confidence == 0 and informant_reputation == 0 and child_action == 0 and informant_action == 0):
                cost = 0.0  # (non-knowledge, non-knowledge, reject, reject) = zero_cost
            else:
                raise ValueError("ERROR: the '4yo' Bayesian Networks input values are out of range")
            return cost
        else:
            raise ValueError("ERROR: input value not recognised, correct values are '3yo' and '4yo'")

    def training(self, dataset, repeat=1, gamma=1.0, learning_rate=0.1, model='3yo'):
        """This function does learning and set the values in the
        ERA architecture weights
        
        @param model: a string representing the model to use 3yo=3-years-old, 4yo=4-years-old
        @param dataset: list of tuple where each tuple is an episode and contains 4 values:
            image_index(int), label_index(int), informant_index(int), informant_action(0=reject,1=accept)
        @param repeat: an integer specifying how many times the training for that dataset is repeated
        @param gamma: reinforcement learning gamma (not used)
        @param learning_rate: the learning rate for the reinforcement step
        """
        for _ in range(repeat):
            for episode in dataset:
                # 1- Get the data stored inside the dataset
                image_index = episode[0]  # image of the object
                label_index = episode[1]  # label given by the informant
                informant_index = episode[2]  # a integer representing the informant
                informant_action = episode[3]  # 0=reject, 1=accept

                # 2- The agent take an action (with softmax) considering is current state-action table
                # [0=cup, 1=book, 2=ball]
                col = (image_index * self.tot_images) + label_index
                action_array = self.actor_matrix[:, col]
                action_distribution = self._softmax(action_array)
                child_action = np.random.choice(self.tot_actions,
                                                1,
                                                p=action_distribution)  # select the action through softmax

                # 3- (External) New state and reward obtained from the environment
                # u_t = self.critic_vector[0, col]  # previous state
                # New state is estimated, in this simple case nothing happen
                # because the next state is terminal
                # u_t1 = u_t  # Only in this example they are the same

                # 4- (Intrinsic) The informant_reputation is updated:
                # agent_action, agent_confidence, informant_action, reward
                # informant_vector: 0=unreliable, 1=reliable
                # do_actions_agree: False, True
                # Estimating child_confidence
                distance = np.absolute(action_distribution[0] - action_distribution[1])
                child_confidence_distribution = [1 - distance, distance]  # non-knowledgeable, knowledgeable
                child_confidence = np.random.choice(2, 1, p=child_confidence_distribution)
                # Check if child and informant agree
                if (child_action == informant_action):
                    do_actions_agree = True
                else:
                    do_actions_agree = False
                # Increment the counter in the informant_vector.
                # Here we update the counter distribtuion only if
                # the child is confident, because it is only in that
                # case that the child can say if the informant is
                # reliable or not.
                if (do_actions_agree == False and child_confidence == 1):
                    self.informant_vector[informant_index][0] += 1  # unreliable
                elif (do_actions_agree == True and child_confidence == 1):
                    self.informant_vector[informant_index][1] += 1  # reliable
                elif (do_actions_agree == False and child_confidence == 0):
                    self.informant_vector[informant_index][1] += 0  # reliable
                    self.informant_vector[informant_index][0] += 0  # unreliable
                elif (do_actions_agree == True and child_confidence == 0):
                    self.informant_vector[informant_index][1] += 0  # reliable
                    self.informant_vector[informant_index][0] += 0  # unreliable
                else:
                    raise ValueError("ERROR: anomaly in the IF condition for informant_vector update")
                # Using the informant_vector given as input it estimates the reputation of the informant
                informant_reputation_distribution = np.true_divide(self.informant_vector[informant_index],
                                                                   np.sum(self.informant_vector[informant_index]))
                informant_reputation = np.random.choice(2, 1, p=informant_reputation_distribution)

                # 5- (Intrinsic) The Cost is estimated:
                # current_state, agent_action, agent_confidence, informant_action, informant_reputation
                # child_confidence: 0=non-knowledgeable, 1=knowledgeable
                # informant_reputation: 0=non-knowledgeable, 1=knowledgeable
                # action: 0=reject, 1=accept
                # informant_action: 0=reject, 1=accept
                cost = self._return_cost(child_confidence,
                                         informant_reputation,
                                         child_action,
                                         informant_action,
                                         value=model)

                # 6- The utility table is updated using: previous_state, current_state, cost, reward
                # Updating the critic using Temporal Differencing Learning
                # In this simple case there is not a u_t1 state.
                # The current state is considered terminal.
                # We can delete the term (gamma*u_t1)-u_t and considering
                # only (reward-cost) as utility of the state (see Russel Norvig).
                reward = 0  # only for intrinsic learning reward=0
                delta = (reward - cost)  # + (gamma*u_t1) - u_t
                self.critic_vector[0, col] += learning_rate * delta

                # 7- The actor table is updated using the delta from the critic
                # Update the ACTOR using the delta
                self.actor_matrix[child_action, col] += learning_rate * delta  # the current action
                self.actor_matrix[1 - child_action, col] -= learning_rate * delta  # the opposite action

    def predict_informant_reliability(self, informant_index):
        """Predict the reliability of an informant based on the experience accumulated.
        
        This function is a random sampling in the informant reliability distribution.
        The distribution is updated after each training call for that informant.
        @param informant_index: the index representing the informant (int)
        @return: return 0=unreliable or 1=reliable
        """
        informant_distribution = np.true_divide(self.informant_vector[informant_index],
                                                np.sum(self.informant_vector[informant_index]))
        informant_performance = np.random.choice(2, 1, p=informant_distribution)
        return int(informant_performance)  # 0=unreliable, 1=reliable

    def predict_object_name(self, object_image_index):
        """Given the learning phase and the weights of the actor-ERA
        it returns the label associated to an object index
        representing the image of the object.
        
        @param object_image_index: integer representing the index of the image stored
        @return: an integer representing the label stored in memory
        """
        col_start = (object_image_index * self.tot_images)
        col_stop = (object_image_index * self.tot_images) + self.tot_labels
        child_answer_distribution = self.actor_matrix[1, col_start:col_stop]  # second row (accept)
        # Take an object based on the posterior distribution
        # argmax_index = np.argmax(child_answer_distribution)
        argmax_value = np.amax(child_answer_distribution)
        counter = 0
        index_list = list()
        for n in child_answer_distribution:
            if n == argmax_value:
                index_list.append(counter)
            counter += 1
        if len(index_list) == 1:
            return int(index_list[0])
        else:
            # Sampling from the indeces with the highest values
            return int(np.random.choice(index_list, 1))  # uniform sampling
