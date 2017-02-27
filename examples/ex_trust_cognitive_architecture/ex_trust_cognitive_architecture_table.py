#!/usr/bin/python

## Massimiliano Patacchiola, Plymouth University 2016

import numpy as np
import os 
import time

# General Algorithm Description
# 1- To the agent is presented an object and a label (current state).
# 2- An informant suggest a possible action (accept or reject the label).
# 3- The agent take an action considering is current state-action table
# 4- (External) New state and reward obtained from the environment
# 5- (Intrinsic) The informant_reputation is updated: agent_action, agent_confidence, informant_action
# 6- (Intrinsic) The Cost is estimated: current_state, agent_action, agent_confidence, informant_action, informant_reputation
# 7- The utility table is updated using: preious_state, current_state, cost, reward
# 8- The actor table is updated using the delta from the critic

#Informant reputation is evaluated considering: agent_action, agent_confidence, informant_action
#If the confidence of the agent is high and the action suggested is different from the action taken
#then the informant is evaluated as unreliable and a counter is incremeneted.
#The reputation counter is considered separated by the Cost function. Harris et al. have showed
#that 3yo children can estimate the reliability of the informant but they cannot estimate the
#cost of following the informant suggestion. This is in accordance with our model where the two
#entities are separated.
#
#Intrinsic environment: evaluates the cost of taking an action
#Trusting an unreliable informant has a cost, because the child will store an information which is not useful.
#This mechanism can be considered part of a planning module (e.g. prefrontal cortex).
#The cost function C can be defined as a function that takes as input: current_state, agent_action, agent_confidence, informant_action, informant_reputation.
#The cost function evaluates what's the cost of having taken an action in state S given the informant advice .
#The output of the function C is a real number representing the COST of taking that action given the informant suggestion.
#This table can be represented through a table or can be approximated through a function approximator (e.g. neural network)
#
#The actor architecture is a table of state-action pairs. 
#When the child has to give a label for an object the policy must be used and not the utility table.
#The most common associated label to a visual object can be estimated setting the SOM action node to ACCEPT and then
#computing the activation of the vocabulary unit. The argmax is the value we want.



def softmax(x):
    '''Compute softmax values of array x.

    @param x the input array
    @return the softmax array
    '''
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

def training(dataset, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions):
    #Hyper-Parameters
    reward = 0
    gamma = 0.9
    learning_rate = 0.1

    counter = 1
    for episode in dataset:

      #Get the action from the ACTOR
      #[0=cup, 1=book, 2=ball]
      image = episode[0] #image of the object
      label = episode[1] #label given by the informant
      informant_index = episode[2] #a integer representing the informant
      informant_intention = episode[3] #0=reject, 1=accept
      col = (image * tot_images) + label
      action_array = actor_matrix[:, col]
      action_distribution = softmax(action_array)
      action = np.random.choice(tot_actions, 1, p=action_distribution) #select the action through softmax

      #Estimate the reward and the cost interrogating the Bayesian Network
      #The distance is 0.0 when the two actions have same probability
      #The distance is 1.0 if one of the two actions has probability 1
      distance = np.absolute(action_distribution[0] - action_distribution[1])
      bn_child_knowledge_distribution = [1-distance, distance] #non-knowledgeable, knowledgeable 

      #Increment the counter in the informant_vector based on the current observation
      informant_knowledge = np.random.choice(2, 1, p=action_distribution)
      if(informant_knowledge==1):
          informant_vector[informant_index][1] += 1
      else:
          informant_vector[informant_index][0] += 1
      #Using the informant_vector given as input it estimates the reputation of the informant
      informant_knowledge_distribution = np.true_divide(informant_vector[informant_index], np.sum(informant_vector[informant_index]))

      #Estimate the cost (Accept or Reject the label proposed)
      child_knowledge = np.random.choice(2, 1, p=bn_child_knowledge_distribution)
      informant_knowledge = np.random.choice(2, 1, p=informant_knowledge_distribution)

      #The Bayesian Network is part of the intrinsic environment.
      #Bayesian Network - Conditional Probability Table
      #child_knowledge: 0=non-knowledgeable, 1=knowledgeable
      #informant_knowledge: 0=non-knowledgeable, 1=knowledgeable
      #action: 0=reject, 1=accept
      #informant_intention: 0=reject, 1=accept
      if(child_knowledge==1 and informant_knowledge==1 and action==1 and informant_intention==1): cost = -1.0 # (knowledge, knowledge, accept, accept) = low_cost
      elif(child_knowledge==1 and informant_knowledge==1 and action==0 and informant_intention==1): cost = -1.0 # (knowledge, knowledge, reject, accept) = low_cost
      elif(child_knowledge==1 and informant_knowledge==1 and action==1 and informant_intention==0): cost = -1.0 # (knowledge, knowledge, accept, reject) = low_cost
      elif(child_knowledge==1 and informant_knowledge==1 and action==0 and informant_intention==0): cost = -1.0 # (knowledge, knowledge, reject, reject) = low_cost
      elif(child_knowledge==0 and informant_knowledge==1 and action==1 and informant_intention==1): cost = -1.0 # (non-knowledge, knowledge, accept, accept) = low_cost
      elif(child_knowledge==0 and informant_knowledge==1 and action==0 and informant_intention==0): cost = -1.0 # (non-knowledge, knowledge, reject, reject) = low_cost
      elif(child_knowledge==0 and informant_knowledge==1 and action==0 and informant_intention==1): cost = +1.0 # (non-knowledge, knowledge, reject, accept) = high_cost
      elif(child_knowledge==0 and informant_knowledge==1 and action==1 and informant_intention==0): cost = +1.0 # (non-knowledge, knowledge, accept, reject) = high_cost
      elif(child_knowledge==1 and informant_knowledge==0 and action==1 and informant_intention==1): cost = -1.0 # (knowledge, non-knowledge, accept, accept) = low_cost
      elif(child_knowledge==1 and informant_knowledge==0 and action==0 and informant_intention==1): cost = +1.0 # (knowledge, non-knowledge, reject, accept) = high_cost
      elif(child_knowledge==1 and informant_knowledge==0 and action==1 and informant_intention==0): cost = +1.0 # (knowledge, non-knowledge, accept, reject) = high_cost
      elif(child_knowledge==1 and informant_knowledge==0 and action==0 and informant_intention==0): cost = +1.0 # (knowledge, non-knowledge, reject, reject) = low_cost
      elif(child_knowledge==0 and informant_knowledge==0 and action==1 and informant_intention==1): cost = +1.0 # (non-knowledge, non-knowledge, accept, accept) = high_cost
      elif(child_knowledge==0 and informant_knowledge==0 and action==0 and informant_intention==1): cost = +1.0 # (non-knowledge, non-knowledge, reject, accept) = high_cost
      elif(child_knowledge==0 and informant_knowledge==0 and action==1 and informant_intention==0): cost = +1.0 # (non-knowledge, non-knowledge, accept, reject) = high_cost
      elif(child_knowledge==0 and informant_knowledge==0 and action==0 and informant_intention==0): cost = +1.0 # (non-knowledge, non-knowledge, reject, reject) = high_cost
      else: raise ValueError("ERROR: the Bayesian Networks input values are out of range")


      #Updating the critic using Temporal Differencing Learning
      #and switching to next state based on the action taken
      u_t = critic_vector[0, col] #previous state
      if(cost == 1.0):
          #u_t1 = critic_vector[0, np.random.randint(tot_images*tot_labels)] #random state
          u_t1 = -1.0
      elif(cost == -1.0):
          #u_t1 = u_t
          u_t1 = +1.0
      else:
          raise ValueError("ERROR: cost value is out of range")
      delta = (reward - cost) + (gamma*u_t1) - u_t
      critic_vector[0, col] += learning_rate*delta


      #Update the ACTOR using the delta
      actor_matrix[action, col] += delta

      print("")
      print("===========================")
      print("Episode: " + str(counter))
      print("Child action distribution: " + str(action_distribution))
      print("Child knowledge distribution: " + str(bn_child_knowledge_distribution))
      print("Child knowledge: " + str(child_knowledge))
      print("Informant index: " + str(informant_index))
      print("Informant knowledge: " + str(informant_knowledge))
      print("Informant knowledge distribution: " + str(informant_knowledge_distribution))
      print("")
      print("critic vector: " + str(critic_vector))
      print("")
      print("actor_matrix: " + str(actor_matrix))
      print("")
      print("informant_vector: " + str(informant_vector))
      counter += 1

    return actor_matrix, critic_vector, informant_vector

def main():  

    #- Building the CRITIC
    #The critic is a vector of (tot_images * tot_labels)
    tot_images = 3
    tot_labels = 3
    tot_actions = 2
    critic_vector = np.zeros((1, tot_images*tot_labels))
    #- Building the ACTOR
    #The actor is a matrix of  tot_actions * (tot_images * tot_labels)
    actor_matrix = np.zeros((tot_actions, tot_images*tot_labels))
    #- Dictionary of objects
    dict_images = {'CUP': 0, 'BOOK': 1, 'BALL': 2}
    dict_labels = {'cup': 0, 'book': 1, 'ball': 2}
    #- Reliability vector of the three informants
    #index: 0=caregiver, 1=reliable, 2=unreliable
    informant_vector = np.array([[1, 100], [1, 1], [1, 1]])

    #1- IMPRINTING: a caregiver gives labels to unknown objects.
    #The agent learns the name of the objects presented
    #The dataset contains tuple: (image,label,is_user_reliable)
    print("####### IMPRINTING ########")
    dataset_imprinting = [(dict_images['CUP'], dict_labels['cup'], 0, 1), (dict_images['BOOK'], dict_labels['book'], 0, 1),  
                          (dict_images['CUP'], dict_labels['cup'], 0, 1), (dict_images['BOOK'], dict_labels['book'], 0, 1), 
                          (dict_images['CUP'], dict_labels['cup'], 0, 1), (dict_images['BOOK'], dict_labels['book'], 0, 1),
                          (dict_images['CUP'], dict_labels['cup'], 0, 1), (dict_images['BOOK'], dict_labels['book'], 0, 1)]

    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)

    dataset_imprinting = [(dict_images['CUP'], dict_labels['book'], 0, 0), (dict_images['BOOK'], dict_labels['cup'], 0, 0),  
                          (dict_images['CUP'], dict_labels['book'], 0, 0), (dict_images['BOOK'], dict_labels['cup'], 0, 0),
                          (dict_images['CUP'], dict_labels['book'], 0, 0), (dict_images['BOOK'], dict_labels['cup'], 0, 0),
                          (dict_images['CUP'], dict_labels['book'], 0, 0), (dict_images['BOOK'], dict_labels['cup'], 0, 0), 
                          (dict_images['CUP'], dict_labels['book'], 0, 0), (dict_images['BOOK'], dict_labels['cup'], 0, 0)]

    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)

    #2- FAMILIARISATION: a set of known objects is presented
    #The reliable informant always gives the correct label
    #The unreliable informant always gives the wrong label
    print("####### FAMILIARISATION ########")
    dataset_familiarisation = [(dict_images['CUP'], dict_labels['cup'], 1, 1), (dict_images['CUP'], dict_labels['book'], 2, 1),
                               (dict_images['CUP'], dict_labels['cup'], 1, 1), (dict_images['CUP'], dict_labels['book'], 2, 1),
                               (dict_images['BOOK'], dict_labels['book'], 1, 1), (dict_images['BOOK'], dict_labels['cup'], 2, 1),
                               (dict_images['BOOK'], dict_labels['book'], 1, 1), (dict_images['BOOK'], dict_labels['cup'], 2, 1)]
  
    actor_matrix, critic_vector, informant_vector = training(dataset_familiarisation, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)

    #3- DECISION MAKING: new object presented.
    #The two informants give different labels.
    print("####### DECISION MAKING ########")
    dataset_decision = [(dict_images['BALL'], dict_labels['ball'], 1, 1), (dict_images['BALL'], dict_labels['book'], 2, 1),
                        (dict_images['BALL'], dict_labels['ball'], 1, 1), (dict_images['BALL'], dict_labels['book'], 2, 1)]
    actor_matrix, critic_vector, informant_vector = training(dataset_decision, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)

    #4- ASK TRIAL
    print("")
    print("####### ASK TEST ########")
    #The experimenter ask to the agent the name of the object
    #child_answer_distribution = critic_vector[0,6:]
    #This is the equivalent of setting to 1 the unit ACCEPT of the action layer of the SOM
    #And to activate the BMU of the Visual SOM. The computation returns the argmax.
    child_answer_distribution = actor_matrix[1,6:] #second row (accept) and columns for BALL
    print("Child answer distribution: " + str(child_answer_distribution))
    child_answer_distribution = softmax(child_answer_distribution)
    print("Child answer softmax: " + str(child_answer_distribution))
    child_answer = np.random.choice(3, 1, p=child_answer_distribution)
    #if(child_answer_distribution[1] == child_answer_distribution[2]):
        #child_answer = np.random.randint(1, 3)
    #else:
        #child_answer = np.argmax(child_answer_distribution)
    if(child_answer==0): print("Child answer: cup")
    elif(child_answer==1): print("Child answer: book")
    elif(child_answer==2): print("Child answer: ball")





if __name__ == "__main__":
    main()
