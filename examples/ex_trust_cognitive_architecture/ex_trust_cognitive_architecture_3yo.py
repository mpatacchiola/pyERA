#!/usr/bin/python

## Massimiliano Patacchiola, Plymouth University 2016

import numpy as np
import os 
import time

#From Harris et al. (Preschoolers Mistrust Ignorant and Inaccurate Speakers)
#The 3yo children can discriminate the reliable from the unreliable informant,
#meaning that their reputation counter is working properly. What they miss is
#the ability of using reputation information when deciding whom to trust for
#future information.

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
#This mechanism can be considered part of a planning module (e.g. prefrontal cortex) with inibitory projections.
#The cost function C can be defined as a function that takes as input: current_state, agent_action, agent_confidence, informant_action, informant_reputation.
#The cost function evaluates what's the cost of having taken an action in state S given the informant advice .
#The output of the function C is a real number representing the COST of taking that action given the informant suggestion.
#This table can be represented through a table or can be approximated through a function approximator (e.g. neural network)
#
#The actor architecture is a table of state-action pairs. 
#When the child has to give a label for an object the policy must be used and not the utility table.
#The most common associated label to a visual object can be estimated setting the SOM action node to ACCEPT and then
#computing the activation of the vocabulary unit. The argmax is the value we want.

def return_best_choice(answer_distribution):
    #child_answer = np.random.randint(dict_images['LOMA'], dict_images['MIDO']+1)
 
    tot_images = np.amax(answer_distribution.shape)
    argmax_index = np.argmax(answer_distribution)    
    argmax_value = np.amax(answer_distribution)
    counter = 0
    index_list = list()
    for n in answer_distribution:
        if(n == argmax_value): 
            index_list.append(counter)
        counter += 1

    if(len(index_list) == 1): return int(index_list[0])
    #elif(len(index_list) > 1): return int(np.random.choice(tot_images, 1, p=answer_distribution))
    else: return int(np.random.choice(index_list, 1)) #uniform sampling

def softmax(x):
    '''Compute softmax values of array x.

    @param x the input array
    @return the softmax array
    '''
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

def training(dataset, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions):
    '''Train the actor using Intrinsic Motivated Reinforcement Learning

    General Algorithm Description:
    1- To the agent is presented an object and a label (current state).
    2- An informant suggest a possible action (accept or reject the label).
    3- The agent take an action (with softmax) considering is current state-action table
    4- (External) New state and reward obtained from the environment
    5- (Intrinsic) The informant_reputation is updated through MLE: agent_action, agent_confidence, informant_action, reward
    6- (Intrinsic) The Cost is estimated: current_state, agent_action, agent_confidence, informant_action, informant_reputation
    7- The utility table is updated using: previous_state, current_state, cost, reward
    8- The actor table is updated using the delta from the critic
    '''
    #Hyper-Parameters
    reward = 0
    gamma = 1.0 #no gamma used in thsi example
    learning_rate = 0.1

    counter = 1
    for episode in dataset:

      #1- To the agent is presented an object and a label (current state).
      image = episode[0] #image of the object
      label = episode[1] #label given by the informant
      informant_index = episode[2] #a integer representing the informant
      informant_action = episode[3] #0=reject, 1=accept

      #3- The agent take an action (with softmax) considering is current state-action table
      #[0=cup, 1=book, 2=ball]
      col = (image * tot_images) + label
      action_array = actor_matrix[:, col]
      action_distribution = softmax(action_array)
      child_action = np.random.choice(tot_actions, 1, p=action_distribution) #select the action through softmax

      #4- (External) New state and reward obtained from the environment
      u_t = critic_vector[0, col] #previous state
      #New state is estimated, in this simple case nothing happen
      #because the next state is terminal
      u_t1 = u_t

      #5- (Intrinsic) The informant_reputation is updated: agent_action, agent_confidence, informant_action, reward
      #informant_vector: 0=unreliable, 1=reliable
      #do_actions_agree: False, True
      #Estimating child_confidence
      distance = np.absolute(action_distribution[0] - action_distribution[1])
      child_confidence_distribution = [1-distance, distance] #non-knowledgeable, knowledgeable
      child_confidence = np.random.choice(2, 1, p=child_confidence_distribution)
      #if(distance == 0):
          #child_confidence=0
          #child_confidence_distribution = [1, 0]
      #else: 
          #child_confidence=1
          #child_confidence_distribution = [0, 1]

      #Check if child and informant agree
      if(child_action == informant_action): do_actions_agree = True
      else: do_actions_agree = False
      #Increment the counter in the informant_vector
      if(do_actions_agree==False and child_confidence==1):
          informant_vector[informant_index][0] += 1 #unreliable
      elif(do_actions_agree==True and child_confidence==1):
          informant_vector[informant_index][1] += 1 #reliable
      elif(do_actions_agree==False and child_confidence==0):
          informant_vector[informant_index][1] += 1 #reliable
          informant_vector[informant_index][0] += 1 #unreliable
      elif(do_actions_agree==True and child_confidence==0):
          informant_vector[informant_index][1] += 1 #reliable
          informant_vector[informant_index][0] += 1 #unreliable
      else: 
          raise ValueError("ERROR: anomaly in the IF condition for informant_vector update")
      #Using the informant_vector given as input it estimates the reputation of the informant
      informant_reputation_distribution = np.true_divide(informant_vector[informant_index], np.sum(informant_vector[informant_index]))
      #informant_reputation = np.random.choice(2, 1, p=informant_reputation_distribution)
      informant_reputation = np.argmax(informant_reputation_distribution)

      #6- (Intrinsic) The Cost is estimated: current_state, agent_action, agent_confidence, informant_action, informant_reputation
      #child_confidence: 0=non-knowledgeable, 1=knowledgeable
      #informant_reputation: 0=non-knowledgeable, 1=knowledgeable
      #action: 0=reject, 1=accept
      #informant_action: 0=reject, 1=accept
      if(child_confidence==1 and informant_reputation==1 and child_action==1 and informant_action==1): cost = -1.0 # (knowledge, knowledge, accept, accept) = low_cost
      elif(child_confidence==1 and informant_reputation==1 and child_action==0 and informant_action==1): cost = -1.0 # (knowledge, knowledge, reject, accept) = low_cost
      elif(child_confidence==1 and informant_reputation==1 and child_action==1 and informant_action==0): cost = -1.0 # (knowledge, knowledge, accept, reject) = low_cost
      elif(child_confidence==1 and informant_reputation==1 and child_action==0 and informant_action==0): cost = -1.0 # (knowledge, knowledge, reject, reject) = low_cost
      elif(child_confidence==0 and informant_reputation==1 and child_action==1 and informant_action==1): cost = -1.0 # (non-knowledge, knowledge, accept, accept) = low_cost
      elif(child_confidence==0 and informant_reputation==1 and child_action==0 and informant_action==0): cost = -1.0 # (non-knowledge, knowledge, reject, reject) = low_cost
      elif(child_confidence==0 and informant_reputation==1 and child_action==0 and informant_action==1): cost = +1.0 # (non-knowledge, knowledge, reject, accept) = high_cost
      elif(child_confidence==0 and informant_reputation==1 and child_action==1 and informant_action==0): cost = +1.0 # (non-knowledge, knowledge, accept, reject) = high_cost
      elif(child_confidence==1 and informant_reputation==0 and child_action==1 and informant_action==1): cost = -1.0 # (knowledge, non-knowledge, accept, accept) = low_cost
      elif(child_confidence==1 and informant_reputation==0 and child_action==0 and informant_action==1): cost = -1.0 # (knowledge, non-knowledge, reject, accept) = high_cost
      elif(child_confidence==1 and informant_reputation==0 and child_action==1 and informant_action==0): cost = -1.0 # (knowledge, non-knowledge, accept, reject) = high_cost
      elif(child_confidence==1 and informant_reputation==0 and child_action==0 and informant_action==0): cost = -1.0 # (knowledge, non-knowledge, reject, reject) = low_cost
      elif(child_confidence==0 and informant_reputation==0 and child_action==1 and informant_action==1): cost = -1.0 # (non-knowledge, non-knowledge, accept, accept) = high_cost
      elif(child_confidence==0 and informant_reputation==0 and child_action==0 and informant_action==1): cost = +1.0 # (non-knowledge, non-knowledge, reject, accept) = high_cost
      elif(child_confidence==0 and informant_reputation==0 and child_action==1 and informant_action==0): cost = +1.0 # (non-knowledge, non-knowledge, accept, reject) = high_cost
      elif(child_confidence==0 and informant_reputation==0 and child_action==0 and informant_action==0): cost = -1.0 # (non-knowledge, non-knowledge, reject, reject) = high_cost
      else: raise ValueError("ERROR: the Bayesian Networks input values are out of range")

      #7- The utility table is updated using: preious_state, current_state, cost, reward
      #Updating the critic using Temporal Differencing Learning
      #In this simple case there is not a u_t1 state.
      #The current state is considered terminal.
      #We can delete the term (gamma*u_t1)-u_t and considering 
      #only (reward-cost) as utility of the state (you can cite Russel Norvig).
      delta = (reward - cost) + (gamma*u_t1) - u_t
      critic_vector[0, col] += learning_rate*delta

      #8- The actor table is updated using the delta from the critic
      #Update the ACTOR using the delta
      actor_matrix[child_action, col] += learning_rate*delta #the current action
      actor_matrix[1-child_action, col] -= learning_rate*delta #the opposite action

      print("")
      print("===========================")
      print("Episode: " + str(counter))
      print("Image: " + str(image) + "; Label: " + str(label))
      print("Child action distribution: " + str(action_distribution))
      print("Child action: " + str(child_action))
      print("Child knowledge distribution: " + str(child_confidence_distribution))
      print("Child knowledge: " + str(child_confidence))
      print("Informant index: " + str(informant_index))
      print("Informant action: " + str(informant_action))
      print("Informant knowledge: " + str(informant_reputation))
      print("Informant knowledge distribution: " + str(informant_reputation_distribution))
      print("Cost: " + str(cost))
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
    tot_images = 12
    tot_labels = 12
    tot_actions = 2
    critic_vector = np.zeros((1, tot_images*tot_labels))
    #- Building the ACTOR
    #The actor is a matrix of  tot_actions * (tot_images * tot_labels)
    actor_matrix = np.zeros((tot_actions, tot_images*tot_labels))
    #- Dictionary of objects
    dict_images = {'CUP': 0, 'BOOK': 1, 'BALL': 2, 'SHOE': 3, 'DOG': 4, 'CHAIR': 5, 'LOMA': 6, 'MIDO': 7, 'WUG': 8, 'DAX': 9, 'BLICKET': 10, 'DAWNOO': 11}
    dict_labels = {'cup': 0, 'book': 1, 'ball': 2, 'shoe': 3, 'dog': 4, 'chair': 5, 'loma': 6, 'mido': 7, 'wug': 8, 'dax': 9, 'blicket': 10, 'dawnoo': 11}
    #- Reliability vector of the three informants
    #index: 0=caregiver, 1=reliable, 2=unreliable
    informant_vector = np.array([[1, 100], [1, 1], [1, 1]])

    #1- IMPRINTING: a caregiver gives labels to unknown objects.
    #NOTE: hee to decide how many times the training should be executed maybe
    #it is possible to back-engineering the results from HArris et a.
    #If the Children tested for known objects answer correctly in 70 percent of the cases,
    #then with a grid-search it is possible to find the number of times we should run the learning step
    #in order to have the same results.
    #
    #The agent learns the name of the objects presented
    #The dataset contains tuple: (image,label,is_user_reliable)
    print("####### IMPRINTING ########")
    dataset_imprinting = [(dict_images['CUP'], dict_labels['cup'], 0, 1),
                          (dict_images['BOOK'], dict_labels['book'], 0, 1),  
                          (dict_images['BALL'], dict_labels['ball'], 0, 1),  
                          (dict_images['SHOE'], dict_labels['shoe'], 0, 1),
                          (dict_images['DOG'], dict_labels['dog'], 0, 1),
                          (dict_images['CHAIR'], dict_labels['chair'], 0, 1)]

    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)
    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)
    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)
    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)
    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)
    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)
    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)
    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)
    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)
    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)

    dataset_imprinting = [(dict_images['CUP'], dict_labels['book'], 0, 0),
                          (dict_images['CUP'], dict_labels['ball'], 0, 0),
                          (dict_images['CUP'], dict_labels['shoe'], 0, 0),
                          (dict_images['CUP'], dict_labels['dog'], 0, 0),
                          (dict_images['CUP'], dict_labels['chair'], 0, 0),
                          (dict_images['BOOK'], dict_labels['cup'], 0, 0),  
                          (dict_images['BOOK'], dict_labels['ball'], 0, 0),
                          (dict_images['BOOK'], dict_labels['shoe'], 0, 0),  
                          (dict_images['BOOK'], dict_labels['dog'], 0, 0),
                          (dict_images['BOOK'], dict_labels['chair'], 0, 0),
                          (dict_images['BALL'], dict_labels['cup'], 0, 0),  
                          (dict_images['BALL'], dict_labels['book'], 0, 0),
                          (dict_images['BALL'], dict_labels['shoe'], 0, 0),  
                          (dict_images['BALL'], dict_labels['dog'], 0, 0),
                          (dict_images['BALL'], dict_labels['chair'], 0, 0),  
                          (dict_images['SHOE'], dict_labels['cup'], 0, 0),  
                          (dict_images['SHOE'], dict_labels['book'], 0, 0),
                          (dict_images['SHOE'], dict_labels['ball'], 0, 0),  
                          (dict_images['SHOE'], dict_labels['dog'], 0, 0),
                          (dict_images['SHOE'], dict_labels['chair'], 0, 0),  
                          (dict_images['DOG'], dict_labels['cup'], 0, 0),  
                          (dict_images['DOG'], dict_labels['book'], 0, 0),
                          (dict_images['DOG'], dict_labels['ball'], 0, 0),  
                          (dict_images['DOG'], dict_labels['shoe'], 0, 0),
                          (dict_images['DOG'], dict_labels['chair'], 0, 0),
                          (dict_images['CHAIR'], dict_labels['cup'], 0, 0),  
                          (dict_images['CHAIR'], dict_labels['book'], 0, 0),
                          (dict_images['CHAIR'], dict_labels['ball'], 0, 0),  
                          (dict_images['CHAIR'], dict_labels['shoe'], 0, 0),
                          (dict_images['CHAIR'], dict_labels['dog'], 0, 0)]
    #Learn
    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)
    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)
    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)
    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)
    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)
    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)
    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)
    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)
    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)
    actor_matrix, critic_vector, informant_vector = training(dataset_imprinting, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)

    #2- FAMILIARISATION: a set of known objects is presented
    #The reliable informant always gives the correct label
    #The unreliable informant always gives the wrong label
    print("####### FAMILIARISATION ########")
    dataset_familiarisation = [(dict_images['BALL'], dict_labels['ball'], 1, 1), (dict_images['BALL'], dict_labels['shoe'], 2, 1),
                               (dict_images['CUP'], dict_labels['cup'], 1, 1), (dict_images['CUP'], dict_labels['dog'], 2, 1),
                               (dict_images['BOOK'], dict_labels['book'], 1, 1), (dict_images['BOOK'], dict_labels['chair'], 2, 1)]
  
    actor_matrix, critic_vector, informant_vector = training(dataset_familiarisation, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)


    #3- DECISION MAKING: new object presented.
    #The two informants give different labels.
    print("####### DECISION MAKING ########")
    dataset_decision = [(dict_images['MIDO'], dict_labels['mido'], 1, 1), (dict_images['MIDO'], dict_labels['loma'], 2, 1),
                        (dict_images['WUG'], dict_labels['wug'], 1, 1), (dict_images['WUG'], dict_labels['dax'], 2, 1),
                        (dict_images['BLICKET'], dict_labels['blicket'], 1, 1), (dict_images['BLICKET'], dict_labels['dawnoo'], 2, 1)]

    actor_matrix, critic_vector, informant_vector = training(dataset_decision, actor_matrix, critic_vector, informant_vector, tot_images, tot_labels, tot_actions)

    #4- ASK TRIAL
    print("")
    print("####### ASK TEST ########")
    print("")
    #The experimenter ask to the agent the name of the object
    #child_answer_distribution = critic_vector[0,6:]
    #This is the equivalent of setting to 1 the unit ACCEPT of the action layer of the SOM
    #And to activate the BMU of the Visual SOM. The computation returns the argmax.
    object_name_list = ['cup', 'book', 'ball', 'shoe', 'dog', 'chair', 'loma', 'mido', 'wug', 'dax', 'blicket', 'dawnoo']

    print("---- ASK MIDO ----")
    col_start = (dict_images['MIDO'] * tot_images)
    col_stop =  (dict_images['MIDO'] * tot_images) + tot_labels
    child_answer_distribution = actor_matrix[1,col_start:col_stop] #second row (accept) and columns for MIDO
    print("Object labels: " + str(object_name_list))
    print("Child answer distribution: " + str(child_answer_distribution))
    child_answer_distribution = softmax(child_answer_distribution)
    print("Child answer softmax: " + str(child_answer_distribution))
    child_answer = return_best_choice(child_answer_distribution)
    #'cup': 0, 'book': 1, 'ball': 2, 'shoe': 3, 'dog': 4, 'chair': 5, 'mido': 6
    print("Child answer: " + str(object_name_list[child_answer]))
    print("")

    print("---- ASK WUG ----")
    col_start = (dict_images['WUG'] * tot_images)
    col_stop =  (dict_images['WUG'] * tot_images) + tot_labels
    child_answer_distribution = actor_matrix[1,col_start:col_stop] #second row (accept) and columns for MIDO
    print("Object labels: " + str(object_name_list))
    print("Child answer distribution: " + str(child_answer_distribution))
    child_answer_distribution = softmax(child_answer_distribution)
    print("Child answer softmax: " + str(child_answer_distribution))
    child_answer = return_best_choice(child_answer_distribution)
    #'cup': 0, 'book': 1, 'ball': 2, 'shoe': 3, 'dog': 4, 'chair': 5, 'mido': 6
    print("Child answer: " + str(object_name_list[child_answer]))
    print("")

    print("---- ASK BLICKET ----")
    col_start = (dict_images['BLICKET'] * tot_images)
    col_stop =  (dict_images['BLICKET'] * tot_images) + tot_labels
    child_answer_distribution = actor_matrix[1,col_start:col_stop] #second row (accept) and columns for MIDO
    print("Object labels: " + str(object_name_list))
    print("Child answer distribution: " + str(child_answer_distribution))
    child_answer_distribution = softmax(child_answer_distribution)
    print("Child answer softmax: " + str(child_answer_distribution))
    child_answer = return_best_choice(child_answer_distribution)
    #'cup': 0, 'book': 1, 'ball': 2, 'shoe': 3, 'dog': 4, 'chair': 5, 'mido': 6
    print("Child answer: " + str(object_name_list[child_answer]))
    print("")

if __name__ == "__main__":
    main()
