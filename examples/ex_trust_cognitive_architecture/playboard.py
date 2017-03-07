#!/usr/bin/env python

#MIT License
#Copyright (c) 2017 Massimiliano Patacchiola
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

#Class for creating a playboard for the trust-based reinforcement learning
#experiment with a humanoid robot. The playboard is a NxM matrix which may
#contain a ball and a cup. The robot eye can focus on a single cell and it
#can move the eye in 4 directions (up, down, left, right). The robot can
#decide to grasp the object or to release a previous object behind the eye.
#The eye can detect the presence of the ball and of the cup.
#A positive reward is given if the ball is released into the cup.
#A negative reward is given if the ball is released outside the play area.


import numpy as np

class PlayBoard:

    def __init__(self, tot_row, tot_col, tot_trap):
        ''' Create the PlayBoard

        @param tot_row is the total number of rows
        @param tot_col is the total number of columns
        @param tot_traps define the number of traps to put in the world
        '''
        self.action_space_size = 4 + 2 # (up, down, left, right) + (grasp, release)
        self.world_row = tot_row
        self.world_col = tot_col
        if(tot_trap+2 > tot_row or tot_trap+2 > tot_col):
            raise ValueError('The number of traps is too high for a word of this size.')
        else:
            self.tot_trap = tot_trap

    def render(self):
        ''' Print the current world in the terminal.

        O represents the robot position
        - respresent empty states.
        # represents obstacles
        * represents terminal states
        '''
        graph = ""
        for row in range(self.world_row):
            row_string = ""
            for col in range(self.world_col):
                if(self.state_matrix[row, col, 1]==+1 and self.state_matrix[row, col, 0]==0): row_string += ' {-} ' #eye
                elif(self.state_matrix[row, col, 1]==+1 and self.state_matrix[row, col, 0]==+1): row_string += ' {#} ' #trap
                elif(self.state_matrix[row, col, 1]==+1 and self.state_matrix[row, col, 0]==+2): row_string += ' {*} ' #cup
                elif(self.state_matrix[row, col, 1]==+1 and self.state_matrix[row, col, 0]==+3): row_string += ' {O} ' #ball
                elif(self.state_matrix[row, col, 1]==0 and self.state_matrix[row, col, 0]==+1): row_string += '  #  ' #trap
                elif(self.state_matrix[row, col, 1]==0 and self.state_matrix[row, col, 0]==+2): row_string += '  *  ' #cup
                elif(self.state_matrix[row, col, 1]==0 and self.state_matrix[row, col, 0]==+3 and self.is_grasping==False): row_string += '  O  ' #ball
                else: row_string += '  -  ' #empty
            row_string += '\n'
            row_string += '\n'
            graph += row_string 
        print graph            

    def reset(self, exploring_starts=False):
        ''' Set a random position for the eye, the ball, the traps and the cup.

        @param exploring_starts if True the eye is set randomly otherwise in (0,0)
        @return eye_position, object_observed, is_grasping
        '''
        self.state_matrix = np.zeros((self.world_row, self.world_col, 3))

        if exploring_starts:
            self.eye_position = [np.random.randint(self.world_row), np.random.randint(self.world_col)] #place the eye
        else:
            self.eye_position = [0, 0]

        #To guarantee that the obstacles, the ball and the cup are not
        #overlapping it is necessary to pick elements without replacement
        #This is possible with numpy.random.choice(replace=False)
        row_array = np.random.choice(self.world_row, size=self.tot_trap+2, replace=False)
        col_array = np.random.choice(self.world_col, size=self.tot_trap+2, replace=False)

        #Set all the total traps 
        for i in range(self.tot_trap):
            self.state_matrix[row_array[i], col_array[i], 0] = +1 #First trap
      
        #Set the ball and the cup position
        self.ball_position = [row_array[self.tot_trap], col_array[self.tot_trap]] #the ball position can be changed
        self.cup_position = [row_array[self.tot_trap+1], col_array[self.tot_trap+1]] #the cup position stay the same

        #Set the layers of the state matrix
        self.state_matrix[self.cup_position[0], self.cup_position[1], 0] = +2
        self.state_matrix[self.ball_position[0], self.ball_position[1], 0] = +3
        self.state_matrix[self.eye_position[0], self.eye_position[1], 1] = 1
        self.is_grasping = False

        #It returns the eye_position: row,col; and the trap/cup behind the eye: -1, +1
        return self.eye_position, int(self.state_matrix[self.eye_position[0], self.eye_position[1], 0]), self.is_grasping

    def step(self, action):
        ''' One step in the world.

        [eye_position, object_observed, is_grasping, reward, done = env.step(action)]
        The robot moves one step in the world based on the action given.
        @param action is the action to execute
          0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, 4=GRASP, 5=RELEASE
        @return observation the position of the eye after the step
        @return boolean indicating if the robot is grasping the ball
        @return reward the reward associated with the next state
        @return done True if the state is terminal  
        '''
        if(action >= self.action_space_size): 
            raise ValueError('The action is not included in the action space.')

        #Generating a new position based on the current position and action
        #Action UP
        if(action == 0): 
            new_position = [self.eye_position[0]-1, self.eye_position[1]]
        #Action RIGHT
        elif(action == 1): 
            new_position = [self.eye_position[0], self.eye_position[1]+1]
        #Action DOWN
        elif(action == 2): 
            new_position = [self.eye_position[0]+1, self.eye_position[1]]
        #Action LEFT
        elif(action == 3): 
            new_position = [self.eye_position[0], self.eye_position[1]-1]
        #Action GRASP
        elif(action == 4):
            if(self.is_grasping==False and self.eye_position==self.ball_position):
                self.is_grasping=True
                return self.eye_position, int(self.state_matrix[self.eye_position[0], self.eye_position[1], 0]), self.is_grasping, +0.1, False
            elif(self.is_grasping==False and self.eye_position!=self.ball_position): 
                return self.eye_position, int(self.state_matrix[self.eye_position[0], self.eye_position[1], 0]), False, -0.001, False
            elif(self.is_grasping==True): 
                self.is_grasping=False
                return self.eye_position, int(self.state_matrix[self.eye_position[0], self.eye_position[1], 0]), self.is_grasping, -1.0, True #Terminal state (cannot grasp if grasping)
        #Action RELEASE
        elif(action == 5):
            if(self.is_grasping==False):
                return self.eye_position, int(self.state_matrix[self.eye_position[0], self.eye_position[1], 0]), False, -0.001, False #nothing happend
            if(self.is_grasping==True and int(self.state_matrix[self.eye_position[0], self.eye_position[1], 0])==+2): 
                return self.eye_position, int(self.state_matrix[self.eye_position[0], self.eye_position[1], 0]), False, +1.0, True #released on cup
            if(self.is_grasping==True and int(self.state_matrix[self.eye_position[0], self.eye_position[1], 0])==+1): 
                return self.eye_position, int(self.state_matrix[self.eye_position[0], self.eye_position[1], 0]), False, -1.0, True #released on trap
            if(self.is_grasping==True and int(self.state_matrix[self.eye_position[0], self.eye_position[1], 0])==0):
                self.state_matrix[self.ball_position[0], self.ball_position[1], 0] = 0 #reset old position of ball
                self.ball_position = [self.eye_position[0], self.eye_position[1]] #assign new position to ball
                self.state_matrix[self.ball_position[0], self.ball_position[1], 0] = +3 #set new position in state matrix
                return self.eye_position, int(self.state_matrix[self.eye_position[0], self.eye_position[1], 0]), False, -0.001, False #released on empty
            if(self.is_grasping==True and int(self.state_matrix[self.eye_position[0], self.eye_position[1], 0])==+3):
                return self.eye_position, int(self.state_matrix[self.eye_position[0], self.eye_position[1], 0]), False, -0.001, False #released on previous ball position

        #If the action is MOVE EYE it arrives here
        #Check if the new position is a valid position
        if (new_position[0]>=0 and new_position[0]<self.world_row):
            if(new_position[1]>=0 and new_position[1]<self.world_col):
                    self.state_matrix[self.eye_position[0], self.eye_position[1], 1] = 0 #reset old position of eye
                    self.eye_position = new_position
                    self.state_matrix[self.eye_position[0], self.eye_position[1], 1] = 1 #set new position of eye
        return self.eye_position, int(self.state_matrix[self.eye_position[0], self.eye_position[1], 0]), self.is_grasping, -0.001, False





