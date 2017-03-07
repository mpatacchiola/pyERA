import numpy as np
from playboard import PlayBoard





def main():

    env = PlayBoard(tot_row=5, tot_col=5, tot_trap=3)
    tot_epoch = 100

    for epoch in range(tot_epoch):
        #Reset and return the first observation
        observation_eye, observation_object, observation_grasping = env.reset(exploring_starts=True)
        action_list = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'GRASP', 'RELEASE']
        object_list = ['EMPTY', 'TRAP', 'CUP', 'BALL']
        reward = 0
        done = False
        for step in range(1000):

            action = np.random.randint(6)

            print("Reward: " + str(reward))
            print("Eye: " + str(observation_eye))
            print("Object: " + str(object_list[observation_object]))
            print("Grasping: " + str(observation_grasping))
            print("Action: " + action_list[action])
            print("")
            env.render()

            #Move one step in the environment and get obs and reward
            observation_eye, observation_object, observation_grasping, reward, done = env.step(action)
            if done: break


if __name__ == "__main__":
    main()
