""""
This module contains the DRL class which is used for testing and saving history.
Attributes:
    None
Methods:
    __init__(): Initializes the DRL class and creates necessary directories if they don't exist.
    test(total_episode, filename): Tests the model and saves the testing history if it doesn't already exist.
    save_history(history, name): Saves the history as a CSV file.
"""
import os
import numpy as np
import pandas as pd

class DRL:
    def __init__(self):
        
        if not os.path.exists('model'):
            os.mkdir('model')

        if not os.path.exists('history'):
            os.mkdir('history')

        if not os.path.exists('plots'):
            os.mkdir('plots')
            
    def test(self, total_episode,filename): #ddpg.test()
        """hedge with model.
        """
        print('testing...')
        if os.path.exists('./history/'+'testing_history'+filename+'.csv'):
            print("Test File already exists, exiting function")
            return

        beta_discount = 0.02
        period_T = self.env.num_period - 1
        avg_reward = []
        avg_V_opt = []
        V_opt_store = []
        avg_action = np.empty((0,4))
        y_action_store = np.empty((0,4))       
        history_details = {
            "name": filename,
            "episode": [], 
            "period": [],
            "period_reward": [], 
            "y1": [],  
            "y2": [], 
            "z1": [],  
            "z2": [],  
            "s1": [], "s2": [],
            "pC": [], "p1": [], "p2": [],
            "cH": [], "cP": [],
            "violation_cm": [],
            "violation_s1": [],
            "violation_s2": []
            } # store in a dic.
        
        history = {
            "name": filename,
            "episode": [], "episode_w_T": [], 
            "avg_reward": [], "avg_reward_opt": [],
            "reward_gap": [], 
            "avg_y1": [],  
            "avg_y2": [],  
            "avg_z1": [],  
            "avg_z2": [], 
        } # store in a dic.
        self.epsilon = -1 # then the random choice option is off, the actor generate actions from learned policy  

        w_T_store = []

        for i in range(total_episode):
            observation, V_opt = self.env.reset()
            done = False
            action_store = []
            reward_store = []
            y_action_test = np.empty((0,4))
            pt = 1
            while not done:

                x = np.array(observation).reshape(1, -1)
                # choose action from epsilon-greedy; epsilon has been set to -1
                action, _, _, action_sampled = self.egreedy_action(x)

                action_store.append(action)
                y_action_test = np.vstack((y_action_test, action))

                observation, reward, done, info = self.env.step(x, action)
                reward_store.append(reward)
                history_details["episode"].append(i)
                history_details["period"].append(pt) 
                history_details["period_reward"].append(reward)
                history_details["pC"].append(observation[0])
                history_details["p2"].append(observation[1])
                history_details["p1"].append(observation[2])
                history_details["cH"].append(observation[3])
                history_details["cP"].append(observation[4])
                history_details["s1"].append(observation[5])
                history_details["s2"].append(observation[6])
                
                history_details["y1"].append(action[0])
                history_details["y2"].append(action[1])
                history_details["z1"].append(action[2])
                history_details["z2"].append(action[3])
                history_details["violation_cm"].append(action[0]+action[1]-98107-1)
                history_details["violation_s1"].append(action[0]+action[2]-x[0][5]-1)
                history_details["violation_s2"].append(action[1]+action[3]-x[0][6]-1)
                pt = pt + 1
                                
            reward_store = reward_store*((1-beta_discount)**np.arange(0,period_T))

            w_T = np.sum(reward_store) # equivalent to one episode's V_opt
            w_T_store.append(w_T) # Store each episodes V_opt
            
            y_action_store = np.vstack((y_action_store, np.mean(y_action_test, axis = 0)))
            
            V_opt_store = np.append(V_opt_store, V_opt)

            
            self.test_gap = 1
            if i % self.test_gap == 0:  
                history["episode"].append(i)
                history["episode_w_T"].append(w_T)
                episode10_avgvalue = np.mean(w_T_store[-self.test_gap:])
                episode10_avgaction = np.mean(y_action_store[-self.test_gap:], axis = 0)                
                episode10_avg_y1 = episode10_avgaction[0]
                episode10_avg_y2 = episode10_avgaction[1]
                episode10_avg_z1 = episode10_avgaction[2]
                episode10_avg_z2 = episode10_avgaction[3]              
                avg_reward.append(episode10_avgvalue)
                avg_action = np.vstack((avg_action, episode10_avgaction))               
                episode10_avg_V_opt = np.mean(V_opt_store[-self.test_gap:])
                avg_V_opt = np.append(avg_V_opt, episode10_avg_V_opt)
                episode10_reward_gap = (episode10_avg_V_opt - episode10_avgvalue)/episode10_avg_V_opt               
                history["avg_reward"].append(episode10_avgvalue)
                history["avg_reward_opt"].append(episode10_avg_V_opt)
                history["reward_gap"].append(episode10_reward_gap)                
                history["avg_y1"].append(episode10_avg_y1)
                history["avg_y2"].append(episode10_avg_y2)
                history["avg_z1"].append(episode10_avg_z1)
                history["avg_z2"].append(episode10_avg_z2)

        return history_details, history 
                    

    def save_history(self, history, name):
        name = os.path.join('history', name)

        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')