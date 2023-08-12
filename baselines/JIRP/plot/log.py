import numpy as np
import matplotlib.pylab as plt
import pickle as pk
from copy import copy, deepcopy
from pathlib import Path
import os 
import csv

class Log_experiments:
    def __init__(self, lp = 100, ep = 100):
        self._episode_data = {'reward': [], 'success': [], 'steps': [], 'episode': []}
        self._learning_period = lp 
        self._eval_period = ep
        self._recent_success_rate = 0

    def log_episode (self, reward, success, steps):
        self._episode_data['reward'].append(int(reward))
        self._episode_data['success'].append(int(success))
        self._episode_data['steps'].append(steps)
        if len(self._episode_data['episode']) == 0: self._episode_data['episode'].append(1)
        else: 
            last = self._episode_data['episode'][-1] 
            self._episode_data['episode'].append(last + 1)

    def recent_success_rate (self, last):
        size = len(self._episode_data['success'])
        if last < size: x = size - last
        else: x = 0
        success = self._episode_data['success'][x: size]
        succ = 0
        for s in success:
            if s == 1: succ += 1
        self._recent_success_rate = round( succ/len(success), 3)
        return self._recent_success_rate 

    def plot_learning (self, moving_number, param):
        y = self._episode_data[param]
        x = self._episode_data['episode']
        x_m = []
        y_m = []
        for i in range (moving_number, len(x)):
            sum_temp = 0
            for j in range (i - moving_number, i):
                sum_temp += y[j]
            sum_temp /= moving_number
            y_m.append(sum_temp)
            x_m.append(i)
        plt.plot (x_m,y_m)
        plt.show()
       
    def save_execution(self, filepath):
        with open(filepath, "wb") as output_file:
            pk.dump(self._episode_data, output_file)
        output_file.close()

