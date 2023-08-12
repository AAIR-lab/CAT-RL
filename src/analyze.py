import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
import os

def moving_average (y, moving_number):
    y_m = []
    for i in range (moving_number, len(y)):
        sum_temp = 0
        for j in range (i - moving_number, i):
            sum_temp += y[j]
        sum_temp /= moving_number
        y_m.append(sum_temp)
    return y_m

def read_data (map_name, approach_name, exp_number):
    data = []
    p = os.getcwd()
    for i in range (exp_number):
        path = p + "/results/" + map_name + "_" + approach_name + "_" + str(i + 1)
        with open(path, "rb") as file:
            data.append( pk.load(file) )
    return data

def smooth_data(data, moving_number, param):
    for i in range (len(data)):
        data[i][param] = moving_average(data[i][param], moving_number)
    return data

def prepare_avg_bound (param, data):
    epi_number = len(data[0][param])
    output = np.zeros ((3,epi_number))
    for i in range (epi_number):
        temp = []
        for j in range (len(data)):
            data_param = data[j][param]
            temp.append( data_param[i] )
        temp = np.array(temp)
        data_avg = np.average(temp)
        data_std = np.std(temp)
        output[0][i] = data_avg + data_std
        output[1][i] = data_avg
        output[2][i] = data_avg - data_std
    return output

def gen_episode_ax (a, b):
    episodes = []
    for i in range (a,b + a):
        episodes.append(i)
    return episodes

def compare_with_bounds(methods, map_name, exp_number, param, moving_number):
    colors = {'adrl-non': ['#45AA99','#D0E9E5'], 'adrl': ['#f56f0f','#edb68e'],'ppo': ['#332288','#CBC7E1'], 'hrl': ['#CD6A7B','#F2D8DC'], 'jirp': ['#BFB113','#FFF8A7']}
    for m in methods:
        data_temp = read_data (map_name, m, exp_number)
        plot_data_smooth = smooth_data(data_temp, moving_number, param)
        plot_data = prepare_avg_bound (param, plot_data_smooth)
        episodes = gen_episode_ax(moving_number, len(plot_data[1]))
        plt.plot(episodes, plot_data[1], color=colors[m][0], linestyle='solid', linewidth = 1, label = m)
        plt.fill_between(episodes, plot_data[0], plot_data[2], alpha=0.5, edgecolor=colors[m][1], facecolor=colors[m][1],linewidth=0)
        
    plt.xlabel("episodes")
    plt.ylabel(param)
    plt.title(map_name)
    plt.legend()
    plt.savefig("cc.svg")
    plt.show()

def read_data_single (map_name, approach_name, exp_number):
    data = []
    path = os.getcwd() + "/results/" + map_name + "_" + approach_name + "_" + str(exp_number)
    with open(path, "rb") as file:
        data.append( pk.load(file) )
    return data

def plot_single (map_name, approach_name, exp_number, moving_number, param):
    data = read_data_single(map_name, approach_name, exp_number)
    y = data[0][param]
    x = data[0]['episode']
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

map = "Mountain_Car"
methods = ['adrl']
exp_number = 10
smoothing = 100

#compare_with_bounds(methods, map, exp_number, 'reward', smoothing)

compare_with_bounds(methods, map, exp_number, 'success', smoothing)

#compare_with_bounds(methods, map, exp_number, 'steps', smoothing)

#plot_single ('Mountain_Car', 'adrl', 6, smoothing, 'success')
