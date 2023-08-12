import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
from log import Log_experiments
import csv

def moving_average (y, moving_number):
    y_m = []
    for i in range (moving_number, len(y)):
        sum_temp = 0
        for j in range (i - moving_number, i):
            sum_temp += y[j]
        sum_temp /= moving_number
        y_m.append(sum_temp)
    return y_m

def read_data (map_name, approach_name, exp_number, dirpath):
    data = []
    for i in range (exp_number):
        # path = "./../plotdata/final/" + map_name + "_" + approach_name + "_" + str(i + 1)
        path = dirpath+"_" + approach_name + "_" + str(i + 1)
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

def compare_with_bounds(methods, map_name, exp_number, max_episode, param, moving_number, title, dirpath, filepath):
    colors = {'adrl': ['#45AA99','#D0E9E5'], 'q': ['#332288','#CBC7E1'], 'hrl': ['#CD6A7B','#F2D8DC'], 'jirp': ['#b38c32','#fcf5e3']}
    for m in methods:
        print("\mMethod:",m)
        data_temp = read_data (map_name, m, exp_number, dirpath)
        # only using data until max_episode
        data_temp[0]["success"] = data_temp[0]["success"][:max_episode]
        plot_data_smooth = smooth_data(data_temp, moving_number, param)
        plot_data = prepare_avg_bound (param, plot_data_smooth)
        episodes = gen_episode_ax(moving_number, len(plot_data[1]))
        plt.plot(episodes, plot_data[1], color=colors[m][0], linestyle='solid', linewidth = 1, label = m)
        plt.fill_between(episodes, plot_data[0], plot_data[2], alpha=0.5, edgecolor=colors[m][1], facecolor=colors[m][1],linewidth=0)
        
    plt.xlabel("episodes")
    plt.ylabel(param)
    plt.title(title)
    plt.legend()
    # plt.show()
    plt.savefig(filepath)



if __name__=="__main__":
    # map = 'taxi_30x30_map1'
    # map = "grid_64x64_map1"
    map = "office_36x36_map1"
    alg = "jirp"
    domain_name = "officeworld"
    dirpath = "./../plotdata/"+domain_name+"/"+map
    max_episode = 10000
    exp_number = 10
    # methods = ['q','jirp','adrl','hrl']
    methods = ['jirp']
    title = "Office World"

    for i in range(1,exp_number+1):
        csv_filepath = dirpath+"_"+alg+"_"+str(i)+".csv"
        pkl_filepath = dirpath+"_"+alg+"_"+str(i)
        log = Log_experiments()
        # opening the CSV file
        with open(csv_filepath, mode ='r') as file:
            # reading the CSV file
            f = csv.reader(file)
            
            # displaying the contents of the CSV file
            i = 0
            for line in f:
                if i == 0:
                    pass
                elif i <= max_episode:
                    rewards = line[0]
                    success = line[1]
                    steps = line[2]
                    log.log_episode(rewards, success, steps)
                else:
                    break
                i += 1
            log.save_execution(pkl_filepath)
        print("Saved pickle..")
    # print(len(log._episode_data["reward"]))
    # print(len(log._episode_data["success"]))
    # print(len(log._episode_data["episode"]))
    # print(len(log._episode_data["steps"]))

    smoothing = 100

    filepath = dirpath+".png"
    #compare_with_bounds(methods, map, exp_number, 'reward', smoothing)

    compare_with_bounds(methods, map, exp_number, max_episode, 'success', smoothing, title, dirpath, filepath)

    #compare_with_bounds(methods, map, exp_number, 'steps', smoothing)


