import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
import csv

t = False

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


def moving_average (y, moving_number):
    y_m = []
    for i in range (moving_number, len(y)):
        sum_temp = 0
        for j in range (i - moving_number, i):
            sum_temp += int(y[j])
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
    # print(data[0]["steps"])
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

def compare_with_bounds(ax,methods, map_name, exp_number, max_episode, param, moving_number, title, dirpath, colors):
    lines = []
    for m in methods:
        print("Method:",m)
        data_temp = read_data (map_name, m, exp_number, dirpath)
        # only using data until max_episode
        data_temp[0][param] = data_temp[0][param][:max_episode]
        plot_data_smooth = smooth_data(data_temp, moving_number, param)
        plot_data = prepare_avg_bound (param, plot_data_smooth)
        episodes = gen_episode_ax(moving_number, len(plot_data[1]))
        lines.append(ax.plot(episodes, plot_data[1], color=colors[m][0], linestyle='solid', linewidth = 1, label = m)[0])
        # temp = []
        # for x in plot_data[0]:
        #     if x>0:
        #         temp.append(x)
        #     else:
        #         temp.append(0)
        ax.fill_between(episodes, plot_data[0], plot_data[2], alpha=0.25, edgecolor=colors[m][1], facecolor=colors[m][1],linewidth=0)
    ax.set_ylim(0,1.1)

    # y_ticks = [-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2]
    # y_ticks = [-0.2,0.0,0.2,0.4,0.6,0.8,1.0]
    y_ticks = []
    step = 0.2 
    start = 0 
    end = 1.1
    while start < end:
        y_ticks.append(int(start*10)/10.0)
        start += step

    y_ticklabels1 = [str(i) for i in y_ticks]
    y_ticklabels2 = ["" for i in y_ticks]
    ax.set_yticks(y_ticks)
    global t
    if not t: 
        ax.set_yticklabels(y_ticklabels1)
        t = True
    else:
        ax.set_yticklabels(y_ticklabels2)

    # ax.set_xlabel("Episodes")
    # ax.set_ylabel(param)
    ax.set_title(title)
    # plt.legend()
    # # plt.show()
    # plt.savefig(filepath)
    return lines

def draw(ax,domain_name,param,max_episode,map,title,methods,colors,exp_number):
    dirpath = "./"+domain_name+"/"+map
    smoothing = 100
    return compare_with_bounds(ax,methods, map, exp_number, max_episode, param, smoothing, title, dirpath, colors),methods

if __name__=="__main__":
    # domain_names = ["gridworld","officeworld","taxiworld"]
    domain_names = ["officeworld","gridworld","taxiworld","waterworld","mountaincar"]
    # domain_names = ["officeworld","gridworld","mountaincar"]
    titles = {"taxiworld": "Discrete \nTaxi World (30x30)", "gridworld": "Discrete\n Wumpus World (64x64)", "officeworld": "Discrete \nOffice World (36x36)", "waterworld": "Continuous \n Water World", "mountaincar": "Continuous \nMountain Car "}
    max_episodes = {"taxiworld": 20000, "gridworld": 5000, "officeworld": 3000, "waterworld": 10000, "mountaincar": 2000}
    map_names = {"taxiworld": "taxi_30x30_map1", "gridworld": "grid_64x64_map1", "officeworld": "office_36x36_map1", "waterworld": "water_300x300_map1", "mountaincar": "mountaincar"}
    methods = ['adrl','q','jirp','hrl',"dqn","a2c","ppo"]
    methods_office = ['adrl','q','jirp','hrl',"dqn","a2c","ppo"]
    methods_water = ['adrl',"dqn","a2c","ppo"]
    # colors = {'adrl': ['#45AA99','#D0E9E5'], 'q': ['#332288','#CBC7E1'], 'hrl': ['#CD6A7B','#F2D8DC'], 'jirp': ['#b38c32','#f7f0df'], 'dqn': ['#b38c32','#f7f0df'], 'ppo': ['#b38c32','#f7f0df']}
    colors = {'adrl': ['tab:green','tab:green'], 'q': ['tab:purple','tab:purple'], 'hrl': ['tab:cyan','tab:cyan'], 'jirp': ['tab:orange','tab:orange'], 'dqn': ['tab:red','tab:red'], 'a2c': ['tab:grey', 'tab:grey'], 'ppo': ['tab:blue','tab:blue']}
    exp_numbers = {"officeworld":9,"gridworld":10,"taxiworld":10,"waterworld":10,"mountaincar":9}
    param = "success" # rewards

    fig,ax = plt.subplots(1,5,figsize =(10,1.75))
    legends = [False,False,True]
    for i,domain_name in enumerate(domain_names):
        print("\nDomain:",domain_name)
        if domain_name == "waterworld":
            methods = methods_water
        if domain_name == "mountaincar":
            methods = methods_water
        if domain_name == "officeworld":
            methods = methods_office
        lines,methods = draw(ax[i],domain_name,param,max_episodes[domain_name],map_names[domain_name],titles[domain_name],methods,colors,exp_numbers[domain_name])
        if domain_name=="gridworld":
            lines_legend = lines
            methods_legend = methods

        # if domain_name == "gridworld":
        #     x_ticks = list(range(0,5001,1000))
        #     ax[i].set_xticks(x_ticks)    
        if domain_name=="gridworld":
            x_ticks = [0,1600,3200,5000]
            ax[i].set_xticks(x_ticks)
        if domain_name=="taxiworld":
            # x_ticks = [10,2500,5000,7500,10000,12500,15000,17500,20000]
            x_ticks = [10,7000,13000,20000]
            ax[i].set_xticks(x_ticks)
        if domain_name=="waterworld":
            x_ticks = [0,3000,6000,10000]
            ax[i].set_xticks(x_ticks)
        if domain_name=="mountaincar":
            x_ticks = [0,600,1300,2000]
            ax[i].set_xticks(x_ticks)    

        xticks = ax[i].get_xticks()
        new_xticks = []
        for s in xticks:
            if len(str(int(s))) > 3:
                new_xticks.append(str(int(s))[:-3]+"K")
            else:
                new_xticks.append(str(int(s)))
        ax[i].set_xticklabels(new_xticks)
        ax[i].set_ylim(-0.02,1.05)

    method_names = ["CAT+RL (ours)", "Q-learning", "JIRP", "Option-critic", "DQN", "A2C", "PPO"]
    fig.legend(lines_legend, method_names, loc='upper center',ncol = 7,bbox_to_anchor = (0.5,1.17))
    # fig.text(0.5,-0.05, 'Training episodes', ha = 'center')
    fig.text(0.08,0.5, "Success rate", va = "center",rotation = "vertical")
    # fig.legend(lines, methods, loc='upper center',ncol =3,bbox_to_anchor=(1,1))
    plt.subplots_adjust(wspace = 0.05,hspace = 0.25)
    # plt.tight_layout(pad=1,h_pad=1)
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
    # plt.show()


    filepath = "plot_"+param+"_.png"
    plt.savefig(filepath,dpi=300,bbox_inches='tight')
    filepath = "plot_"+param+"_.svg"
    plt.savefig(filepath,dpi=300,bbox_inches='tight')
    filepath = "plot_"+param+"_.pdf"
    plt.savefig(filepath,dpi=300,bbox_inches='tight')

