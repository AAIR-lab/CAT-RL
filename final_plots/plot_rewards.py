
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
plt.rc('figure', titlesize=MEDIUM_SIZE)

def moving_average (y, moving_number):
    y_m = []
    for i in range (moving_number, len(y)):
        sum_temp = 0
        for j in range (i - moving_number, i):
            sum_temp += int(y[j])
        sum_temp /= moving_number
        y_m.append(sum_temp)
    return y_m

if __name__=="__main__":
    # domain_names = ["gridworld","officeworld","taxiworld","waterworld"]
    # domain_names = ["officeworld","gridworld","taxiworld"]
    domain_names = ["officeworld","gridworld","taxiworld","waterworld","mountaincar"]
    titles = {"taxiworld": "Taxi World", "gridworld": "Wumpus World", "officeworld": "Office World", "waterworld": "Water World", "mountaincar": "mountaincar"}
    max_episodes = {"taxiworld": 20000, "gridworld": 10000, "officeworld": 3000, "waterworld": 10000, "mountaincar":2000}
    map_names = {"taxiworld": "taxi_30x30_map1", "gridworld": "grid_64x64_map1", "officeworld": "office_36x36_map1", "waterworld": "water_300x300_map1", "mountaincar": "mountaincar_map1"}

    fig,ax = plt.subplots(1,5,figsize =(10,1.75))
    legends = [False,False,True]

    domains_to_methods = {"officeworld":['ppo','adrl'], "gridworld":['q','adrl'], "taxiworld": ['q','adrl'], "waterworld": ['ppo','adrl'], "mountaincar": ['dqn','adrl']}
    method_to_names = {"adrl":"CA+RL (ours)", "q":"Q-learning", "ppo":"PPO", "dqn":"DQN"}
    
    dirpath = "./results/acc_rewards/"
    interval_episodes = 10
    colors = {'adrl': ['tab:green','tab:green'], 'q': ['tab:purple','tab:purple'], 'hrl': ['tab:cyan','tab:cyan'], 'jirp': ['tab:orange','tab:orange'], 'dqn': ['tab:red','tab:red'], 'a2c': ['tab:grey', 'tab:grey'], 'ppo': ['tab:blue','tab:blue']}

    for i,domain_name in enumerate(domain_names):
        print("\n",domain_name)
        lines = []
        methods = domains_to_methods[domain_name]
        map_name = map_names[domain_name]
        file_q = dirpath + map_name+"_"+methods[0]+"_1.csv"
        file_a = dirpath + map_name+"_"+methods[1]+"_1.csv"

        x = list()
        q_mean = list()
        q_std = list()
        a_mean = list()
        a_std = list()

        with open(file_q,"r") as f:
            csvreader = csv.reader(f, delimiter=',')
            next(csvreader)
            for j,item in enumerate(csvreader):
                # if domain_name == "taxiworld":
                #     if j < 10:
                #         continue
                # x.append(int(item[0]))
                if int(item[0]) <= max_episodes[domain_name]:
                    list_ = [float(i) for i in list(item[1][1:-1].split(","))]
                    q_mean.append(np.mean(list_))
                    q_std.append(np.std(list_)/np.linalg.norm(list_))
            # if domain_name=="taxiworld":
            #     for k in range(1000):
            #         q_mean.append(0)
            #         q_std.append(0)

        with open(file_a,"r") as f:
            csvreader = csv.reader(f, delimiter=',')
            next(csvreader)
            for item in csvreader:
                if int(item[0]) <= max_episodes[domain_name]:
                    x.append(int(item[0]))
                    list_ = [float(i) for i in list(item[1][1:-1].split(","))]
                    a_mean.append(np.mean(list_))
                    a_std.append(np.std(list_)/np.linalg.norm(list_))

        q_mean = np.array(q_mean)
        q_std = np.array(q_std)
        a_mean = np.array(a_mean)
        a_std = np.array(a_std)


        print(len(x))
        print(len(q_mean))
        print(len(a_mean))

        if domain_name=="taxiworld":
            q_mean = q_mean[1:]
            a_mean = a_mean[1:]
            x = x[1:]

        interval = 1
        x = x[::interval]
        q_mean = q_mean[::interval]
        # q_std = q_std[::interval]
        a_mean = a_mean[::interval]
        # a_std = a_std[::interval]

        moving_avg = 5
        x = moving_average(x, moving_avg)
        q_mean = moving_average(q_mean, moving_avg)
        a_mean = moving_average(a_mean, moving_avg)

        # if domain_name=="taxiworld":
        #     a_mean = (np.array(a_mean)-np.min(q_mean))/(np.max(q_mean)-np.min(q_mean))
        #     q_mean = (np.array(q_mean)-np.min(q_mean))/(np.max(q_mean)-np.min(q_mean))
        # else:
        #     q_mean = (np.array(q_mean)-np.min(a_mean))/(np.max(a_mean)-np.min(a_mean))
        #     a_mean = (np.array(a_mean)-np.min(a_mean))/(np.max(a_mean)-np.min(a_mean))

        print("q:", np.min(q_mean), ",", np.max(q_mean))
        print("adrl:", np.min(a_mean), ",", np.max(a_mean))
        # minm = min()
        
        q_mean = (np.array(q_mean)-np.min(a_mean))/(np.max(a_mean)-np.min(a_mean))
        a_mean = (np.array(a_mean)-np.min(a_mean))/(np.max(a_mean)-np.min(a_mean))

        q_mean = q_mean[:max_episodes[domain_name]]
        a_mean = a_mean[:max_episodes[domain_name]]
        x = x[:max_episodes[domain_name]]

        # if domain_name=="taxiworld":
        #     start, end = 0//(10*interval), 10000//(10*interval)
        #     x = x[start:end]
        #     q_mean = q_mean[start:end]
        #     q_std = q_std[start:end]
        #     a_mean = a_mean[start:end]
        #     a_std = a_std[start:end]

        print(len(x),len(q_mean))
        lines.append(ax[i].plot(x, q_mean, color=colors[methods[0]][0], linestyle='solid', linewidth = 1)[0])
        # ax[i].fill_between(x, q_mean - q_std, q_mean + q_std, alpha=0.9, edgecolor=colors[methods[0]][1], facecolor=colors[methods[0]][1],linewidth=0)   
        lines.append(ax[i].plot(x, a_mean, color=colors[methods[1]][0], linestyle='solid', linewidth = 1)[0])
        # ax[i].fill_between(x, a_mean - a_std, a_mean + a_std, alpha=0.9, edgecolor=colors[methods[1]][1], facecolor=colors[methods[1]][1],linewidth=0)
      
        y_ticks = [0.0,0.2,0.4,0.6,0.8,1.0]
        y_ticklabels1 = [str(i) for i in y_ticks]
        y_ticklabels2 = ["" for i in y_ticks]
        ax[i].set_yticks(y_ticks)
        ax[i].set_ylim(0.0,1.05)
        if not t: 
            ax[i].set_yticklabels(y_ticklabels1)
            t = True
        else:
            ax[i].set_yticklabels(y_ticklabels2)

        if domain_name=="gridworld":
            x_ticks = [0,3000,6000,10000]
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
        
        # ax[i].set_title(titles[domain_name])
        # ax[i].set_yscale('log')

    method_names = [method_to_names[methods[0]], method_to_names[methods[1]]]
    # fig.legend(lines, method_names, loc='upper center',ncol=4,bbox_to_anchor = (0.5,1.1))
    fig.text(0.5,-0.08, 'Episodes', ha = 'center')
    fig.text(0.06,0.48, "       Normalized \n Cumulative Reward", va = "center",rotation = "vertical")
    # fig.legend(lines, methods, loc='upper center',ncol =3,bbox_to_anchor=(1,1))
    plt.subplots_adjust(wspace = 0.05,hspace = 0.3)
    # plt.tight_layout(pad=1,h_pad=1)

    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
    filepath = "plot_reward_.png"
    plt.savefig(filepath,dpi=300,bbox_inches='tight')
    filepath = "plot_reward_.svg"
    plt.savefig(filepath,dpi=300,bbox_inches='tight')
    filepath = "plot_reward_.pdf"
    plt.savefig(filepath,dpi=300,bbox_inches='tight')









