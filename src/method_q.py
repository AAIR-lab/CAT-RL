import numpy as np
import random 
import time

from src.learning import *
from src.log import Log_experiments
import src.hyper_param as hyper_param

env_alg_to_time = dict()
for trial in range (1,6):
    # ____________ main Parameters ___________________________
    random.seed(23*trial)
    approach_name = 'q'
    map_name = hyper_param.map_name
    file_name = map_name + "_" + approach_name  + "_" + str(trial)
    step_max = hyper_param.step_max
    episodes = hyper_param.episode_max
    env = hyper_param.env
    #_________________________________________________________
    start_time = time.time()

    agent_con_qlearning = qlearning (env, state_size = env._state_size, action_size = env._action_size)
    agent = agent_con_qlearning

    log = Log_experiments()
    agent._acc_reward_data["Num_episodes"] = list()
    agent._acc_reward_data["Cumulative_rewards"] = list()
    for i in range (episodes):
        state = env.reset()
        done = False
        reward = 0
        epoch = 0
        while (not done) and (epoch < step_max):
            env.update_visited(state)
            action = agent.policy (state)
            new_state, r, done, success = env.step (action) 
            agent.train (state, new_state, action, None, r)
            state = new_state
            reward += r
            epoch += 1
        agent.decay()
        log.log_episode(reward, success, epoch)
    
        # print ("_______________________________")
        # print ("episode: " + str(i) + "\t" + "reward: " + str (reward) + "\t" + "epochs: " + str(epoch) 
            #    + "\t" + "epsilon: " + str(round(agent._epsilon,3)) + "\t" + "success: " + str(success))
        if (i % 10 == 0 and i>0):
            total_reward_list = []
            for j in range (10):
                state = env.reset()
                done = False
                total_reward = 0
                epoch = 0
                while (not done) and (epoch < step_max):
                    env.update_visited(state)
                    action = agent.policy_greedy (state)
                    new_state, r, done, success = env.step (action) 
                    state = new_state
                    total_reward += r
                    epoch += 1
                total_reward_list.append(total_reward)
            agent._acc_reward_data["Num_episodes"].append(i)
            agent._acc_reward_data["Cumulative_rewards"].append(total_reward_list)

    env_alg_to_time[file_name] = float(round((time.time() - start_time),2))
    print("Time taken for"+str(file_name)+"(s):"+str(env_alg_to_time[file_name]))

    log.save_execution (file_name)
    log.save_acc_rewards(file_name, agent._acc_reward_data)
    # log.plot_learning(500, "success")

    for filename,timetaken in env_alg_to_time.items():
        print(filename+":"+str(timetaken))
    print("Time mean:",np.mean(list(env_alg_to_time.values())))
    print("Time std:",np.std(list(env_alg_to_time.values())))
