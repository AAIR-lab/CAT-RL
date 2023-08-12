from ssl import cert_time_to_seconds
import numpy as np
import gym
import os
import time
import gc
import torch
import csv
import tqdm
import sys

from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from callback import CustomCallback, CustomStopTrainingCallback
from convert_np_to_csv import convert_np_to_csv
from gym_envs.officeworld import OfficeworldEnv
from gym_envs.taxiworld import TaxiworldEnv
from gym_envs.gridworld import GridworldEnv
# from gym_envs.waterworld import WaterworldEnv
from gym_envs.waterworldnew import WaterworldEnv


def make_env(env): 
  def _init():
    return env
  return _init

# runid = sys.argv[1]
# grid = sys.argv[1]
torch.cuda.set_device("cuda:1")

def create_subproc_envs(env_name, num):
  envs = []
  eval_env = None
  if env_name == "gridworld":
          episode_max = 5000
          step_max = 1200
          total_timesteps = episode_max * step_max
          epsilon_min = 0.05
          decay = 0.9991
          alpha = 0.05
          gamma = 0.95
          for i in range(num):
            envs.append(make_env(GridworldEnv("./gym_envs/map_1.map",(0,0),(63,63),step_max)))
          # eval_env = GridworldEnv("./gym_envs/map_2.map",(0,0),(63,63),step_max)
  elif env_name == "taxiworld":
          episode_max = 20000
          step_max = 1500
          total_timesteps = episode_max * step_max
          epsilon_min = 0.05
          decay = 0.999
          alpha = 0.05
          gamma = 1.0
          for i in range(num):
            envs.append(make_env(TaxiworldEnv((30,30),(0,0),(29,29),step_max)))
          # eval_env = TaxiworldEnv((30,30),(0,0),(29,29),step_max)
  elif env_name == "officeworld":
          episode_max = 3000
          step_max = 1000
          total_timesteps = episode_max * step_max
          epsilon_min = 0.05
          decay = 0.9992
          alpha = 0.05
          gamma = 0.99
          for i in range(num):
            envs.append(make_env(OfficeworldEnv(step_max)))
          # eval_env = OfficeworldEnv(step_max)
  elif env_name == "waterworld":
          episode_max = 10000
          step_max = 100
          if gridsize == (200,200):
            step_max = 100
          elif gridsize == (250,250):
            step_max = 100
          elif gridsize == (300,300):
            step_max = 100
          elif gridsize == (350,350):
            step_max = 100
          elif gridsize == (400,400):
            step_max = 100
          total_timesteps = episode_max * step_max
          epsilon_min = 0.05
          decay = 0.999
          alpha = 0.05
          gamma = 0.95
          # types = {'good': [1,1], 'bad': [1,1]}
          # for i in range(num):
          #   envs.append(make_env(WaterworldEnv(types, step_max)))
          for i in range(num):
            envs.append(make_env(WaterworldEnv(step_max, gridsize)))
          # eval_env = WaterworldEnv(step_max, prob)
  # print(envs)
  # check_env(envs[0]())
  # return SubprocVecEnv(envs, start_method = "spawn"), total_timesteps, gamma, episode_max
  return SubprocVecEnv(envs, start_method = "spawn"), eval_env, total_timesteps, gamma, episode_max
  # return envs[0](), None, total_timesteps, gamma, episode_max

gc.collect()
torch.cuda.empty_cache()

def train(model, env_name, call_backs, total_timesteps):
  model.learn(total_timesteps, callback=call_backs)
  model.save(env_name)

if __name__=="__main__":
    # env_names = ["gridworld", "taxiworld", "officeworld"]
    # algorithms = ["dqn", "a2c", "ppo"]
    env_names = ["waterworld"]
    algorithms = ["dqn"]
    # exps = [str(runid)]
    exps = ["1","2","3","4","5"]
    num_subproc = 1
    verbose = 0
    gridsizes = [(200, 200), (250, 250), (300, 300), (350, 350), (400, 400)]
    # gridsize = gridsizes[int(grid)]
    gridsize = (200, 200)
    max_time = 10*60*60
    map_names = {"gridworld": "grid_64x64_map1", "taxiworld": "taxi_30x30_map1", "officeworld": "office_36x36_map1", "waterworld": "water_"+str(gridsize[0])+"x"+str(gridsize[1])+"_map1"}
    env_alg_to_time = dict()

    try:
      for env_name in env_names:
        for exp in exps:
          for alg in algorithms:
            start_time = time.time()
            train_result_dir = "./results/"+env_name+"/"+alg+"/train/"
            eval_result_dir = "./results/"+env_name+"/"+alg+"/eval/"
            train_result_file = train_result_dir + map_names[env_name]+"_"+alg+"_"+str(exp)+".csv"
            if not os.path.exists(train_result_dir):
              os.makedirs(train_result_dir)
            if not os.path.exists(eval_result_dir):
              os.makedirs(eval_result_dir)

            envs, eval_env, total_timesteps, gamma, episode_max = create_subproc_envs(env_name,num_subproc)
            # envs, eval_env, total_timesteps, gamma, episode_max = create_subproc_envs(env_name,num_subproc)
            if alg == "dqn":
              model = DQN('MlpPolicy', envs, verbose=verbose, gamma=gamma, exploration_fraction=1.0)
            elif alg == "a2c":
              model = A2C('MlpPolicy', envs, verbose=verbose, gamma=gamma)
            elif alg == "ppo":
              model = PPO('MlpPolicy', envs, verbose=verbose, gamma=gamma)

            custom_callback = CustomCallback(train_result_file)
            # eval_callback = EvalCallback(eval_env,n_eval_episodes=100,runid=runid,eval_freq=10,log_path = eval_result_dir)
            new_callback = CustomStopTrainingCallback(episode_max,start_time,max_time)
            # call_backs = CallbackList([custom_callback, eval_callback, new_callback])
            call_backs = CallbackList([custom_callback, new_callback])
            train(model, env_name, call_backs, total_timesteps)

            # np_filepath = eval_result_dir+"evaluations.npz"
            # csv_filepath = eval_result_dir+"eval_"+env_name+"_"+alg+".csv"
            # convert_np_to_csv(np_filepath, csv_filepath)

            name = env_name+"_"+alg
            if name not in env_alg_to_time:
                env_alg_to_time[name] = []
            env_alg_to_time[name].append(round((time.time() - start_time),2))
            print("Time taken for "+name+"(s): "+str(env_alg_to_time[name]))
            with open(train_result_dir+name+".txt","w") as f:
              f.write(str(name)+"; "+str(env_alg_to_time[name]))

      print("\n\nTime taken (s):")
      for env_name in env_names:
        for alg in algorithms:
          name = env_name+"_"+alg
          print(name+" : "+str(env_alg_to_time[name]))
    except KeyboardInterrupt:
      torch.cuda.empty_cache()
    finally:
      torch.cuda.empty_cache()