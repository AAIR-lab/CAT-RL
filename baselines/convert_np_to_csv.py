import numpy as np
import os
import csv

def convert_np_to_csv(np_filepath, csv_filepath):
    data = np.load(np_filepath)
    data_list = list(zip(data["num_episodes"], data["results"]))
    with open(csv_filepath, "w") as f:
      writer = csv.writer(f)
      writer.writerow(("Num episodes", "Cumulative rewards"))
    i = 10
    with open(csv_filepath, "a") as f:
      writer = csv.writer(f)
      for item in data_list:
        second = "["+",".join(str(x) for x in list(item[1]))+"]"
        new_item = (i, second)
        print(new_item)
        writer.writerow(new_item)
        i += 10

if __name__=="__main__":
    env_name = "officeworld"
    alg = "ppo"
    runid = 10
    train_result_dir = "./results/"+env_name+"/"+alg+"/train/"
    eval_result_dir = "./results/"+env_name+"/"+alg+"/eval/"

    if not os.path.exists(train_result_dir):
      os.makedirs(train_result_dir)
    if not os.path.exists(eval_result_dir):
      os.makedirs(eval_result_dir)
 
    np_filepath = eval_result_dir+"evaluations"+str(runid)+".npz"
    csv_filepath = eval_result_dir+"eval_"+env_name+"_"+alg+"_"+str(runid)+".csv"
    convert_np_to_csv(np_filepath, csv_filepath)
